from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import tlm.ranking_excess_supervised as supervised_module
from tlm.patch_transformer import MultiAssetPatchTransformer
from tlm.ranking_excess_harness import RANKING_EXCESS_HEADS
from tlm.ranking_excess_supervised import (
    SupervisedEarlyStopping,
    _configure_device,
    _load_resume,
    _run_supervised_batches,
    _save_resume,
    configure_supervised_scope,
    fit_fold_excess_scale,
    read_fold_supervised_data,
    supervised_parameter_names,
)
from tlm.supervised_non_target import model_state_sha256


def _medium_architecture() -> dict:
    return json.loads(
        Path("artifacts/v41_ranking_excess_spec/blueprint.json").read_text(
            encoding="utf-8"
        )
    )["architecture"]


def _tiny_architecture() -> dict:
    return {
        "lookback_days": 8,
        "input_triplet_size": 3,
        "patch_length_days": 4,
        "patch_stride_days": 2,
        "d_model": 16,
        "attention_heads": 4,
        "encoder_layers": 1,
        "cross_asset_attention_layers": 1,
        "feed_forward_width": 32,
        "dropout": 0.0,
        "prediction_heads": list(RANKING_EXCESS_HEADS),
    }


def _data_access() -> dict:
    features = [f"feature_{index}" for index in range(8)]
    return {
        "feature_columns": ["date", "symbol", *features],
        "label_columns": [
            "date",
            "symbol",
            "target_window_end_date",
            "target_next_open_to_next_open_log_return",
            "target_realized_volatility_7d",
        ],
        "sequence_columns": ["date", "sequence_start_date", "symbol"],
        "feature_start": "2021-01-01",
        "feature_end": "2024-12-23",
        "supervised_train_start": "2023-01-01",
        "supervised_train_end": "2023-01-01",
        "supervised_train_maturity_end": "2023-01-09",
        "validation_start": "2024-01-01",
        "validation_end": "2024-01-01",
        "validation_maturity_end": "2024-01-09",
        "expected_by_fold": {
            "1": {
                "feature_rows": 20,
                "train_label_rows": 20,
                "validation_label_rows": 20,
                "train_sequence_rows": 20,
                "validation_sequence_rows": 20,
                "train_eligible_pairs": 1140,
                "validation_eligible_pairs": 1140,
            }
        },
    }


def _loss_record(
    *, ranking: float, excess: float, volatility: float, observations: int
) -> dict[str, float | int]:
    core = ranking + excess
    return {
        "ranking": ranking,
        "excess": excess,
        "log_volatility": volatility,
        "core": core,
        "total": core + 0.1 * volatility,
        "pair_count": observations * 3,
        "observations": observations,
    }


def test_medium_supervised_scope_and_parameter_counts_are_exact() -> None:
    model = MultiAssetPatchTransformer(
        9,
        _medium_architecture(),
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    parameters = configure_supervised_scope(model)
    names = supervised_parameter_names(model)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_231_634
    assert sum(parameter.numel() for parameter in parameters) == 1_212_930
    assert sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name not in names
    ) == 18_704
    assert any(name.startswith("cross_asset_encoder.") for name in names)
    assert any(name.startswith("prediction_heads.") for name in names)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(("mask_token", "reconstruction_head."))
    )


def test_supervised_early_stopping_monitors_strict_core_improvement() -> None:
    state = SupervisedEarlyStopping(patience=2)
    assert not state.update(1, 1.0)
    assert not state.update(2, 1.0)
    assert state.update(3, 1.1)
    assert not state.update(4, 0.9)
    assert state.best_epoch == 4
    assert state.stale_epochs == 0


def test_fold_excess_scale_enumerates_every_lexical_train_triplet() -> None:
    date = pd.Timestamp("2023-01-01", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    returns = {"A": 0.03, "B": 0.01, "C": -0.02, "D": 0.00}
    labels = pd.DataFrame({
        "date": [date] * len(symbols),
        "symbol": symbols,
        "target_next_open_to_next_open_log_return": [
            returns[symbol] for symbol in symbols
        ],
    })
    result = fit_fold_excess_scale(labels, {date: symbols}, 1e-6)
    triplets = np.asarray([
        [returns[symbol] for symbol in triplet]
        for triplet in combinations(symbols, 3)
    ])
    excess = triplets - triplets.mean(axis=1, keepdims=True)
    assert result["enumerated_triplets"] == 4
    assert result["enumerated_excess_values"] == 12
    assert result["excess_rms_scale"] == pytest.approx(
        float(np.sqrt(np.mean(np.square(excess))))
    )


def test_fold_reader_uses_five_separate_projected_filtered_reads() -> None:
    access = _data_access()
    train_symbols = [f"A{index:02d}USDT" for index in range(20)]
    test_symbols = [f"H{index:02d}USDT" for index in range(10)]
    train_date = pd.Timestamp("2023-01-01", tz="UTC")
    validation_date = pd.Timestamp("2024-01-01", tz="UTC")
    feature_date = pd.Timestamp("2021-01-01", tz="UTC")
    feature_rows = pd.DataFrame([
        {
            "date": feature_date,
            "symbol": symbol,
            **{name: float(index + 1) for index, name in enumerate(access["feature_columns"][2:])},
        }
        for symbol in train_symbols
    ], columns=access["feature_columns"])

    def labels(date: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "date": date,
                "symbol": symbol,
                "target_window_end_date": date + pd.Timedelta(days=8),
                "target_next_open_to_next_open_log_return": index / 10_000,
                "target_realized_volatility_7d": 0.02 + index / 10_000,
            }
            for index, symbol in enumerate(train_symbols)
        ], columns=access["label_columns"])

    def sequence(date: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "date": date,
                "sequence_start_date": date - pd.Timedelta(days=255),
                "symbol": symbol,
            }
            for symbol in train_symbols
        ], columns=access["sequence_columns"])

    calls: list[dict[str, object]] = []

    def reader(path, *, engine, columns, filters):
        calls.append({
            "path": str(path),
            "engine": engine,
            "columns": list(columns),
            "filters": filters,
        })
        is_validation = any(
            column == "in_validation" for column, _, _ in filters
        )
        if str(path).endswith("sequence.parquet"):
            return sequence(validation_date if is_validation else train_date)
        if columns == access["feature_columns"]:
            return feature_rows.copy()
        return labels(validation_date if is_validation else train_date)

    result = read_fold_supervised_data(
        Path("panel.parquet"),
        Path("sequence.parquet"),
        {"fold": 1, "train_symbols": train_symbols, "test_symbols": test_symbols},
        access,
        reader=reader,
    )
    assert len(calls) == 5
    assert all(call["engine"] == "pyarrow" for call in calls)
    assert [call["columns"] for call in calls] == [
        access["feature_columns"],
        access["label_columns"],
        access["label_columns"],
        access["sequence_columns"],
        access["sequence_columns"],
    ]
    assert result.audit["heldout_symbols_materialized"] == []
    assert result.audit["train_eligible_pairs"] == 1140


def test_fold_reader_fails_if_backend_ignores_symbol_filter() -> None:
    access = _data_access()
    train_symbols = [f"A{index:02d}USDT" for index in range(20)]
    test_symbols = [f"H{index:02d}USDT" for index in range(10)]

    def bad_reader(_path, *, engine, columns, filters):
        del engine, filters
        rows = [{"date": pd.Timestamp("2023-01-01", tz="UTC"), "symbol": symbol}
                for symbol in [*train_symbols, test_symbols[0]]]
        return pd.DataFrame(rows).reindex(columns=columns)

    with pytest.raises(RuntimeError, match="ignored the fold symbol filter"):
        read_fold_supervised_data(
            Path("panel.parquet"),
            Path("sequence.parquet"),
            {
                "fold": 1,
                "train_symbols": train_symbols,
                "test_symbols": test_symbols,
            },
            access,
            reader=bad_reader,
        )


def test_epoch_ranking_loss_is_weighted_by_active_pair_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model(nn.Module):
        def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
            batch = len(value)
            return {
                "excess_return_z": torch.zeros(batch, 3),
                "log_volatility_7d": torch.zeros(batch, 3),
            }

    class Store:
        def materialize_batch(self, samples, _scaler):
            batch = len(samples)
            return (
                np.zeros((batch, 1), dtype=np.float32),
                np.zeros((batch, 3, 2), dtype=np.float32),
            )

    def fake_loss(_output, labels, _scale, _objective):
        if len(labels) == 2:
            ranking, pairs = 1.0, 6
        else:
            ranking, pairs = 10.0, 1
        return {
            "ranking": torch.tensor(ranking),
            "excess": torch.tensor(2.0),
            "log_volatility": torch.tensor(3.0),
            "core": torch.tensor(ranking + 2.0),
            "total": torch.tensor(ranking + 2.3),
            "pair_count": torch.tensor(pairs),
        }

    monkeypatch.setattr(supervised_module, "_validated_loss", fake_loss)
    losses, steps = _run_supervised_batches(
        Model(),
        Store(),
        scaler=object(),
        samples=[{"i": 0}, {"i": 1}, {"i": 2}],
        batch_size=2,
        target_scale=1.0,
        objective={"log_volatility_weight": 0.1},
        device=torch.device("cpu"),
        optimizer=None,
        gradient_clip_norm=1.0,
    )
    assert steps == 2
    assert losses["ranking"] == pytest.approx((1.0 * 6 + 10.0) / 7)
    assert losses["excess"] == pytest.approx(2.0)
    assert losses["observations"] == 3
    assert losses["pair_count"] == 7


def test_resume_roundtrip_and_history_hash_tamper_rejection(
    tmp_path: Path,
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(17)
    model = MultiAssetPatchTransformer(
        9,
        _tiny_architecture(),
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    parameters = configure_supervised_scope(model)
    optimizer = torch.optim.AdamW(parameters, lr=0.001)
    sum(parameter.square().mean() for parameter in parameters).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    early = SupervisedEarlyStopping(patience=3)
    early.update(1, 0.7)
    history = [{
        "epoch": 1,
        "train_losses": _loss_record(
            ranking=0.3, excess=0.4, volatility=0.5, observations=2
        ),
        "validation_losses": _loss_record(
            ranking=0.3, excess=0.4, volatility=0.5, observations=2
        ),
        "improved": True,
        "train_optimizer_steps": 1,
        "validation_steps": 1,
    }]
    metadata = {"fold": 1, "seed": 17}
    best_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_model_hash = model_state_sha256(model.state_dict())
    path = tmp_path / "resume.pt"
    _save_resume(
        path,
        model=model,
        optimizer=optimizer,
        early_stopping=early,
        history=history,
        completed_epoch=1,
        metadata=metadata,
        best_model_state=best_state,
        device=device,
        format_version="v44_test_resume",
        best_state_format="v44_test_best",
        optimizer_parameter_names=supervised_parameter_names(model),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    completed, restored_early, restored_history, _ = _load_resume(
        path,
        model=model,
        optimizer=optimizer,
        expected_metadata=metadata,
        device=device,
        format_version="v44_test_resume",
        expected_best_state_format="v44_test_best",
        expected_patience=3,
        maximum_epochs=3,
        expected_optimizer_parameter_names=supervised_parameter_names(model),
        expected_train_observations=2,
        expected_validation_observations=2,
        expected_train_steps=1,
        expected_validation_steps=1,
        expected_volatility_weight=0.1,
    )
    assert completed == 1
    assert asdict(restored_early) == asdict(early)
    assert restored_history == history
    assert model_state_sha256(model.state_dict()) == expected_model_hash

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["history"][0]["validation_losses"]["core"] = 99.0
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="history hash drift"):
        _load_resume(
            path,
            model=model,
            optimizer=optimizer,
            expected_metadata=metadata,
            device=device,
            format_version="v44_test_resume",
            expected_best_state_format="v44_test_best",
            expected_patience=3,
            maximum_epochs=3,
            expected_optimizer_parameter_names=supervised_parameter_names(model),
            expected_train_observations=2,
            expected_validation_observations=2,
            expected_train_steps=1,
            expected_validation_steps=1,
            expected_volatility_weight=0.1,
        )


def test_mps_fallback_environment_is_rejected_before_device_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="forbids PYTORCH_ENABLE_MPS_FALLBACK"):
        _configure_device("mps", 1)
