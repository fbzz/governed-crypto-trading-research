from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import tlm.ranking_excess_pretraining as pretraining_module
from tlm.patch_transformer import MultiAssetPatchTransformer
from tlm.non_target_pretraining import TripletTensorStore
from tlm.ranking_excess_harness import RANKING_EXCESS_HEADS
from tlm.ranking_excess_pretraining import (
    PretrainingEarlyStopping,
    _load_resume,
    _load_prior_gate,
    _metadata_context,
    _register_forbidden_forward_guards,
    _run_reconstruction_batches,
    _save_resume,
    _sha256_file,
    _train_job,
    build_pretraining_spec,
    configure_pretraining_scope,
    load_ranking_excess_pretrained_checkpoint,
    pretraining_parameter_names,
    read_fold_feature_data,
)
from tlm.scientific_harness import FeatureScaler
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
        "panel_columns": ["date", "symbol", *features],
        "sequence_columns": ["date", "sequence_start_date", "symbol"],
        "representation_train_start": "2021-01-01",
        "representation_train_end": "2023-12-31",
        "feature_only_validation_start": "2024-01-01",
        "feature_only_validation_end": "2024-12-23",
        "maximum_loaded_date": "2024-12-23",
        "expected_by_fold": {
            "1": {
                "panel_rows": 40,
                "scaler_finite_train_rows": 20,
                "train_sequence_rows": 20,
                "validation_sequence_rows": 20,
                "train_eligible_pairs": 1140,
                "validation_eligible_pairs": 1140,
            }
        },
    }


def test_medium_pretraining_scope_is_exact_and_excludes_supervised_modules() -> None:
    model = MultiAssetPatchTransformer(
        9,
        _medium_architecture(),
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    parameters = configure_pretraining_scope(model)
    names = pretraining_parameter_names(model)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_231_634
    assert sum(parameter.numel() for parameter in parameters) == 834_576
    assert len(names) == 56
    assert not any(name.startswith("cross_asset_encoder.") for name in names)
    assert not any(name.startswith("prediction_heads.") for name in names)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(("cross_asset_encoder.", "prediction_heads."))
    )


def test_forbidden_forward_guards_allow_reconstruction_but_reject_full_model() -> None:
    model = MultiAssetPatchTransformer(
        9,
        _tiny_architecture(),
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    x = torch.randn(2, 8, 3, 9)
    mask = torch.zeros(2, 3, model.patch_count, dtype=torch.bool)
    mask[:, :, 1] = True
    handles = _register_forbidden_forward_guards(model)
    try:
        reconstruction = model.reconstruct_masked_patches(x, mask)
        assert reconstruction.shape == (2, 3, 3, 4, 9)
        with pytest.raises(RuntimeError, match="forbidden"):
            model(x)
    finally:
        for handle in handles:
            handle.remove()


def test_early_stopping_resets_sticky_stop_after_real_improvement() -> None:
    state = PretrainingEarlyStopping(patience=2)
    assert not state.update(1, 1.0)
    assert not state.update(2, 1.1)
    assert state.update(3, 1.2)
    assert not state.update(4, 0.9)
    assert state.stale_epochs == 0
    assert not state.should_stop


def test_resume_roundtrip_restores_best_state_optimizer_and_cpu_rng(
    tmp_path: Path,
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(17)
    model = MultiAssetPatchTransformer(
        9,
        _tiny_architecture(),
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    parameters = configure_pretraining_scope(model)
    optimizer = torch.optim.AdamW(parameters, lr=0.001)
    loss = sum(parameter.square().mean() for parameter in parameters)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    early = PretrainingEarlyStopping(patience=3)
    early.update(1, 0.75)
    metadata = {"fold": 1, "seed": 17}
    history = [
        {"epoch": 1, "validation_loss": 0.75, "train_optimizer_steps": 1}
    ]
    expected_model_hash = model_state_sha256(model.state_dict())
    best_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_rng = torch.get_rng_state().clone()
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
        format_version="v43_test_resume",
        best_state_format="v43_test_best",
        optimizer_parameter_names=pretraining_parameter_names(model),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    torch.manual_seed(999)
    completed, restored_early, restored_history, restored_best = _load_resume(
        path,
        model=model,
        optimizer=optimizer,
        expected_metadata=metadata,
        device=device,
        format_version="v43_test_resume",
        expected_best_state_format="v43_test_best",
        expected_patience=3,
        maximum_epochs=10,
        expected_optimizer_parameter_names=pretraining_parameter_names(model),
    )
    assert completed == 1
    assert asdict(restored_early) == asdict(early)
    assert restored_history == history
    assert model_state_sha256(model.state_dict()) == expected_model_hash
    assert model_state_sha256(restored_best) == model_state_sha256(best_state)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_resume_rejects_early_stopping_patience_drift(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model = MultiAssetPatchTransformer(
        9,
        _tiny_architecture(),
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    parameters = configure_pretraining_scope(model)
    optimizer = torch.optim.AdamW(parameters, lr=0.001)
    loss = sum(parameter.square().mean() for parameter in parameters)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    early = PretrainingEarlyStopping(patience=3)
    early.update(1, 0.75)
    path = tmp_path / "resume.pt"
    _save_resume(
        path,
        model=model,
        optimizer=optimizer,
        early_stopping=early,
        history=[{
            "epoch": 1,
            "validation_loss": 0.75,
            "train_optimizer_steps": 1,
        }],
        completed_epoch=1,
        metadata={"fold": 1, "seed": 17},
        best_model_state=model.state_dict(),
        device=device,
        format_version="v43_test_resume",
        best_state_format="v43_test_best",
        optimizer_parameter_names=pretraining_parameter_names(model),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["early_stopping"]["patience"] = 999
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="early-stopping coherence drift"):
        _load_resume(
            path,
            model=model,
            optimizer=optimizer,
            expected_metadata={"fold": 1, "seed": 17},
            device=device,
            format_version="v43_test_resume",
            expected_best_state_format="v43_test_best",
            expected_patience=3,
            maximum_epochs=10,
            expected_optimizer_parameter_names=pretraining_parameter_names(model),
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("learning_rate", "optimizer hyperparameter/group drift"),
        ("non_finite_moment", "optimizer contains non-finite state"),
        ("finite_moment", "optimizer semantic hash drift"),
        ("step", "optimizer step-state drift"),
    ),
)
def test_resume_rejects_optimizer_drift(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    device = torch.device("cpu")
    model = MultiAssetPatchTransformer(
        9,
        _tiny_architecture(),
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    parameters = configure_pretraining_scope(model)
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.01)
    loss = sum(parameter.square().mean() for parameter in parameters)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    early = PretrainingEarlyStopping(patience=3)
    early.update(1, 0.75)
    path = tmp_path / f"{tamper}.pt"
    _save_resume(
        path,
        model=model,
        optimizer=optimizer,
        early_stopping=early,
        history=[{
            "epoch": 1,
            "validation_loss": 0.75,
            "train_optimizer_steps": 1,
        }],
        completed_epoch=1,
        metadata={"fold": 1, "seed": 17},
        best_model_state=model.state_dict(),
        device=device,
        format_version="v43_test_resume",
        best_state_format="v43_test_best",
        optimizer_parameter_names=pretraining_parameter_names(model),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if tamper == "learning_rate":
        payload["optimizer_state_dict"]["param_groups"][0]["lr"] = 9.0
    elif tamper in {"non_finite_moment", "finite_moment"}:
        first_parameter = next(iter(payload["optimizer_state_dict"]["state"]))
        moment = payload["optimizer_state_dict"]["state"][first_parameter][
            "exp_avg"
        ].flatten()
        moment[0] = float("nan") if tamper == "non_finite_moment" else moment[0] + 123
    else:
        first_parameter = next(iter(payload["optimizer_state_dict"]["state"]))
        payload["optimizer_state_dict"]["state"][first_parameter]["step"] = (
            torch.tensor(999.0)
        )
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match=message):
        _load_resume(
            path,
            model=model,
            optimizer=optimizer,
            expected_metadata={"fold": 1, "seed": 17},
            device=device,
            format_version="v43_test_resume",
            expected_best_state_format="v43_test_best",
            expected_patience=3,
            maximum_epochs=10,
            expected_optimizer_parameter_names=pretraining_parameter_names(model),
        )


def test_fold_reader_pushes_filters_and_materializes_only_train_features() -> None:
    train_symbols = [f"A{index:02d}USDT" for index in range(20)]
    test_symbols = [f"H{index:02d}USDT" for index in range(10)]
    access = _data_access()
    features = access["panel_columns"][2:]
    panel_rows = []
    for symbol in train_symbols:
        for date in (
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2024-12-23", tz="UTC"),
        ):
            panel_rows.append({
                "date": date,
                "symbol": symbol,
                **{feature: 1.0 for feature in features},
            })
    panel = pd.DataFrame(panel_rows, columns=access["panel_columns"])
    train_date = pd.Timestamp("2021-10-13", tz="UTC")
    validation_date = pd.Timestamp("2024-01-01", tz="UTC")
    train_index = pd.DataFrame(
        [
            {
                "date": train_date,
                "sequence_start_date": train_date - pd.Timedelta(days=255),
                "symbol": symbol,
            }
            for symbol in train_symbols
        ],
        columns=access["sequence_columns"],
    )
    validation_index = pd.DataFrame(
        [
            {
                "date": validation_date,
                "sequence_start_date": validation_date - pd.Timedelta(days=255),
                "symbol": symbol,
            }
            for symbol in train_symbols
        ],
        columns=access["sequence_columns"],
    )
    calls = []

    def reader(_path, *, engine, columns, filters):
        calls.append({"engine": engine, "columns": columns, "filters": filters})
        if any(item[0] == "in_representation_train" for item in filters):
            return train_index.copy()
        if any(item[0] == "in_validation" for item in filters):
            return validation_index.copy()
        return panel.copy()

    result = read_fold_feature_data(
        Path("panel.parquet"),
        Path("sequence.parquet"),
        {
            "fold": 1,
            "train_symbols": train_symbols,
            "test_symbols": test_symbols,
        },
        access,
        reader=reader,
    )
    assert len(calls) == 3
    assert all(call["engine"] == "pyarrow" for call in calls)
    assert calls[0]["columns"] == access["panel_columns"]
    assert not any(
        column.startswith(("target_", "label_"))
        for call in calls
        for column in call["columns"]
    )
    assert result.audit["heldout_symbols_materialized"] == []
    assert result.audit["label_column_read_count"] == 0
    assert result.audit["train_eligible_pairs"] == 1140


def test_fold_reader_fails_closed_if_backend_ignores_symbol_filter() -> None:
    train_symbols = [f"A{index:02d}USDT" for index in range(20)]
    test_symbols = [f"H{index:02d}USDT" for index in range(10)]
    access = _data_access()
    features = access["panel_columns"][2:]

    def bad_reader(_path, *, engine, columns, filters):
        del engine, filters
        rows = [{
            "date": pd.Timestamp("2021-01-01", tz="UTC"),
            "sequence_start_date": pd.Timestamp("2020-04-21", tz="UTC"),
            "symbol": symbol,
            **{feature: 1.0 for feature in features},
        } for symbol in [*train_symbols, test_symbols[0]]]
        return pd.DataFrame(rows).reindex(columns=columns)

    with pytest.raises(RuntimeError, match="ignored the fold symbol filter"):
        read_fold_feature_data(
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


def test_metadata_context_does_not_hash_or_deserialize_binary_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = [f"feature_{index}" for index in range(8)]
    symbols = [f"A{index:02d}USDT" for index in range(30)]
    payloads = {
        "v41_specification": {
            "decision": "authorize_v42_synthetic_ranking_excess_harness_only"
        },
        "v41_blueprint": {"candidate_family_id": "fixture"},
        "v41_audit": {"passed": True},
        "v42_result": {
            "decision": "authorize_v43_medium_non_target_pretraining_only"
        },
        "v42_audit": {"passed": True},
        "v32_dataset_manifest": {
            "panel_sha256": "declared-panel",
            "sequence_index_sha256": "declared-sequence",
            "panel_features": features,
            "symbols": symbols,
        },
        "v32_feature_schema": {
            "model_feature_order": [*features, "relative"]
        },
        "v32_asset_folds": {
            "folds": [{"fold": fold} for fold in (1, 2, 3)]
        },
    }
    inputs = {}
    expected_hashes = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        inputs[name] = path.name
        expected_hashes[name] = _sha256_file(path)
    for name in ("panel", "sequence_index"):
        path = tmp_path / f"{name}.parquet"
        path.write_bytes(b"not-a-real-parquet")
        inputs[name] = path.name
    expected_hashes["panel"] = "declared-panel"
    expected_hashes["sequence_index"] = "declared-sequence"
    config = {
        "ranking_excess_pretraining": {
            "project_root": str(tmp_path),
            "inputs": inputs,
            "expected_input_sha256": expected_hashes,
            "data_access": {
                "per_fold_filtered_read_required": True,
                "panel_columns": ["date", "symbol", *features],
                "sequence_columns": ["date", "sequence_start_date", "symbol"],
                "labels_allowed": False,
                "heldout_assets_allowed": False,
                "target_assets_allowed": False,
                "post_validation_dates_allowed": False,
                "physical_row_group_isolation_claimed": False,
            },
            "initialization": {"synthetic_v42_checkpoint_allowed": False},
        }
    }
    real_sha256_file = pretraining_module._sha256_file
    hashed_paths = []

    def metadata_only_sha256(path: Path) -> str:
        path = Path(path)
        if path.suffix == ".parquet":
            raise AssertionError("Preflight attempted to hash a Parquet file")
        hashed_paths.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(
        pretraining_module,
        "_sha256_file",
        metadata_only_sha256,
    )
    context = _metadata_context(config)
    assert len(hashed_paths) == 8
    assert context["paths"]["panel"].stat().st_size == len(b"not-a-real-parquet")


def test_checkpoint_loader_rejects_old_formats(tmp_path: Path) -> None:
    model = MultiAssetPatchTransformer(
        9,
        _tiny_architecture(),
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    path = tmp_path / "old.pt"
    torch.save(
        {
            "format_version": "v42_ranking_excess_synthetic_v1",
            "state_dict": model.state_dict(),
        },
        path,
    )
    with pytest.raises(RuntimeError, match="Unsupported or old"):
        load_ranking_excess_pretrained_checkpoint(path)


def test_smoke_requires_exact_passing_preflight_gate(tmp_path: Path) -> None:
    config = yaml.safe_load(
        Path("configs/v43_ranking_excess_pretraining.yaml").read_text(
            encoding="utf-8"
        )
    )
    pretraining = config["ranking_excess_pretraining"]
    pretraining["preflight_output_dir"] = "preflight"
    pretraining["smoke_output_dir"] = "smoke"
    blueprint = json.loads(
        Path("artifacts/v41_ranking_excess_spec/blueprint.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(RuntimeError, match="prior passing gate"):
        _load_prior_gate(tmp_path, pretraining, blueprint, "smoke")
    result_path = tmp_path / "preflight" / "result.json"
    result_path.parent.mkdir(parents=True)
    fabricated = {
        "decision": "authorize_v43_one_job_mps_smoke_only",
        "audit": {"passed": True},
        "pretraining_spec": {
            "mode": "preflight",
            "candidate_family_id": blueprint["candidate_family_id"],
            "pretraining_spec_sha256": "fixture-spec",
        },
    }
    result_path.write_text(
        json.dumps(fabricated),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="prior preflight gate is invalid"):
        _load_prior_gate(tmp_path, pretraining, blueprint, "smoke")

    expected_spec = build_pretraining_spec(
        blueprint,
        pretraining,
        "preflight",
        prior_gate=None,
    )
    result_path.write_text(
        json.dumps({
            "decision": "authorize_v43_one_job_mps_smoke_only",
            "audit": {"passed": True, "checks": {"contract_valid": True}},
            "pretraining_spec": expected_spec,
            "summary": {
                "checkpoint_count": 0,
                "total_optimizer_steps": 0,
                "total_parameters": pretraining["expected_total_parameters"],
                "pretraining_parameters": pretraining[
                    "expected_pretraining_parameters"
                ],
                "parquet_files_deserialized": 0,
            },
            "tested": {
                "panel_or_sequence_deserialized": False,
                "optimizer_executed": False,
            },
        }),
        encoding="utf-8",
    )
    gate = _load_prior_gate(tmp_path, pretraining, blueprint, "smoke")
    assert gate is not None
    assert gate["mode"] == "preflight"
    assert gate["result_sha256"] == _sha256_file(result_path)


def test_synthetic_train_job_updates_only_reconstruction_scope(
    tmp_path: Path,
) -> None:
    train_symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    dates = pd.date_range("2021-01-01", periods=12, freq="D", tz="UTC")
    features = [f"feature_{index}" for index in range(8)]
    rows = []
    for symbol_index, symbol in enumerate(train_symbols):
        for day, date in enumerate(dates):
            rows.append({
                "date": date,
                "symbol": symbol,
                **{
                    feature: float(symbol_index + day + feature_index) / 10.0
                    for feature_index, feature in enumerate(features)
                },
            })
    panel = pd.DataFrame(rows)
    scaler = FeatureScaler.fit_from_panel(
        panel,
        features,
        "2021-01-01",
        "2021-01-12",
        "2021-01-12",
        features[1],
    )
    store = TripletTensorStore(
        panel,
        features,
        lookback_days=8,
        relative_source_feature=features[1],
    )
    train_availability = {
        dates[index]: train_symbols for index in (7, 8, 9)
    }
    validation_availability = {dates[10]: train_symbols}
    seed = 42
    torch.manual_seed(seed)
    initial_model = MultiAssetPatchTransformer(
        9,
        _tiny_architecture(),
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    initialization_hash = model_state_sha256(initial_model.state_dict())
    del initial_model
    fold_access_path = tmp_path / "fold_data_access.json"
    fold_access_path.write_text("{}", encoding="utf-8")
    scaler_path = tmp_path / "scaler.json"
    scaler_path.write_text(json.dumps(asdict(scaler)), encoding="utf-8")
    job = _train_job(
        fold_entry={
            "fold": 1,
            "train_symbols": train_symbols,
            "test_symbols": ["DDDUSDT"],
        },
        seed=seed,
        architecture=_tiny_architecture(),
        feature_names=features,
        store=store,
        scaler=scaler,
        train_availability=train_availability,
        validation_availability=validation_availability,
        blueprint={
            "candidate_family_id": "fixture-ranking-excess",
            "training": {"learning_rate": 3e-4, "weight_decay": 1e-4},
        },
        pretraining={
            "validation_sampling_epoch": 0,
            "mask_fraction": 0.15,
            "gradient_clip_norm": 1.0,
            "resume_format": "v43_ranking_excess_resume_v1",
            "best_state_format": "v43_ranking_excess_best_state_v1",
            "checkpoint_format": "v43_ranking_excess_pretraining_v1",
        },
        effective={
            "validation_samples": 4,
            "train_samples_per_epoch": 4,
            "batch_size": 2,
            "early_stopping_patience": 2,
            "maximum_epochs": 2,
        },
        pretraining_spec={"pretraining_spec_sha256": "fixture-spec"},
        artifact_hashes={"fixture_sha256": "fixture"},
        fold_data_access_path=fold_access_path,
        fold_data_access_sha256=_sha256_file(fold_access_path),
        scaler_path=scaler_path,
        scaler_artifact_sha256=_sha256_file(scaler_path),
        checkpoint_root=tmp_path / "checkpoints",
        device=torch.device("cpu"),
        expected_initialization_sha256=initialization_hash,
    )
    assert job["completed_epochs"] == 2
    assert job["train_optimizer_steps"] == 4
    assert job["forbidden_initial_state_sha256"] == job[
        "forbidden_final_state_sha256"
    ]
    assert not job["labels_loaded"]
    assert not job["supervised_heads_used"]
    restored, payload = load_ranking_excess_pretrained_checkpoint(
        job["checkpoint_path"], expected_architecture=_tiny_architecture()
    )
    assert payload["model_state_sha256"] == model_state_sha256(
        restored.state_dict()
    )


def test_interrupted_resume_matches_uninterrupted_cpu_training(
    tmp_path: Path,
) -> None:
    architecture = {**_tiny_architecture(), "dropout": 0.2}
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    features = [f"feature_{index}" for index in range(8)]
    dates = pd.date_range("2021-01-01", periods=9, freq="D", tz="UTC")
    panel = pd.DataFrame([
        {
            "date": date,
            "symbol": symbol,
            **{
                feature: float(asset + day + index) / 10.0
                for index, feature in enumerate(features)
            },
        }
        for asset, symbol in enumerate(symbols)
        for day, date in enumerate(dates)
    ])
    scaler = FeatureScaler.fit_from_panel(
        panel,
        features,
        "2021-01-01",
        "2021-01-09",
        "2021-01-09",
        features[1],
    )
    store = TripletTensorStore(
        panel,
        features,
        lookback_days=8,
        relative_source_feature=features[1],
    )
    samples = [
        {"date": dates[-1], "triplet": tuple(symbols)} for _ in range(4)
    ]
    device = torch.device("cpu")

    def build():
        torch.manual_seed(91)
        model = MultiAssetPatchTransformer(
            9,
            architecture,
            expected_prediction_heads=RANKING_EXCESS_HEADS,
        )
        optimizer = torch.optim.AdamW(
            configure_pretraining_scope(model), lr=3e-4, weight_decay=1e-4
        )
        return model, optimizer

    uninterrupted_model, uninterrupted_optimizer = build()
    _run_reconstruction_batches(
        uninterrupted_model,
        store,
        scaler,
        samples,
        batch_size=2,
        seed=91,
        fold=1,
        mask_epoch=1,
        mask_fraction=0.15,
        device=device,
        optimizer=uninterrupted_optimizer,
        gradient_clip_norm=1.0,
    )
    uninterrupted_loss, _ = _run_reconstruction_batches(
        uninterrupted_model,
        store,
        scaler,
        samples,
        batch_size=2,
        seed=91,
        fold=1,
        mask_epoch=2,
        mask_fraction=0.15,
        device=device,
        optimizer=uninterrupted_optimizer,
        gradient_clip_norm=1.0,
    )

    interrupted_model, interrupted_optimizer = build()
    first_loss, first_steps = _run_reconstruction_batches(
        interrupted_model,
        store,
        scaler,
        samples,
        batch_size=2,
        seed=91,
        fold=1,
        mask_epoch=1,
        mask_fraction=0.15,
        device=device,
        optimizer=interrupted_optimizer,
        gradient_clip_norm=1.0,
    )
    early = PretrainingEarlyStopping(patience=3)
    early.update(1, first_loss)
    history = [{
        "epoch": 1,
        "validation_loss": first_loss,
        "train_optimizer_steps": first_steps,
    }]
    resume_path = tmp_path / "interrupted_resume.pt"
    _save_resume(
        resume_path,
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        early_stopping=early,
        history=history,
        completed_epoch=1,
        metadata={"fold": 1, "seed": 91},
        best_model_state=interrupted_model.state_dict(),
        device=device,
        format_version="v43_test_resume",
        best_state_format="v43_test_best",
        optimizer_parameter_names=pretraining_parameter_names(interrupted_model),
    )
    resumed_model, resumed_optimizer = build()
    _load_resume(
        resume_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        expected_metadata={"fold": 1, "seed": 91},
        device=device,
        format_version="v43_test_resume",
        expected_best_state_format="v43_test_best",
        expected_patience=3,
        maximum_epochs=10,
        expected_optimizer_parameter_names=pretraining_parameter_names(
            resumed_model
        ),
    )
    resumed_loss, _ = _run_reconstruction_batches(
        resumed_model,
        store,
        scaler,
        samples,
        batch_size=2,
        seed=91,
        fold=1,
        mask_epoch=2,
        mask_fraction=0.15,
        device=device,
        optimizer=resumed_optimizer,
        gradient_clip_norm=1.0,
    )
    assert resumed_loss == uninterrupted_loss
    assert model_state_sha256(resumed_model.state_dict()) == model_state_sha256(
        uninterrupted_model.state_dict()
    )
