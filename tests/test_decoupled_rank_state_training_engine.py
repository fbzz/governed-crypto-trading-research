from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tlm.decoupled_rank_state_training_data import (
    BASE_FEATURES,
    FoldScale,
    FoldTensorStore,
    FoldTrainingData,
)
from tlm.decoupled_rank_state_training_engine import (
    instantiate_models,
    run_training_job,
    verify_checkpoint,
)
from tlm.scientific_harness import FeatureScaler


ROOT = Path(__file__).resolve().parents[1]


def _tiny_fold() -> FoldTrainingData:
    dates = pd.date_range("2020-01-01", periods=260, tz="UTC")
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
    panel_rows = []
    labels_rows = []
    roles_rows = []
    rng = np.random.default_rng(12)
    for symbol_index, symbol in enumerate(symbols):
        for date_index, date in enumerate(dates):
            values = rng.normal(0.0, 0.02, size=len(BASE_FEATURES))
            panel_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    **dict(zip(BASE_FEATURES, values, strict=True)),
                    "target_realized_volatility_7d": 0.02 + 0.001 * symbol_index,
                }
            )
            labels_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "target_h1_maturity_date": date + pd.Timedelta(days=2),
                    "target_h1_open_to_open_log_return": 0.005 * (symbol_index - 1),
                    "h1_label_complete": True,
                }
            )
            if date_index >= 255:
                roles_rows.append(
                    {
                        "date": date,
                        "sequence_start_date": dates[date_index - 255],
                        "symbol": symbol,
                        "h1_label_complete": True,
                        "eligible_train": date_index < 258,
                        "eligible_consumed_development_validation": date_index >= 258,
                    }
                )
    panel = pd.DataFrame(panel_rows)
    labels = pd.DataFrame(labels_rows)
    roles = pd.DataFrame(roles_rows)
    scaler = FeatureScaler(
        feature_names=BASE_FEATURES,
        mean=tuple(float(value) for value in panel[list(BASE_FEATURES)].mean()),
        scale=tuple(float(value) for value in panel[list(BASE_FEATURES)].std(ddof=0)),
        source_relative_feature_index=1,
        fit_scope="eligible_train_unique_symbol_date_cells_only",
        fit_start="2020-09-12",
        fit_end="2020-09-14",
        fit_rows=9,
    )
    scale = FoldScale(
        fold=1,
        feature_scaler=scaler,
        excess_rms=0.005,
        market_rms=0.001,
        exact_train_triplet_pairs=3,
        exact_train_excess_cells=9,
    )
    store = FoldTensorStore(panel, labels, lookback_days=256)
    return FoldTrainingData(
        fold=1,
        train_symbols=symbols,
        heldout_symbols=("DDDUSDT",),
        registered_triplets=(symbols,),
        panel=panel,
        labels=labels,
        roles=roles,
        store=store,
        train_availability={date: symbols for date in dates[255:258]},
        validation_availability={date: symbols for date in dates[258:]},
        supervised_train_availability={date: symbols for date in dates[255:258]},
        supervised_validation_availability={date: symbols for date in dates[258:]},
        scale=scale,
        access_receipt={
            "access_sha256": "a" * 64,
            "target_assets_loaded": [],
            "heldout_symbols_loaded": [],
        },
    )


def test_v63_exact_model_counts_and_zero_step_completed_replay(tmp_path: Path) -> None:
    blueprint = json.loads(
        (ROOT / "artifacts/v60_decoupled_rank_state_spec/blueprint.json").read_text()
    )
    contract = yaml.safe_load(
        (ROOT / "research/phase_contracts/v063.yaml").read_text()
    )
    data = _tiny_fold()
    ranker, gate = instantiate_models(blueprint, device=__import__("torch").device("cpu"))
    assert sum(parameter.numel() for parameter in ranker.parameters()) == 1_231_634
    assert sum(parameter.numel() for parameter in gate.parameters()) == 27_489
    context = {
        "phase": "v63",
        "family_id": contract["family_id"],
        "job_id": "1|42",
        "fold": 1,
        "seed": 42,
        "phase_contract_sha256": "b" * 64,
        "source_bundle_sha256": "c" * 64,
        "fold_scale_sha256": data.scale.record()["fold_scale_sha256"],
        "data_access_sha256": "a" * 64,
        "train_symbols": list(data.train_symbols),
        "heldout_symbols_loaded": [],
        "target_assets_loaded": [],
    }
    kwargs = dict(
        blueprint=blueprint,
        contract=contract,
        data=data,
        seed=42,
        context=context,
        resume_path=tmp_path / "job.resume.pt",
        final_path=tmp_path / "job.final.pt",
        device="cpu",
        pretraining_samples=2,
        supervised_samples=2,
        validation_samples=2,
        batch_size=2,
        pretraining_epochs=1,
        supervised_epochs=1,
        patience=1,
    )
    result = run_training_job(**kwargs)
    assert result["completed"]
    assert result["optimizer_steps"] == {"pretraining": 1, "ranker": 1, "gate": 1}
    assert verify_checkpoint(
        tmp_path / "job.final.pt",
        blueprint=blueprint,
        contract=contract,
        context=context,
        device="cpu",
    )["passed"]
    replay = run_training_job(**kwargs)
    assert replay["status"] == "already_complete"
    assert replay["new_optimizer_steps"] == 0
