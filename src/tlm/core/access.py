from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SyntheticAccessLedger:
    authorized_metadata_reads: int = 0
    synthetic_tensor_generations: int = 0
    synthetic_optimizer_steps: int = 0
    synthetic_checkpoint_writes: int = 0
    synthetic_checkpoint_reads: int = 0
    parquet_deserializations: int = 0
    real_panel_or_label_reads: int = 0
    previous_checkpoint_reads: int = 0
    real_training_epochs: int = 0
    real_market_predictions: int = 0
    real_performance_metrics: int = 0
    real_pnl_evaluations: int = 0
    target_asset_loads: int = 0

    def forbidden_operations_are_zero(self) -> bool:
        return all(
            getattr(self, name) == 0
            for name in (
                "parquet_deserializations",
                "real_panel_or_label_reads",
                "previous_checkpoint_reads",
                "real_training_epochs",
                "real_market_predictions",
                "real_performance_metrics",
                "real_pnl_evaluations",
                "target_asset_loads",
            )
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class DatasetAccessLedger:
    authorized_metadata_reads: int = 0
    authorized_parquet_deserializations: int = 0
    authorized_panel_rows: int = 0
    authorized_sequence_rows: int = 0
    parquet_writes: int = 0
    scaler_fits: int = 0
    model_instantiations: int = 0
    optimizer_steps: int = 0
    checkpoint_reads: int = 0
    market_predictions: int = 0
    performance_metrics: int = 0
    pnl_evaluations: int = 0
    target_asset_loads: int = 0
    missing_value_imputations: int = 0
    universe_reselections: int = 0

    def forbidden_operations_are_zero(self) -> bool:
        return all(
            getattr(self, name) == 0
            for name in (
                "scaler_fits",
                "model_instantiations",
                "optimizer_steps",
                "checkpoint_reads",
                "market_predictions",
                "performance_metrics",
                "pnl_evaluations",
                "target_asset_loads",
                "missing_value_imputations",
                "universe_reselections",
            )
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
