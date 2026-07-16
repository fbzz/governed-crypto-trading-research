from __future__ import annotations

from tlm.state_conditioned_multi_horizon_spec import analytic_parameter_count


def _architecture() -> dict:
    return {
        "input_features": 9,
        "lookback_days": 256,
        "patch_length_days": 16,
        "patch_stride_days": 8,
        "d_model": 96,
        "temporal_encoder_layers": 3,
        "cross_asset_attention_layers": 1,
        "feed_forward_width": 384,
        "output_horizons": [1, 3, 7],
        "output_quantiles": [0.2, 0.5, 0.8],
    }


def test_v55_analytic_parameter_count_is_frozen() -> None:
    assert analytic_parameter_count(_architecture()) == 465_513


def test_v55_parameter_count_changes_with_output_contract() -> None:
    architecture = _architecture()
    architecture["output_quantiles"] = [0.1, 0.5, 0.9, 0.95]
    assert analytic_parameter_count(architecture) == 465_804
