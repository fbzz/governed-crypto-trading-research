from __future__ import annotations

from pathlib import Path

import yaml

from tlm.__main__ import build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_v77_config_matches_frozen_input_and_source_contracts() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/v77_persistent_duration_training.yaml").read_text()
    )
    contract = yaml.safe_load(
        (ROOT / "research/phase_contracts/v077.yaml").read_text()
    )
    training = config["persistent_duration_training"]
    assert set(training["inputs"].values()) == set(
        contract["access_contract"]["allowed_inputs"]
    )
    assert training["require_clean_git"] is True
    assert len(training["source_receipt_files"]) == len(
        set(training["source_receipt_files"])
    )
    assert all((ROOT / path).is_file() for path in training["source_receipt_files"])
    assert config["output_dir"] == contract["artifact_contract"]["output_dir"]
    assert config["checkpoint_dir"] == contract["artifact_contract"][
        "checkpoint_dir"
    ]


def test_v77_cli_exposes_all_frozen_operator_phases() -> None:
    parser = build_parser()
    for command in (
        "persistent-duration-training-preflight",
        "persistent-duration-training-smoke",
        "persistent-duration-training",
        "persistent-duration-training-verify",
        "persistent-duration-training-replay",
    ):
        parsed = parser.parse_args(
            [command, "--config", "configs/v77_persistent_duration_training.yaml"]
        )
        assert parsed.command == command

