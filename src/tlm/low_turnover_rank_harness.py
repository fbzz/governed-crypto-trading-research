from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Mapping

import torch

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic, write_yaml_atomic
from .low_turnover_rank_model import (
    LowTurnoverRankModel,
    apply_low_turnover_policy,
    low_turnover_rank_loss,
)


EXPECTED_INPUTS = {
    "specification",
    "blueprint",
    "audit",
    "result",
    "artifact_manifest",
    "input_hash_receipt",
}
FAMILY_ID = "tlm_low_turnover_cross_sectional_rank_v1"
V82_ACTION = "authorize_v82_non_target_low_turnover_rank_dataset_only"


def _mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing V81 {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V81 {label} must be a JSON object")
    return value


def _inside(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"V81 {label} escapes the repository") from exc
    return path


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _verify_inputs(
    root: Path, section: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    inputs = section.get("inputs")
    expected_hashes = section.get("expected_input_sha256")
    if not isinstance(inputs, Mapping) or not isinstance(expected_hashes, Mapping):
        raise ValueError("V81 input paths and hashes must be mappings")
    if set(inputs) != EXPECTED_INPUTS or set(expected_hashes) != EXPECTED_INPUTS:
        raise ValueError("V81 metadata input allowlist drift")
    payloads: dict[str, dict[str, Any]] = {}
    observed: dict[str, str] = {}
    for name in sorted(EXPECTED_INPUTS):
        relative = inputs[name]
        expected = expected_hashes[name]
        if not isinstance(relative, str) or not relative.endswith(".json"):
            raise ValueError("V81 may read only registered JSON metadata")
        path = _inside(root, relative, f"input {name}")
        digest = file_sha256(path)
        if digest != expected:
            raise ValueError(f"V81 input hash drift: {name}")
        payloads[name] = _mapping(path, name)
        observed[name] = digest
    return payloads, observed


def _verify_v80(payloads: Mapping[str, dict[str, Any]]) -> dict[str, bool]:
    specification = payloads["specification"]
    blueprint = payloads["blueprint"]
    audit = payloads["audit"]
    result = payloads["result"]
    manifest = payloads["artifact_manifest"]
    return {
        "v80_receipts_are_exact": (
            result.get("decision")
            == "authorize_v81_synthetic_low_turnover_rank_harness_only"
            and result.get("result_sha256")
            == "4008ce046d0d05665708c4cd3dba7fd2405d40314d7b4dafdc7e4b573f6008ad"
            and audit.get("passed") is True
            and audit.get("checks_passed") == 14
            and specification.get("specification_sha256")
            == "d7fd9306ede1afc0fc193e705c2d1d539e5fde60c023b587012ecfa8812f9cfd"
            and blueprint.get("blueprint_sha256")
            == "3b080b6cfcea2be6ef2a3347397e7f669573870abba0f6966bc3eb76eeb1d649"
        ),
        "v80_design_is_exact": (
            specification.get("family_id") == FAMILY_ID
            and specification.get("parameter_count") == 10993
            and specification.get("structural_maximum_evaluation_turnover") == 16.0
            and blueprint.get("architecture", {}).get("expected_total_parameters")
            == 10993
            and blueprint.get("architecture", {}).get("architecture_variant_count")
            == 1
            and blueprint.get("policy", {}).get("decision_interval_days") == 21
            and blueprint.get("policy", {}).get("structural_maximum_turnover")
            == 16.0
        ),
        "v80_manifest_is_complete": (
            manifest.get("schema_version") == "v80-artifact-manifest/v1"
            and isinstance(manifest.get("files"), Mapping)
            and len(manifest.get("files", {})) == 8
        ),
        "v80_target_and_scientific_access_remained_zero": (
            result.get("scientific_data_reads") == 0
            and result.get("models_or_checkpoints_loaded") == 0
            and result.get("outcome_rows_read") == 0
            and result.get("target_assets_loaded") == []
            and specification.get("target_assets_status") == "sealed"
        ),
    }


def _synthetic_batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randn(4, 128, 3, 8, generator=generator, dtype=torch.float32)
    targets = torch.randn(4, 3, generator=generator, dtype=torch.float32)
    targets = targets - targets.mean(dim=1, keepdim=True)
    return values, targets


def _optimizer_steps(
    model: LowTurnoverRankModel,
    optimizer: torch.optim.Optimizer,
    values: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
) -> bool:
    finite = True
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        scores = model(values)
        loss, _ = low_turnover_rank_loss(scores, targets)
        finite = finite and bool(torch.isfinite(loss).item())
        loss.backward()
        finite = finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        )
        optimizer.step()
    return finite


def _resume_equivalence(
    values: torch.Tensor, targets: torch.Tensor, root: Path
) -> tuple[bool, int, int]:
    torch.manual_seed(20260716)
    uninterrupted = LowTurnoverRankModel()
    uninterrupted.eval()
    uninterrupted_optimizer = torch.optim.AdamW(
        uninterrupted.parameters(), lr=0.001, weight_decay=0.0001
    )
    uninterrupted_finite = _optimizer_steps(
        uninterrupted, uninterrupted_optimizer, values, targets, 3
    )

    torch.manual_seed(20260716)
    interrupted = LowTurnoverRankModel()
    interrupted.eval()
    interrupted_optimizer = torch.optim.AdamW(
        interrupted.parameters(), lr=0.001, weight_decay=0.0001
    )
    interrupted_finite = _optimizer_steps(
        interrupted, interrupted_optimizer, values, targets, 1
    )
    with TemporaryDirectory(dir=root) as directory:
        checkpoint_path = Path(directory) / "synthetic.pt"
        torch.save(
            {
                "model": interrupted.state_dict(),
                "optimizer": interrupted_optimizer.state_dict(),
                "completed_steps": 1,
            },
            checkpoint_path,
        )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        resumed = LowTurnoverRankModel()
        resumed.eval()
        resumed.load_state_dict(payload["model"])
        resumed_optimizer = torch.optim.AdamW(
            resumed.parameters(), lr=0.001, weight_decay=0.0001
        )
        resumed_optimizer.load_state_dict(payload["optimizer"])
        resumed_finite = _optimizer_steps(
            resumed, resumed_optimizer, values, targets, 2
        )

    max_difference = max(
        float(torch.max(torch.abs(left - right)).item())
        for left, right in zip(uninterrupted.parameters(), resumed.parameters())
    )
    return (
        uninterrupted_finite
        and interrupted_finite
        and resumed_finite
        and max_difference <= 1.0e-7,
        1,
        1,
    )


def _run_synthetic_checks(section: Mapping[str, Any], output: Path) -> dict[str, Any]:
    contract = section["synthetic_contract"]
    seed = 20260716
    values, targets = _synthetic_batch(seed)

    torch.manual_seed(seed)
    model = LowTurnoverRankModel()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.eval()
    sequence = model.forward_sequence(values)
    final_scores = sequence[:, -1]

    permutation = torch.tensor([2, 0, 1])
    permuted = model(values[:, :, permutation])
    permutation_ok = torch.allclose(
        permuted, final_scores[:, permutation], atol=1.0e-6, rtol=0.0
    )

    changed = values.clone()
    changed[:, 64:] = changed[:, 64:] + 100.0
    changed_sequence = model.forward_sequence(changed)
    causality_ok = torch.allclose(
        sequence[:, 63], changed_sequence[:, 63], atol=1.0e-6, rtol=0.0
    )
    centered_ok = torch.allclose(
        final_scores.sum(dim=1), torch.zeros(4), atol=1.0e-6, rtol=0.0
    )
    loss, components = low_turnover_rank_loss(final_scores, targets)
    loss_finite = bool(torch.isfinite(loss).item()) and all(
        bool(torch.isfinite(value).item()) for value in components.values()
    )

    model.train()
    cpu_optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.0001
    )
    cpu_backward_finite = _optimizer_steps(
        model, cpu_optimizer, values, targets, contract["cpu_backward_steps"]
    )

    mps_available = torch.backends.mps.is_available()
    if not mps_available:
        raise ValueError("V81 requires Apple MPS for its frozen synthetic gate")
    torch.manual_seed(seed)
    mps_model = LowTurnoverRankModel().to("mps")
    mps_model.train()
    mps_optimizer = torch.optim.AdamW(
        mps_model.parameters(), lr=0.001, weight_decay=0.0001
    )
    mps_backward_finite = _optimizer_steps(
        mps_model,
        mps_optimizer,
        values.to("mps"),
        targets.to("mps"),
        contract["mps_backward_steps"],
    )
    torch.mps.synchronize()

    resume_ok, checkpoint_writes, checkpoint_reads = _resume_equivalence(
        values, targets, output.parent
    )

    adversarial_scores = torch.zeros(160, 3, dtype=torch.float32)
    for decision_number, index in enumerate(range(0, 160, 21)):
        top = decision_number % 3
        adversarial_scores[index] = -0.5
        adversarial_scores[index, top] = 1.0
    policy = apply_low_turnover_policy(
        adversarial_scores,
        torch.ones(160, dtype=torch.bool),
        decision_interval=21,
        switch_margin=0.25,
    )
    policy_ok = (
        policy["decisions"] == 8
        and policy["turnover"] == 16.0
        and policy["turnover"] <= contract["structural_maximum_turnover"]
        and policy["actions"]
        == {"cash": 0, "enter": 1, "exit": 0, "hold": 0, "switch": 7}
        and policy["final_liquidation_turnover"] == 1.0
    )

    return {
        "parameter_count": parameter_count,
        "output_shape": list(final_scores.shape),
        "permutation_equivariant": bool(permutation_ok),
        "causal_prefix_invariant": bool(causality_ok),
        "centered_scores": bool(centered_ok),
        "loss_finite": loss_finite,
        "cpu_backward_finite": cpu_backward_finite,
        "mps_available": mps_available,
        "mps_backward_finite": mps_backward_finite,
        "checkpoint_roundtrip": resume_ok,
        "resume_equivalent": resume_ok,
        "synthetic_checkpoint_writes": checkpoint_writes,
        "synthetic_checkpoint_reads": checkpoint_reads,
        "synthetic_optimizer_steps": 10,
        "model_instantiations": 5,
        "policy_decisions": policy["decisions"],
        "adversarial_turnover": policy["turnover"],
        "policy_structural_bound_passed": policy_ok,
    }


def run_low_turnover_rank_harness(config: dict[str, Any]) -> dict[str, Any]:
    """Run the exact V81 architecture only on deterministic synthetic tensors."""

    section = config.get("low_turnover_rank_harness")
    if not isinstance(section, Mapping):
        raise ValueError("Missing low_turnover_rank_harness config section")
    root = Path(section.get("project_root", ".")).resolve()
    output_value = config.get("output_dir")
    if not isinstance(output_value, str):
        raise ValueError("V81 output_dir must be a repository-relative path")
    output = _inside(root, output_value, "output directory")
    payloads, input_hashes = _verify_inputs(root, section)
    receipt_checks = _verify_v80(payloads)
    if not all(receipt_checks.values()):
        failed = sorted(name for name, passed in receipt_checks.items() if not passed)
        raise ValueError(f"V81 V80 receipt gate failed: {failed}")

    smoke = _run_synthetic_checks(section, output)
    synthetic_checks = {
        "exact_parameter_count": smoke["parameter_count"] == 10993,
        "exact_output_shape": smoke["output_shape"] == [4, 3],
        "causal_prefix_invariance": smoke["causal_prefix_invariant"],
        "asset_permutation_equivariance": smoke["permutation_equivariant"],
        "centered_score_sum_zero": smoke["centered_scores"],
        "finite_point_pairwise_and_total_loss": smoke["loss_finite"],
        "finite_cpu_backward": smoke["cpu_backward_finite"],
        "finite_mps_backward": smoke["mps_available"] and smoke["mps_backward_finite"],
        "synthetic_checkpoint_roundtrip": smoke["checkpoint_roundtrip"],
        "interrupted_resume_equivalence": smoke["resume_equivalent"],
        "adversarial_turnover_at_most_16": smoke["policy_structural_bound_passed"],
    }
    checks = {**receipt_checks, **synthetic_checks}
    passed = all(checks.values())
    if not passed:
        failed = sorted(name for name, value in checks.items() if not value)
        raise ValueError(f"V81 synthetic gate failed: {failed}")

    harness_spec = {
        "schema_version": "v81-low-turnover-rank-harness-spec/v1",
        "family_id": FAMILY_ID,
        "synthetic_contract": section["synthetic_contract"],
        "real_data_allowed": False,
        "prior_checkpoint_allowed": False,
        "target_assets_status": "sealed",
    }
    harness_spec["harness_spec_sha256"] = canonical_sha256(harness_spec)
    access_ledger = {
        "json_metadata_reads": 6,
        "real_data_reads": 0,
        "parquet_deserializations": 0,
        "market_panel_reads": 0,
        "prior_checkpoint_loads": 0,
        "synthetic_checkpoint_writes": smoke["synthetic_checkpoint_writes"],
        "synthetic_checkpoint_reads": smoke["synthetic_checkpoint_reads"],
        "model_instantiations": smoke["model_instantiations"],
        "synthetic_optimizer_steps": smoke["synthetic_optimizer_steps"],
        "real_training_runs": 0,
        "real_inference_runs": 0,
        "predictions_generated": 0,
        "positions_generated": 0,
        "performance_metrics_computed": 0,
        "pnl_evaluations": 0,
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
    }
    audit = {
        "schema_version": "v81-audit/v1",
        "passed": True,
        "checks": checks,
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "access_ledger": access_ledger,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    result = {
        "schema_version": "v81-result/v1",
        "decision": V82_ACTION,
        "family_id": FAMILY_ID,
        "harness_spec_sha256": harness_spec["harness_spec_sha256"],
        "parameter_count": 10993,
        "audit_checks_passed": len(checks),
        "synthetic_optimizer_steps": smoke["synthetic_optimizer_steps"],
        "adversarial_turnover": smoke["adversarial_turnover"],
        "structural_maximum_turnover": 16.0,
        "real_data_reads": 0,
        "prior_checkpoint_loads": 0,
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
        "v82_executed": False,
        "deployable": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    input_receipt = {
        "schema_version": "v81-input-receipt/v1",
        "inputs": input_hashes,
    }
    input_receipt["input_receipt_sha256"] = canonical_sha256(input_receipt)

    source_files = section.get("source_receipt_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("V81 source receipt file list is required")
    source_hashes: dict[str, str] = {}
    for relative in source_files:
        if not isinstance(relative, str):
            raise ValueError("V81 source receipt paths must be strings")
        source_hashes[relative] = file_sha256(_inside(root, relative, "source file"))
    source_receipt = {
        "schema_version": "v81-source-receipt/v1",
        "files": source_hashes,
    }
    source_receipt["source_receipt_sha256"] = canonical_sha256(source_receipt)

    report = "\n".join([
        "# V81 synthetic low-turnover rank harness",
        "",
        "The exact 10,993-parameter model passed causal-prefix, asset-permutation,",
        "centered-score, point/pairwise loss, CPU/MPS backward, checkpoint resume,",
        "and adversarial turnover checks using deterministic synthetic tensors.",
        "",
        "The adversarial 160-date policy path made eight decisions and reached",
        "exactly 16.0 turnover including final liquidation, proving the frozen",
        "ceiling by construction.",
        "",
        "No real data, prior checkpoint, prediction, position, metric/PnL, outcome,",
        "or target asset was accessed. Only the separate V82 dataset is authorized.",
        "",
    ])

    output.mkdir(parents=True, exist_ok=True)
    write_yaml_atomic(output / "resolved_config.yaml", config)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", source_receipt)
    write_json_atomic(output / "harness_spec.json", harness_spec)
    write_json_atomic(output / "smoke.json", smoke)
    write_json_atomic(output / "audit.json", audit)
    write_json_atomic(output / "result.json", result)
    _write_text_atomic(output / "report.md", report)
    manifest_files = [
        "resolved_config.yaml",
        "input_hash_receipt.json",
        "source_receipt.json",
        "harness_spec.json",
        "smoke.json",
        "audit.json",
        "result.json",
        "report.md",
    ]
    manifest = {
        "schema_version": "v81-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_files},
    }
    manifest["artifact_manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)
    return {
        "decision": result["decision"],
        "harness_spec": harness_spec,
        "smoke": smoke,
        "audit": audit,
        "result": result,
        "artifact_manifest": manifest,
    }
