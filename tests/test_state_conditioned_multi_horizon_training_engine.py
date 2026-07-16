from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from tlm.state_conditioned_multi_horizon_training_engine import (
    _cpu_float64_sum,
    GlobalLossAccumulator,
    StrictEarlyStopping,
    V58Batch,
    V58BatchStream,
    V58CheckpointContext,
    build_v58_adamw,
    capture_v58_rng_state,
    clip_v58_gradients,
    configure_v58_runtime,
    load_v58_checkpoint,
    model_state_sha256,
    optimizer_contract,
    prove_v58_interrupted_resume_equivalence,
    restore_v58_rng_state,
    run_v58_training_job,
    semantic_state_sha256,
    v58_loss_component_sums,
    verify_v58_checkpoint_roundtrip,
)


class _TinyDropoutQuantileModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(5, 12)
        self.dropout = nn.Dropout(p=0.25)
        self.output = nn.Linear(12, 27)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.hidden(features))
        return self.output(self.dropout(hidden)).reshape(-1, 3, 3, 3)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _context(*, job: str = "job-a") -> V58CheckpointContext:
    return V58CheckpointContext(
        scaler_sha256=_sha("scaler"),
        data_access_sha256=_sha("data-access"),
        phase_contract_sha256=_sha("phase-contract"),
        source_bundle_sha256=_sha("source-bundle"),
        job_metadata={
            "job_key": job,
            "origin": "origin_2024",
            "geometry": "expanding",
            "fold": 1,
            "seed": 42,
        },
    )


class _DeterministicBatchProvider:
    def __call__(self, role: str, epoch: int) -> V58BatchStream:
        registered_epoch = epoch if role == "train" else 0
        generator = torch.Generator(device="cpu").manual_seed(
            10_000 + registered_epoch + (0 if role == "train" else 1_000)
        )
        batches = []
        for batch_index, size in enumerate((3, 2)):
            features = torch.randn(size, 5, generator=generator)
            targets = torch.randn(size, 3, 3, generator=generator) * 0.02
            # Keep every h7 target active and non-tied while varying pinball counts.
            targets[:, :, 2] += torch.tensor([-0.05, 0.0, 0.05])
            mask = torch.ones_like(targets, dtype=torch.bool)
            if batch_index == 1:
                mask[0, 0, 0] = False
                targets[0, 0, 0] = float("nan")
            batches.append(V58Batch(features, targets, mask))
        receipt = _sha(f"{role}:{registered_epoch}:ordered-draws")
        return V58BatchStream(batches=batches, sampler_receipt=receipt)


def test_strict_early_stopping_ties_are_non_improvements_and_best_is_earliest() -> None:
    state = StrictEarlyStopping(patience=2)
    assert state.update(1, 1.0)
    assert not state.update(2, 1.0)
    assert state.best_epoch == 1
    assert state.consecutive_non_improvements == 1
    assert state.update(3, 0.9)
    assert state.best_epoch == 3
    assert state.consecutive_non_improvements == 0
    assert not state.update(4, 0.9)
    assert not state.should_stop
    assert not state.update(5, 0.91)
    assert state.should_stop
    assert state.best_epoch == 3


def test_global_loss_uses_component_sums_and_exact_cell_pair_counts() -> None:
    generator = torch.Generator().manual_seed(91)
    predictions = torch.randn(5, 3, 3, 3, generator=generator) * 0.02
    targets = torch.randn(5, 3, 3, generator=generator) * 0.02
    targets[:, :, 2] += torch.tensor([-0.05, 0.0, 0.05])
    mask = torch.ones_like(targets, dtype=torch.bool)
    mask[0, 0, :2] = False
    mask[4, 2, 0] = False
    targets[~mask] = float("nan")

    accumulator = GlobalLossAccumulator()
    accumulator.update(predictions[:1], targets[:1], mask[:1])
    accumulator.update(predictions[1:], targets[1:], mask[1:])
    aggregate = accumulator.finalize()
    direct = v58_loss_component_sums(predictions, targets, target_mask=mask)

    assert aggregate["pinball_count"] == int(mask.sum()) * 3
    assert aggregate["crossing_count"] == 5 * 3 * 3
    assert aggregate["ranking_pair_count"] == direct["ranking_pair_count"]
    assert aggregate["pinball"] == pytest.approx(
        direct["pinball_sum"] / direct["pinball_count"], rel=0, abs=1e-15
    )
    assert aggregate["ranking"] == pytest.approx(
        direct["ranking_sum"] / direct["ranking_pair_count"], rel=0, abs=1e-15
    )
    assert aggregate["crossing"] == pytest.approx(
        direct["crossing_sum"] / direct["crossing_count"], rel=0, abs=1e-15
    )
    assert aggregate["total"] == pytest.approx(
        aggregate["pinball"]
        + 0.5 * aggregate["ranking"]
        + 0.1 * aggregate["crossing"],
        rel=0,
        abs=1e-15,
    )


def test_device_loss_sum_moves_to_cpu_before_float64_conversion() -> None:
    events: list[str] = []

    class DeviceValues:
        def detach(self) -> "DeviceValues":
            events.append("detach")
            return self

        def cpu(self) -> "CpuValues":
            events.append("cpu")
            return CpuValues()

    class CpuValues:
        def double(self) -> "CpuValues":
            events.append("double")
            return self

        def sum(self) -> "CpuValues":
            events.append("sum")
            return self

        def item(self) -> float:
            events.append("item")
            return 2.5

    assert _cpu_float64_sum(DeviceValues()) == 2.5  # type: ignore[arg-type]
    assert events == ["detach", "cpu", "double", "sum", "item"]


def test_cpu_loss_sum_preserves_the_frozen_float64_reduction() -> None:
    values = torch.tensor(
        [1.0e20, 1.0, -1.0e20, 3.25, -0.125], dtype=torch.float32
    )
    expected = float(values.detach().double().sum().cpu())
    assert _cpu_float64_sum(values) == expected


def test_loss_component_sums_uses_float_zero_when_no_ranking_pairs() -> None:
    predictions = torch.zeros(2, 3, 3, 3, dtype=torch.float32)
    targets = torch.zeros(2, 3, 3, dtype=torch.float32)
    result = v58_loss_component_sums(predictions, targets)
    assert result["ranking_pair_count"] == 0
    assert result["ranking_sum"] == 0.0
    assert isinstance(result["ranking_sum"], float)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS is unavailable"
)
def test_loss_component_sums_support_float32_mps_without_float64_device_cast() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260714)
    predictions_cpu = torch.randn(2, 3, 3, 3, generator=generator)
    targets_cpu = torch.randn(2, 3, 3, generator=generator)
    targets_cpu[:, :, 2] += torch.tensor([-0.05, 0.0, 0.05])
    expected = v58_loss_component_sums(predictions_cpu, targets_cpu)
    observed = v58_loss_component_sums(
        predictions_cpu.to("mps"), targets_cpu.to("mps")
    )
    assert observed["pinball_count"] == expected["pinball_count"] == 54
    assert observed["crossing_count"] == expected["crossing_count"] == 18
    assert observed["ranking_pair_count"] == expected["ranking_pair_count"]
    for key in ("pinball_sum", "ranking_sum", "crossing_sum"):
        assert math.isfinite(float(observed[key]))
        assert float(observed[key]) == pytest.approx(
            float(expected[key]), rel=1e-6, abs=1e-6
        )


def test_adamw_and_gradient_clip_match_the_frozen_v58_contract() -> None:
    model = _TinyDropoutQuantileModel()
    optimizer = build_v58_adamw(model)
    group = optimizer.param_groups[0]
    assert type(optimizer).__name__ == "AdamW"
    assert group["lr"] == 0.0003
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1.0e-8
    assert group["weight_decay"] == 0.0001
    assert group["amsgrad"] is False
    assert group["foreach"] is False
    assert group["fused"] is False
    assert group["capturable"] is False
    assert group["maximize"] is False
    assert group["differentiable"] is False
    contract = optimizer_contract(optimizer, model)
    assert contract["parameter_names"] == [
        "hidden.weight",
        "hidden.bias",
        "output.weight",
        "output.bias",
    ]

    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)
    unclipped_norm = clip_v58_gradients(model)
    clipped_norm = math.sqrt(
        sum(float(parameter.grad.square().sum()) for parameter in model.parameters())
    )
    assert unclipped_norm > 1.0
    assert clipped_norm <= 1.0 + 1e-6


def test_cpu_rng_capture_restore_and_semantic_hash_are_exact() -> None:
    configure_v58_runtime("cpu", seed=123)
    state = capture_v58_rng_state("cpu")
    first = torch.rand(7)
    restore_v58_rng_state(
        cpu_rng_state=state["cpu_rng_state"],
        mps_rng_state=state["mps_rng_state"],
        device="cpu",
    )
    second = torch.rand(7)
    assert torch.equal(first, second)
    assert state["mps_rng_state"] is None
    assert semantic_state_sha256(state) != semantic_state_sha256(
        capture_v58_rng_state("cpu")
    )


def test_interrupted_epoch_boundary_resume_matches_uninterrupted_and_replays_zero_steps(
    tmp_path: Path,
) -> None:
    provider = _DeterministicBatchProvider()
    proof = prove_v58_interrupted_resume_equivalence(
        model_factory=_TinyDropoutQuantileModel,
        batch_provider=provider,
        job_seed=42,
        context=_context(),
        work_dir=tmp_path / "proof",
        device="cpu",
        maximum_epochs=2,
        patience=2,
        interrupt_after_completed_epoch=1,
    )
    assert proof["passed"]
    assert proof["interrupted"]["completed_epoch"] == 1
    assert proof["resumed"]["new_optimizer_steps"] == 2
    assert all(proof["comparisons"].values())
    assert proof["resumed_roundtrip"]["checks"][
        "current_and_best_state_are_distinct"
    ]

    replay = run_v58_training_job(
        model_factory=_TinyDropoutQuantileModel,
        batch_provider=provider,
        job_seed=42,
        context=_context(),
        resume_path=tmp_path / "proof" / "interrupted.resume.pt",
        final_path=tmp_path / "proof" / "interrupted.final.pt",
        device="cpu",
        maximum_epochs=2,
        patience=2,
    )
    assert replay["status"] == "already_complete"
    assert replay["new_optimizer_steps"] == 0
    assert not list((tmp_path / "proof").glob(".*.tmp"))


def test_resume_is_same_job_only_and_optimizer_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    provider = _DeterministicBatchProvider()
    resume = tmp_path / "job.resume.pt"
    final = tmp_path / "job.final.pt"
    interrupted = run_v58_training_job(
        model_factory=_TinyDropoutQuantileModel,
        batch_provider=provider,
        job_seed=42,
        context=_context(),
        resume_path=resume,
        final_path=final,
        device="cpu",
        maximum_epochs=2,
        patience=2,
        interrupt_after_completed_epoch=1,
    )
    assert interrupted["status"] == "interrupted"
    with pytest.raises(RuntimeError, match="same-job"):
        run_v58_training_job(
            model_factory=_TinyDropoutQuantileModel,
            batch_provider=provider,
            job_seed=42,
            context=_context(job="job-b"),
            resume_path=resume,
            final_path=final,
            device="cpu",
            maximum_epochs=2,
            patience=2,
        )

    payload = torch.load(resume, map_location="cpu", weights_only=False)
    first_parameter = next(iter(payload["optimizer_state"]["state"]))
    payload["optimizer_state"]["state"][first_parameter]["exp_avg"].flatten()[0] += 1.0
    torch.save(payload, resume)
    model = _TinyDropoutQuantileModel()
    optimizer = build_v58_adamw(model)
    with pytest.raises(RuntimeError, match="optimizer semantic hash drift"):
        load_v58_checkpoint(
            resume,
            expected_kind="resume",
            model=model,
            optimizer=optimizer,
            context=_context(),
            device="cpu",
            maximum_epochs=2,
            patience=2,
        )


def test_checkpoint_roundtrip_verifier_rejects_current_model_hash_tamper(
    tmp_path: Path,
) -> None:
    provider = _DeterministicBatchProvider()
    result = run_v58_training_job(
        model_factory=_TinyDropoutQuantileModel,
        batch_provider=provider,
        job_seed=42,
        context=_context(),
        resume_path=tmp_path / "resume.pt",
        final_path=tmp_path / "final.pt",
        device="cpu",
        maximum_epochs=1,
        patience=2,
    )
    assert result["completed"]
    verification = verify_v58_checkpoint_roundtrip(
        tmp_path / "final.pt",
        model_factory=_TinyDropoutQuantileModel,
        job_seed=42,
        context=_context(),
        checkpoint_kind="final",
        device="cpu",
        maximum_epochs=1,
        patience=2,
    )
    assert verification["passed"]

    payload = torch.load(tmp_path / "final.pt", map_location="cpu", weights_only=False)
    first_name = next(iter(payload["current_model_state"]))
    payload["current_model_state"][first_name].flatten()[0] += 1.0
    torch.save(payload, tmp_path / "final.pt")
    with pytest.raises(RuntimeError, match="current model semantic hash drift"):
        verify_v58_checkpoint_roundtrip(
            tmp_path / "final.pt",
            model_factory=_TinyDropoutQuantileModel,
            job_seed=42,
            context=_context(),
            checkpoint_kind="final",
            device="cpu",
            maximum_epochs=1,
            patience=2,
        )


def test_checkpoint_roundtrip_returns_independently_verified_manifest_hashes(
    tmp_path: Path,
) -> None:
    provider = _DeterministicBatchProvider()
    context = _context()
    result = run_v58_training_job(
        model_factory=_TinyDropoutQuantileModel,
        batch_provider=provider,
        job_seed=42,
        context=context,
        resume_path=tmp_path / "resume.pt",
        final_path=tmp_path / "final.pt",
        device="cpu",
        maximum_epochs=1,
        patience=2,
    )
    assert result["completed"]
    payload = torch.load(
        tmp_path / "final.pt", map_location="cpu", weights_only=False
    )

    verification = verify_v58_checkpoint_roundtrip(
        tmp_path / "final.pt",
        model_factory=_TinyDropoutQuantileModel,
        job_seed=42,
        context=context,
        checkpoint_kind="final",
        device="cpu",
        maximum_epochs=1,
        patience=2,
    )

    assert verification["passed"]
    assert verification["current_model_state_sha256"] == model_state_sha256(
        payload["current_model_state"]
    )
    assert verification["best_model_state_sha256"] == model_state_sha256(
        payload["best_model_state"]
    )
    assert verification["optimizer_state_sha256"] == semantic_state_sha256(
        payload["optimizer_state"]
    )
    assert verification["rng_state_sha256"] == semantic_state_sha256(
        {
            "cpu_rng_state": payload["cpu_rng_state"],
            "mps_rng_state": payload["mps_rng_state"],
        }
    )
    assert verification["early_stopping_state_sha256"] == semantic_state_sha256(
        payload["early_stopping_state"]
    )
    assert verification["full_history_sha256"] == semantic_state_sha256(
        payload["full_history"]
    )
    assert verification["scaler_sha256"] == context.scaler_sha256
    assert verification["data_access_sha256"] == context.data_access_sha256
    assert verification["phase_contract_sha256"] == context.phase_contract_sha256
    assert verification["source_bundle_sha256"] == context.source_bundle_sha256
    assert verification["job_metadata"] == dict(context.job_metadata)
    assert verification["job_metadata"] is not payload["job_metadata"]


def test_model_state_hash_is_name_shape_dtype_and_value_sensitive() -> None:
    model = _TinyDropoutQuantileModel()
    original = model_state_sha256(model)
    with torch.no_grad():
        model.hidden.weight.flatten()[0] += 1.0
    assert model_state_sha256(model) != original
