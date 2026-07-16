from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_passed(payload: dict) -> bool:
    if "passed" in payload:
        return bool(payload["passed"])
    return bool(payload.get("audit", {}).get("passed"))


def _finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def build_control_certificate(config: dict) -> dict[str, object]:
    certificate = config["control_certificate"]
    paths = {
        "validation_result": Path(certificate["validation_result_path"]),
        "validation_audit": Path(certificate["validation_audit_path"]),
        "validation_config": Path(certificate["validation_config_path"]),
        "evidence_ledger": Path(certificate["evidence_ledger_path"]),
        "evidence_audit": Path(certificate["evidence_audit_path"]),
        "control_implementation": Path(certificate["control_implementation_path"]),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Control certificate inputs are missing: {missing}")

    validation = _load_json(paths["validation_result"])
    validation_audit = _load_json(paths["validation_audit"])
    ledger = _load_json(paths["evidence_ledger"])
    ledger_audit = _load_json(paths["evidence_audit"])
    validation_config = yaml.safe_load(
        paths["validation_config"].read_text(encoding="utf-8")
    )

    control_name = str(certificate["frozen_control_name"])
    expected_scenarios = list(certificate["expected_scenarios"])
    expected_blocks = {str(int(value)) for value in certificate["expected_blocks"]}
    expected_paths = int(certificate["expected_paths"])
    frozen_lookback = int(certificate["frozen_lookback_days"])

    scenarios: dict[str, object] = {}
    p05_returns: list[float] = []
    p05_sharpes: list[float] = []
    p05_drawdowns: list[float] = []
    historical_drawdowns: list[float] = []
    for scenario_name in expected_scenarios:
        scenario = validation["scenarios"][scenario_name]
        metrics = scenario["metrics"][control_name]
        historical_drawdowns.append(float(metrics["max_drawdown"]))
        blocks: dict[str, object] = {}
        for block_name in sorted(expected_blocks, key=int):
            bootstrap = scenario["monte_carlo"][block_name]
            distribution = bootstrap["distributions"]["dual_momentum"]
            row = {
                "paths": int(bootstrap["paths"]),
                "block_length": int(bootstrap["block_length"]),
                "total_return_median": float(distribution["total_return"]["median"]),
                "total_return_p05": float(distribution["total_return"]["p05"]),
                "sharpe_median": float(distribution["sharpe"]["median"]),
                "sharpe_p05": float(distribution["sharpe"]["p05"]),
                "max_drawdown_median": float(
                    distribution["max_drawdown"]["median"]
                ),
                "max_drawdown_p05": float(distribution["max_drawdown"]["p05"]),
            }
            blocks[block_name] = row
            p05_returns.append(row["total_return_p05"])
            p05_sharpes.append(row["sharpe_p05"])
            p05_drawdowns.append(row["max_drawdown_p05"])
        scenarios[scenario_name] = {
            "validation": scenario["validation"],
            "historical_metrics": metrics,
            "bootstrap": blocks,
        }

    risk_summary = {
        "worst_observed_max_drawdown": min(historical_drawdowns),
        "worst_bootstrap_p05_total_return": min(p05_returns),
        "worst_bootstrap_p05_sharpe": min(p05_sharpes),
        "worst_bootstrap_p05_max_drawdown": min(p05_drawdowns),
        "bootstrap_cells_with_negative_p05_return": sum(
            value < 0.0 for value in p05_returns
        ),
        "bootstrap_cell_count": len(p05_returns),
    }
    checks = {
        "all_inputs_exist": not missing,
        "v6_validation_audit_passes": _audit_passed(validation_audit),
        "v20_evidence_audit_passes": _audit_passed(ledger_audit),
        "v20_halts_new_historical_search": ledger["decision"]
        == "halt_new_historical_model_search",
        "v20_has_no_active_candidate": not ledger["synthesis"][
            "active_candidate_versions"
        ],
        "frozen_lookback_matches_v6": int(
            validation_config["consensus"]["momentum_lookback"]
        )
        == frozen_lookback,
        "control_name_matches_lookback": control_name
        == f"dual_momentum_{frozen_lookback}",
        "scenario_registry_exact": list(validation["scenarios"])
        == expected_scenarios,
        "bootstrap_blocks_exact": all(
            set(validation["scenarios"][name]["monte_carlo"])
            == expected_blocks
            for name in expected_scenarios
        ),
        "bootstrap_paths_exact": all(
            validation["scenarios"][name]["monte_carlo"][block]["paths"]
            == expected_paths
            for name in expected_scenarios
            for block in expected_blocks
        ),
        "control_beats_buy_hold_in_frozen_scenarios": bool(
            validation["dual_momentum_beats_buy_hold_in_all_scenarios"]
        ),
        "control_risk_is_explicit": risk_summary[
            "bootstrap_cells_with_negative_p05_return"
        ]
        > 0
        and risk_summary["worst_observed_max_drawdown"] < -0.25,
        "results_are_finite": _finite(scenarios) and _finite(risk_summary),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Control certificate audit failed: {checks}")

    return {
        "version": "v21",
        "method": "artifact_only_deterministic_control_certification",
        "decision": "certify_as_research_control_only",
        "benchmark_status": "certified_research_control",
        "deployment_status": "not_authorized",
        "control": {
            "name": control_name,
            "assets": list(validation_config["data"]["assets"]),
            "lookback_days": frozen_lookback,
            "signal": "log_close_t_over_close_t_minus_30",
            "action": "long_highest_momentum_asset_if_max_positive_else_cash",
            "target_accounting": validation_config["target"]["mode"],
            "cost_bps_per_unit_turnover": float(
                validation_config["strategy"]["cost_bps"]
            ),
            "learned_parameters": 0,
            "lookback_tuning_authorized": False,
        },
        "risk_summary": risk_summary,
        "scenarios": scenarios,
        "source_hashes": {
            str(path): _sha256_file(path) for path in paths.values()
        },
        "audit": {"passed": True, "checks": checks},
    }


def _report(result: dict) -> str:
    risk = result["risk_summary"]
    lines = [
        "# TLM v21 Deterministic Research-Control Certificate",
        "",
        "## Decision",
        "",
        "**CERTIFIED AS A RESEARCH CONTROL ONLY. DEPLOYMENT IS NOT AUTHORIZED.**",
        "",
        "The frozen control uses no learned parameters: at each daily signal date it ranks BTC, ETH, and SOL by 30-day log close momentum, holds the strongest asset only when its momentum is positive, and otherwise holds cash. The backtest charges 10 bps per unit turnover and accounts returns open-to-open on the following daily target.",
        "",
        "Certification means the implementation and benchmark role are frozen and reproducible. It is not evidence of a safe or deployable strategy.",
        "",
        "## Frozen historical evidence",
        "",
        "| Scenario | Observations | Total return | Sharpe | Max DD |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, scenario in result["scenarios"].items():
        metrics = scenario["historical_metrics"]
        lines.append(
            f"| {name} | {metrics['observations']} | {metrics['total_return']:.2%} | "
            f"{metrics['sharpe']:.3f} | {metrics['max_drawdown']:.2%} |"
        )
    lines.extend([
        "",
        "## Bootstrap risk",
        "",
        f"- Worst observed max drawdown: **{risk['worst_observed_max_drawdown']:.2%}**",
        f"- Worst 5th-percentile bootstrap total return: **{risk['worst_bootstrap_p05_total_return']:.2%}**",
        f"- Worst 5th-percentile bootstrap Sharpe: **{risk['worst_bootstrap_p05_sharpe']:.3f}**",
        f"- Worst 5th-percentile bootstrap max drawdown: **{risk['worst_bootstrap_p05_max_drawdown']:.2%}**",
        f"- Negative 5th-percentile return cells: **{risk['bootstrap_cells_with_negative_p05_return']}/{risk['bootstrap_cell_count']}**",
        "",
        "These are paired circular block-bootstrap diagnostics with 3,000 paths at 7/21/63-day blocks in each frozen walk-forward scenario. Every lower-tail return cell is negative, so benchmark certification must not be read as capital authorization.",
        "",
        "## Restrictions",
        "",
        "- No lookback, threshold, universe, cost, or allocation tuning on the exposed history.",
        "- No live, shadow, paper, or real-capital execution is authorized.",
        "- No learned candidate may inherit this certificate.",
        "- New claims require a prospectively frozen, untouched evaluation protocol.",
        "",
    ])
    return "\n".join(lines)


def run_control_certificate(config: dict) -> dict[str, object]:
    result = build_control_certificate(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "certificate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "audit.json").write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
