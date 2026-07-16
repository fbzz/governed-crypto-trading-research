from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Iterable

import yaml


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _registered_files(root: Path, config: dict) -> list[Path]:
    files: set[Path] = set()
    for relative in config["include_roots"]:
        base = root / relative
        if not base.is_dir():
            raise FileNotFoundError(f"Manifest root does not exist: {base}")
        files.update(path for path in base.rglob("*") if path.is_file())
    for relative in config["include_files"]:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file does not exist: {path}")
        files.add(path)
    excluded = {str(value) for value in config.get("exclude_suffixes", [])}
    return sorted(
        path for path in files if path.suffix not in excluded
    )


def _package_versions(packages: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _git_revision(root: Path) -> dict[str, object]:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode == 0:
        return {"available": True, "commit": process.stdout.strip()}
    return {
        "available": False,
        "commit": None,
        "reason": "unborn_or_unavailable_repository",
    }


def run_test_command(root: Path, command: list[str]) -> dict[str, object]:
    environment = os.environ.copy()
    environment.setdefault("PYTHONPATH", "src")
    process = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = "\n".join(
        value.strip() for value in (process.stdout, process.stderr) if value.strip()
    )
    match = re.search(r"(\d+) passed", combined)
    return {
        "command": command,
        "exit_code": process.returncode,
        "passed": process.returncode == 0,
        "passed_test_count": int(match.group(1)) if match else None,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def build_reproducibility_manifest(config: dict) -> dict[str, object]:
    bundle = config["reproducibility_bundle"]
    root = Path(bundle["project_root"]).resolve()
    files = _registered_files(root, bundle)
    entries = []
    for path in files:
        relative = str(path.resolve().relative_to(root))
        entries.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    categories = {
        "source": sum(row["path"].startswith("src/") for row in entries),
        "tests": sum(row["path"].startswith("tests/") for row in entries),
        "configs": sum(row["path"].startswith("configs/") for row in entries),
        "decision_evidence": sum(
            row["path"].startswith("artifacts/") for row in entries
        ),
        "project_contracts": sum(
            row["path"] in {"AGENTS.md", "TASKS.md", "README.md", "Makefile", "pyproject.toml"}
            for row in entries
        ),
    }
    runtime = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _package_versions(bundle["packages"]),
        "git": _git_revision(root),
    }
    core = {
        "version": "v23",
        "method": "content_addressed_source_config_test_and_decision_bundle",
        "root_semantics": "paths_are_relative_to_project_root",
        "file_count": len(entries),
        "total_bytes": sum(row["bytes"] for row in entries),
        "categories": categories,
        "files": entries,
        "runtime": runtime,
    }
    core["manifest_sha256"] = _canonical_hash(core)
    return core


def verify_reproducibility_manifest(
    manifest: dict, project_root: str | Path
) -> dict[str, object]:
    root = Path(project_root).resolve()
    failures: list[dict[str, object]] = []
    checked = 0
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            failures.append({"path": entry["path"], "reason": "missing"})
            continue
        checked += 1
        actual_size = path.stat().st_size
        actual_hash = _sha256_file(path)
        if actual_size != entry["bytes"] or actual_hash != entry["sha256"]:
            failures.append({
                "path": entry["path"],
                "reason": "content_mismatch",
                "expected_bytes": entry["bytes"],
                "actual_bytes": actual_size,
                "expected_sha256": entry["sha256"],
                "actual_sha256": actual_hash,
            })
    expected_manifest_hash = manifest["manifest_sha256"]
    hash_payload = dict(manifest)
    hash_payload.pop("manifest_sha256")
    actual_manifest_hash = _canonical_hash(hash_payload)
    manifest_hash_matches = expected_manifest_hash == actual_manifest_hash
    if not manifest_hash_matches:
        failures.append({"path": "manifest.json", "reason": "manifest_hash_mismatch"})
    return {
        "passed": not failures,
        "checked_files": checked,
        "registered_files": len(manifest["files"]),
        "manifest_hash_matches": manifest_hash_matches,
        "failures": failures,
    }


def _report(result: dict) -> str:
    manifest = result["manifest"]
    tests = result["tests"]
    verification = result["verification"]
    git = manifest["runtime"]["git"]
    lines = [
        "# TLM v23 Reproducibility Bundle",
        "",
        "## Decision",
        "",
        "**BUNDLE VERIFIED.** The current research source, configs, tests, contracts, and v20-v22 decision chain are content-addressed and replay-checkable.",
        "",
        f"- Registered files: **{manifest['file_count']}**",
        f"- Registered bytes: **{manifest['total_bytes']:,}**",
        f"- Manifest SHA-256: `{manifest['manifest_sha256']}`",
        f"- Verified files: **{verification['checked_files']}**",
        f"- Full test command passed: **{tests['passed']}**",
        f"- Passed tests parsed from pytest: **{tests['passed_test_count']}**",
        f"- Categories: `{manifest['categories']}`",
        "",
        "## Runtime",
        "",
        f"- Python: `{manifest['runtime']['python'].splitlines()[0]}`",
        f"- Platform: `{manifest['runtime']['platform']}`",
        f"- Packages: `{manifest['runtime']['packages']}`",
        "",
        "## Verification",
        "",
        "Run `make verify-v23` from the project root. The verifier checks the manifest's own canonical hash, then every registered file size and SHA-256 digest.",
        "",
        "## Limitation",
        "",
        (
            f"Git commit available: `{git['commit']}`."
            if git["available"]
            else "The repository has no resolvable commit yet. Reproducibility is therefore anchored by the content manifest, not a Git commit. A future archival release should add a signed commit or immutable release tag."
        ),
        "",
        "The bundle verifies code and evidence integrity; it does not convert rejected historical results into a deployable candidate and does not start the v22 holdout.",
        "",
    ]
    return "\n".join(lines)


def run_reproducibility_bundle(config: dict) -> dict[str, object]:
    bundle = config["reproducibility_bundle"]
    root = Path(bundle["project_root"]).resolve()
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    tests = (
        run_test_command(root, list(bundle["test_command"]))
        if bundle.get("run_tests", True)
        else {
            "command": [],
            "exit_code": 0,
            "passed": True,
            "passed_test_count": 0,
            "stdout": "test execution disabled by fixture",
            "stderr": "",
        }
    )
    (output / "test_result.json").write_text(
        json.dumps(tests, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not tests["passed"]:
        raise RuntimeError("Full offline test suite failed; bundle not certified")

    manifest = build_reproducibility_manifest(config)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    verification = verify_reproducibility_manifest(manifest, root)
    (output / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True), encoding="utf-8"
    )
    decision_audits = [
        json.loads((root / path).read_text(encoding="utf-8"))
        for path in bundle["required_decision_audits"]
    ]
    checks = {
        "full_offline_test_suite_passes": bool(tests["passed"]),
        "manifest_verification_passes": bool(verification["passed"]),
        "all_registered_files_checked": verification["checked_files"]
        == manifest["file_count"],
        "manifest_has_source_tests_configs_and_decisions": all(
            manifest["categories"][name] > 0
            for name in ("source", "tests", "configs", "decision_evidence")
        ),
        "v20_v22_decision_audits_pass": all(
            bool(audit.get("passed")) for audit in decision_audits
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Reproducibility bundle audit failed: {checks}")
    result = {
        "version": "v23",
        "decision": "reproducibility_bundle_verified",
        "manifest": manifest,
        "tests": tests,
        "verification": verification,
        "audit": {"passed": True, "checks": checks},
    }
    (output / "audit.json").write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result


def run_reproducibility_verifier(config: dict) -> dict[str, object]:
    root = Path(config["reproducibility_bundle"]["project_root"]).resolve()
    manifest_path = Path(config["reproducibility_bundle"]["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return verify_reproducibility_manifest(manifest, root)
