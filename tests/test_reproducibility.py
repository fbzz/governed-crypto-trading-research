import json

from tlm.reproducibility import (
    build_reproducibility_manifest,
    run_reproducibility_bundle,
    verify_reproducibility_manifest,
)


def _config(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "src" / "code.py").write_text("VALUE = 1\n")
    (tmp_path / "tests" / "test_code.py").write_text("def test_ok(): assert True\n")
    (tmp_path / "configs" / "config.yaml").write_text("seed: 1\n")
    (tmp_path / "README.md").write_text("fixture\n")
    audits = []
    for index in range(3):
        path = tmp_path / "artifacts" / f"audit_{index}.json"
        path.write_text(json.dumps({"passed": True}))
        audits.append(str(path.relative_to(tmp_path)))
    return {
        "reproducibility_bundle": {
            "project_root": str(tmp_path),
            "include_roots": ["src", "tests", "configs"],
            "include_files": ["README.md", *audits],
            "exclude_suffixes": [".pyc"],
            "packages": ["pytest"],
            "test_command": ["python3", "-m", "pytest", "-q"],
            "run_tests": False,
            "required_decision_audits": audits,
            "manifest_path": "output/manifest.json",
        },
        "output_dir": "output",
    }


def test_manifest_detects_content_change(tmp_path):
    config = _config(tmp_path)
    manifest = build_reproducibility_manifest(config)
    assert verify_reproducibility_manifest(manifest, tmp_path)["passed"]
    (tmp_path / "src" / "code.py").write_text("VALUE = 2\n")
    verification = verify_reproducibility_manifest(manifest, tmp_path)
    assert not verification["passed"]
    assert verification["failures"][0]["reason"] == "content_mismatch"


def test_bundle_writes_verified_manifest(tmp_path):
    config = _config(tmp_path)
    result = run_reproducibility_bundle(config)
    output = tmp_path / "output"
    assert result["audit"]["passed"]
    assert result["verification"]["passed"]
    assert (output / "manifest.json").is_file()
    assert (output / "test_result.json").is_file()
    assert (output / "report.md").is_file()
