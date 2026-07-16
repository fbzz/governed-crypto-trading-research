from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_public_snapshot.py"
SEEDER = ROOT / "scripts/seed_public_repo.py"


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "scripts").mkdir()
    (repo / "research").mkdir()
    (repo / "artifacts").mkdir()
    (repo / "data").mkdir()
    (repo / "docs/social").mkdir(parents=True)
    (repo / "README.md").write_text("# Public fixture\n", encoding="utf-8")
    (repo / "CITATION.cff").write_text(
        "cff-version: 1.2.0\ntitle: Public fixture\n", encoding="utf-8"
    )
    (repo / "src/app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "research/current.yaml").write_text(
        "schema_version: 1\nproject: TLM\n", encoding="utf-8"
    )
    shutil.copy2(BUILDER, repo / "scripts/build_public_snapshot.py")
    shutil.copy2(SEEDER, repo / "scripts/seed_public_repo.py")
    (repo / "artifacts/model.pt").write_bytes(b"opaque checkpoint")
    (repo / "data/panel.parquet").write_bytes(b"opaque table")
    (repo / "docs/social/image.png").write_bytes(b"opaque image metadata")
    (repo / ".gitignore").write_text("dist/\n.env*\n", encoding="utf-8")
    (repo / ".env.local").write_text("NOT_READ=fixture\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Snapshot Test")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo


def _run_builder(repo: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(repo / "scripts/build_public_snapshot.py"),
            "--repo-root",
            str(repo),
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_seeder(repo: Path, release: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(repo / "scripts/seed_public_repo.py"),
            "--release-dir",
            str(release),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_public_snapshot_uses_immutable_head_and_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    repo = _fixture(tmp_path)
    output = repo / "dist/public"
    first = _run_builder(repo, output)
    assert first.returncode == 0, first.stderr
    archive = output / "tlm-public-source.tar.gz"
    manifest_path = output / "tlm-public-source.manifest.json"
    first_archive = archive.read_bytes()
    first_manifest = manifest_path.read_bytes()
    private_head = _git(repo, "rev-parse", "HEAD").encode("ascii")

    with tarfile.open(archive, "r:gz") as package:
        names = package.getnames()
        assert "README.md" in names
        assert "CITATION.cff" in names
        assert "src/app.py" in names
        assert "PUBLIC_SNAPSHOT_MANIFEST.json" in names
        assert all(
            not name.startswith(("artifacts/", "data/", "docs/social/", ".git/"))
            for name in names
        )
        embedded = json.load(package.extractfile("PUBLIC_SNAPSHOT_MANIFEST.json"))

    manifest = json.loads(first_manifest)
    assert manifest["history_included"] is False
    assert manifest["current_source_head_and_tree_identifiers_included"] is False
    assert "source_commit" not in manifest
    assert private_head not in first_archive
    assert private_head not in first_manifest
    assert manifest["license_status"] == "no_license_all_rights_reserved"
    assert manifest["exclusions"]["total_tracked_files"] == 3
    assert manifest["archive"]["sha256"] == hashlib.sha256(first_archive).hexdigest()
    assert embedded["source_tree_sha256"] == manifest["source_tree_sha256"]
    assert all(row["mode"] in {"100644", "100755"} for row in manifest["files"])

    (repo / "README.md").write_text("dirty worktree content\n", encoding="utf-8")
    (repo / "UNTRACKED_SECRET.txt").write_text("not read\n", encoding="utf-8")
    second = _run_builder(repo, output)
    assert second.returncode == 0, second.stderr
    assert archive.read_bytes() == first_archive
    assert manifest_path.read_bytes() == first_manifest


def test_public_snapshot_redacts_secret_value_from_failure(tmp_path: Path) -> None:
    repo = _fixture(tmp_path)
    (repo / "configs").mkdir()
    value = "sk-proj-" + "A" * 32
    (repo / "configs/unsafe.yaml").write_text(f"api_key: {value}\n", encoding="utf-8")
    _git(repo, "add", "configs/unsafe.yaml")
    _git(repo, "commit", "-qm", "add unsafe fixture")
    result = _run_builder(repo, repo / "dist/public")
    assert result.returncode == 2
    assert "sensitive pattern openai_api_key: configs/unsafe.yaml" in result.stderr
    assert value not in result.stderr


def test_public_snapshot_rejects_sensitive_parent_path(tmp_path: Path) -> None:
    repo = _fixture(tmp_path)
    (repo / "private").mkdir()
    (repo / "private/config.txt").write_text("placeholder\n", encoding="utf-8")
    _git(repo, "add", "private/config.txt")
    _git(repo, "commit", "-qm", "add forbidden path fixture")
    result = _run_builder(repo, repo / "dist/public")
    assert result.returncode == 2
    assert "forbidden sensitive path: private/config.txt" in result.stderr


def test_public_snapshot_rejects_sensitive_file_type(tmp_path: Path) -> None:
    repo = _fixture(tmp_path)
    (repo / "src/private.pem").write_text("placeholder\n", encoding="utf-8")
    _git(repo, "add", "src/private.pem")
    _git(repo, "commit", "-qm", "add forbidden fixture")
    result = _run_builder(repo, repo / "dist/public")
    assert result.returncode == 2
    assert "forbidden file type: src/private.pem" in result.stderr


@pytest.mark.parametrize(
    "content",
    [
        '{"password": "' + "A" * 32 + '"}\n',
        "AWS_SECRET_ACCESS_KEY=" + "B" * 40 + "\n",
    ],
)
def test_public_snapshot_rejects_generic_secret_assignments(
    tmp_path: Path,
    content: str,
) -> None:
    repo = _fixture(tmp_path)
    (repo / "configs").mkdir()
    (repo / "configs/unsafe.json").write_text(content, encoding="utf-8")
    _git(repo, "add", "configs/unsafe.json")
    _git(repo, "commit", "-qm", "add generic secret fixture")
    result = _run_builder(repo, repo / "dist/public")
    assert result.returncode == 2
    assert "sensitive pattern generic_secret_assignment" in result.stderr
    assert "A" * 32 not in result.stderr
    assert "B" * 40 not in result.stderr


def test_public_snapshot_rejects_nested_git_path() -> None:
    code = (
        "import runpy; "
        f"m=runpy.run_path({str(BUILDER)!r}); "
        "m['_validate_public_path']('src/.git/config.py')"
    )
    result = subprocess.run(
        ["python3", "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "forbidden sensitive path: src/.git/config.py" in result.stderr


def test_public_snapshot_rejects_destructive_output_path(tmp_path: Path) -> None:
    repo = _fixture(tmp_path)
    readme_before = (repo / "README.md").read_bytes()
    result = _run_builder(repo, repo)
    assert result.returncode == 2
    assert "public output must be exactly" in result.stderr
    assert (repo / "README.md").read_bytes() == readme_before
    assert (repo / ".git").is_dir()


def test_public_snapshot_requires_immutable_builder(tmp_path: Path) -> None:
    repo = _fixture(tmp_path)
    builder = repo / "scripts/build_public_snapshot.py"
    builder.write_text(builder.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    result = _run_builder(repo, repo / "dist/public")
    assert result.returncode == 2
    assert "running snapshot builder does not match" in result.stderr


def test_failed_build_preserves_prior_complete_release(tmp_path: Path) -> None:
    repo = _fixture(tmp_path)
    output = repo / "dist/public"
    first = _run_builder(repo, output)
    assert first.returncode == 0, first.stderr
    archive_before = (output / "tlm-public-source.tar.gz").read_bytes()
    manifest_before = (output / "tlm-public-source.manifest.json").read_bytes()

    (repo / "configs").mkdir()
    (repo / "configs/unsafe.yaml").write_text(
        "access_token: " + "B" * 40 + "\n", encoding="utf-8"
    )
    _git(repo, "add", "configs/unsafe.yaml")
    _git(repo, "commit", "-qm", "add unsafe fixture")
    failed = _run_builder(repo, output)
    assert failed.returncode == 2
    assert (output / "tlm-public-source.tar.gz").read_bytes() == archive_before
    assert (output / "tlm-public-source.manifest.json").read_bytes() == manifest_before


def test_public_seeder_creates_one_parentless_verified_commit(tmp_path: Path) -> None:
    repo = _fixture(tmp_path)
    output = repo / "dist/public"
    built = _run_builder(repo, output)
    assert built.returncode == 0, built.stderr
    seeded = _run_seeder(repo, output)
    assert seeded.returncode == 0, seeded.stderr
    public_repo = output / "repository"
    repository_manifest = json.loads(
        (output / "tlm-public-repository.manifest.json").read_text(encoding="utf-8")
    )
    root_commit = _git(public_repo, "rev-parse", "HEAD")

    assert _git(public_repo, "symbolic-ref", "HEAD") == "refs/heads/main"
    assert _git(public_repo, "rev-list", "--count", "--all") == "1"
    assert len(_git(public_repo, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 1
    assert _git(public_repo, "remote") == ""
    assert _git(public_repo, "tag", "--list") == ""
    assert _git(public_repo, "status", "--porcelain=v1") == ""
    assert repository_manifest["root_commit"] == root_commit
    assert repository_manifest["commit_count"] == 1
    assert repository_manifest["history_is_parentless"] is True
    assert repository_manifest["author_email"] == "tlm-research@users.noreply.github.com"

    shutil.rmtree(output)
    rebuilt = _run_builder(repo, output)
    assert rebuilt.returncode == 0, rebuilt.stderr
    reseeded = _run_seeder(repo, output)
    assert reseeded.returncode == 0, reseeded.stderr
    assert _git(output / "repository", "rev-parse", "HEAD") == root_commit


def test_public_seeder_rejects_tampered_archive(tmp_path: Path) -> None:
    repo = _fixture(tmp_path)
    output = repo / "dist/public"
    built = _run_builder(repo, output)
    assert built.returncode == 0, built.stderr
    archive = output / "tlm-public-source.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"tamper")
    seeded = _run_seeder(repo, output)
    assert seeded.returncode == 2
    assert "source archive size mismatch" in seeded.stderr
    assert not (output / "repository").exists()


def test_public_seeder_requires_immutable_seeder(tmp_path: Path) -> None:
    repo = _fixture(tmp_path)
    output = repo / "dist/public"
    built = _run_builder(repo, output)
    assert built.returncode == 0, built.stderr
    seeder = repo / "scripts/seed_public_repo.py"
    seeder.write_text(seeder.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    seeded = _run_seeder(repo, output)
    assert seeded.returncode == 2
    assert "running public seeder does not match" in seeded.stderr
    assert not (output / "repository").exists()


def test_public_seeder_requires_immutable_builder(tmp_path: Path) -> None:
    repo = _fixture(tmp_path)
    output = repo / "dist/public"
    built = _run_builder(repo, output)
    assert built.returncode == 0, built.stderr
    builder = repo / "scripts/build_public_snapshot.py"
    builder.write_text(builder.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    seeded = _run_seeder(repo, output)
    assert seeded.returncode == 2
    assert "running snapshot builder does not match" in seeded.stderr
    assert not (output / "repository").exists()
