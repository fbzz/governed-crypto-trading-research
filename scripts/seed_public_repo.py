#!/usr/bin/env python3
"""Seed and verify a one-commit public Git repository from a TLM snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
from typing import Iterable


ARCHIVE_NAME = "tlm-public-source.tar.gz"
MANIFEST_NAME = "tlm-public-source.manifest.json"
INNER_MANIFEST_NAME = "PUBLIC_SNAPSHOT_MANIFEST.json"
PUBLIC_REPOSITORY_DIR = "repository"
PUBLIC_REPOSITORY_MANIFEST_NAME = "tlm-public-repository.manifest.json"
PUBLIC_AUTHOR_NAME = "TLM Research"
PUBLIC_AUTHOR_EMAIL = "tlm-research@users.noreply.github.com"
PUBLIC_COMMIT_DATE = "2026-07-16T00:00:00Z"
PUBLIC_COMMIT_TIMESTAMP = "1784160000"
PUBLIC_COMMIT_MESSAGE = "Publish TLM source snapshot\n"
POLICY_VERSION = "tlm-public-source-v2"
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 12 * 1024 * 1024
MAX_FILE_COUNT = 1000

PUBLIC_ROOT_FILES = {
    ".gitattributes", ".gitignore", "AGENTS.md", "AUTONOMOUS_TRAINING_LOOP.md",
    "CITATION.cff",
    "COPYING", "LICENCE", "LICENSE", "Makefile", "README.md", "STATUS.md",
    "TASKS.md", "docs/PUBLIC_RELEASE.md", "pyproject.toml", "research/current.yaml",
}
PUBLIC_PREFIXES = (
    ".agents/", "configs/", "docs/research/", "prompts/", "research/amendments/",
    "research/authorizations/", "research/candidates/", "research/experiments/",
    "research/incidents/", "research/phase_contracts/", "research/receipts/",
    "research/schemas/", "research/waivers/", "scripts/", "src/", "tests/",
)
ALLOWED_TEXT_SUFFIXES = {
    ".cff", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"
}
FORBIDDEN_SUFFIXES = {
    ".7z", ".arrow", ".bin", ".bz2", ".cer", ".ckpt", ".crt", ".csv",
    ".db", ".dylib", ".exe", ".feather", ".gif", ".gz", ".h5", ".hdf5",
    ".ico", ".jar", ".jks", ".jpeg", ".jpg", ".joblib", ".jsonl", ".key",
    ".keystore", ".npy", ".npz", ".p12", ".parquet", ".pem", ".pfx",
    ".pickle", ".pkl", ".png", ".pt", ".pth", ".safetensors", ".so",
    ".sqlite", ".sqlite3", ".svg", ".tar", ".tgz", ".tsv", ".webp",
    ".whl", ".xz", ".zip",
}
FORBIDDEN_BASENAMES = {
    ".env", ".gitmodules", ".npmrc", ".pypirc", "id_ed25519", "id_rsa"
}
FORBIDDEN_COMPONENTS = {
    ".aws", ".azure", ".git", ".gnupg", ".ssh", "backup", "backups",
    "credential", "credentials", "private", "secret", "secrets", "wallet", "wallets",
}
SECRET_PATTERNS = (
    ("private_key", re.compile(br"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(br"(?:AKIA|ASIA)[A-Z0-9]{16}")),
    ("github_legacy_token", re.compile(br"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("github_fine_grained_token", re.compile(br"github_pat_[A-Za-z0-9_]{20,}")),
    ("hugging_face_token", re.compile(br"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}")),
    (
        "openai_api_key",
        re.compile(br"(?<![A-Za-z0-9])sk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{20,}"),
    ),
    ("slack_token", re.compile(br"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("google_api_key", re.compile(br"AIza[0-9A-Za-z_-]{30,}")),
    ("gitlab_token", re.compile(br"glpat-[A-Za-z0-9_-]{20,}")),
    ("npm_token", re.compile(br"npm_[A-Za-z0-9]{30,}")),
    ("pypi_token", re.compile(br"pypi-[A-Za-z0-9_-]{40,}")),
    ("stripe_live_key", re.compile(br"(?:sk|rk)_live_[A-Za-z0-9]{16,}")),
    ("credential_url", re.compile(br"[A-Za-z][A-Za-z0-9+.-]*://[^/\s:]+:[^@\s]+@")),
    (
        "presigned_url",
        re.compile(b"(?i:X-" + b"Amz-(?:Credential|Signature)=|X-" + b"Goog-Signature=)"),
    ),
    (
        "bearer_token",
        re.compile(br"(?i:Authorization\s*:\s*Bearer\s+[A-Za-z0-9._~+/-]{16,})"),
    ),
    ("jwt", re.compile(br"eyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}")),
    ("unix_personal_home", re.compile(br"/(?:Users|home)/[A-Za-z0-9._-]+")),
    ("windows_personal_home", re.compile(br"(?i:[A-Z]:\\Users\\[A-Za-z0-9._ -]+)")),
    (
        "private_ipv4",
        re.compile(
            br"(?<!\d)(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
            br"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?!\d)"
        ),
    ),
    (
        "generic_secret_assignment",
        re.compile(
            br"(?ix)(?:^|[\s,{])[\"']?"
            br"(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|"
            br"client[_-]?secret|private[_-]?key|(?:aws[_-]?)?secret[_-]?access[_-]?key)"
            br"[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9+/_.~-]{16,}"
        ),
    ),
)
EMAIL_PATTERN = re.compile(
    br"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
ALLOWED_EMAILS = {b"tlm-research@users.noreply.github.com"}
ALLOWED_FIXTURE_EMAIL_DOMAINS = (b"@example.com", b"@example.invalid")
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class SeedError(RuntimeError):
    """Raised when a public repository cannot be seeded safely."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _content_digest(rows: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        path = str(row["path"]).encode("utf-8")
        mode = str(row["mode"]).encode("ascii")
        size = int(row["size"])
        digest.update(len(path).to_bytes(4, "big"))
        digest.update(path)
        digest.update(mode)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(str(row["sha256"])))
    return digest.hexdigest()


def _validate_archive_path(relative: str) -> PurePosixPath:
    if unicodedata.normalize("NFC", relative) != relative:
        raise SeedError(f"non-NFC archive path: {relative!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in relative):
        raise SeedError(f"control character in archive path: {relative!r}")
    if "\\" in relative:
        raise SeedError(f"backslash in archive path: {relative!r}")
    path = PurePosixPath(relative)
    if (
        path.is_absolute() or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(part.casefold() == ".git" for part in path.parts)
    ):
        raise SeedError(f"unsafe archive path: {relative!r}")
    return path


def _enforce_source_policy(relative: str, value: bytes) -> None:
    path = _validate_archive_path(relative)
    basename = path.name.casefold()
    suffix = path.suffix.casefold()
    components = {part.casefold() for part in path.parts}
    if basename in FORBIDDEN_BASENAMES or basename.startswith(".env."):
        raise SeedError(f"forbidden sensitive filename: {relative}")
    if suffix in FORBIDDEN_SUFFIXES:
        raise SeedError(f"forbidden file type: {relative}")
    if components & FORBIDDEN_COMPONENTS:
        raise SeedError(f"forbidden sensitive path: {relative}")
    if relative not in PUBLIC_ROOT_FILES and not relative.startswith(PUBLIC_PREFIXES):
        raise SeedError(f"path is outside the public allowlist: {relative}")
    if relative not in PUBLIC_ROOT_FILES and suffix not in ALLOWED_TEXT_SUFFIXES:
        raise SeedError(f"unapproved public source type: {relative}")
    if len(value) > MAX_FILE_BYTES or b"\0" in value:
        raise SeedError(f"non-text or oversized public source: {relative}")
    try:
        value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SeedError(f"non-UTF-8 public source: {relative}") from error
    if value.startswith(LFS_POINTER_PREFIX):
        raise SeedError(f"Git LFS pointer is not allowed: {relative}")
    for rule, pattern in SECRET_PATTERNS:
        if pattern.search(value):
            raise SeedError(f"sensitive pattern {rule}: {relative}")
    for match in EMAIL_PATTERN.finditer(value):
        email = match.group(0).lower()
        if email in ALLOWED_EMAILS or email.endswith(ALLOWED_FIXTURE_EMAIL_DOMAINS):
            continue
        raise SeedError(f"non-public email address: {relative}")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SeedError(f"invalid JSON file: {path.name}") from error
    if not isinstance(value, dict):
        raise SeedError(f"JSON root must be an object: {path.name}")
    return value


def _validate_manifest_header(manifest: dict[str, object]) -> None:
    policy = manifest.get("publication_policy")
    scan = manifest.get("built_in_scan")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("project") != "TLM"
        or manifest.get("history_included") is not False
        or manifest.get("current_source_head_and_tree_identifiers_included") is not False
        or not isinstance(policy, dict)
        or policy.get("version") != POLICY_VERSION
        or not isinstance(scan, dict)
        or scan.get("status") != "zero_findings"
    ):
        raise SeedError("source manifest publication header is invalid")
    if (
        not isinstance(manifest.get("file_count"), int)
        or not 0 < int(manifest["file_count"]) <= MAX_FILE_COUNT
        or not isinstance(manifest.get("total_bytes"), int)
        or not 0 < int(manifest["total_bytes"]) <= MAX_TOTAL_BYTES
    ):
        raise SeedError("source manifest size caps are invalid")


def _validate_manifest_rows(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise SeedError("source manifest files must be a list")
    expected: dict[str, dict[str, object]] = {}
    collision_keys: dict[str, str] = {}
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise SeedError("source manifest file row must be an object")
        relative = raw_row.get("path")
        mode = raw_row.get("mode")
        size = raw_row.get("size")
        sha256 = raw_row.get("sha256")
        if not isinstance(relative, str):
            raise SeedError("source manifest file path must be text")
        _validate_archive_path(relative)
        collision_key = relative.casefold()
        if collision_key in collision_keys:
            raise SeedError(f"case-fold or duplicate manifest collision: {relative}")
        collision_keys[collision_key] = relative
        if mode not in {"100644", "100755"}:
            raise SeedError(f"unsupported source manifest mode: {relative}")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= MAX_FILE_BYTES:
            raise SeedError(f"invalid source manifest size: {relative}")
        if (
            not isinstance(sha256, str) or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise SeedError(f"invalid source manifest SHA-256: {relative}")
        expected[relative] = dict(raw_row)
    if manifest.get("file_count") != len(expected):
        raise SeedError("source manifest file count mismatch")
    if manifest.get("total_bytes") != sum(int(row["size"]) for row in expected.values()):
        raise SeedError("source manifest total byte count mismatch")
    if manifest.get("source_tree_sha256") != _content_digest([expected[path] for path in sorted(expected)]):
        raise SeedError("source manifest tree digest mismatch")
    return expected


def _validate_archive(release_dir: Path) -> tuple[dict[str, object], dict[str, tuple[bytes, str]]]:
    manifest = _load_json(release_dir / MANIFEST_NAME)
    _validate_manifest_header(manifest)
    archive_path = release_dir / ARCHIVE_NAME
    archive_row = manifest.get("archive")
    if not isinstance(archive_row, dict) or archive_row.get("path") != ARCHIVE_NAME:
        raise SeedError("source manifest archive row is invalid")
    if archive_row.get("size") != archive_path.stat().st_size:
        raise SeedError("source archive size mismatch")
    if archive_row.get("sha256") != _sha256_file(archive_path):
        raise SeedError("source archive SHA-256 mismatch")

    expected_rows = _validate_manifest_rows(manifest)
    expected_names = set(expected_rows) | {INNER_MANIFEST_NAME}
    values: dict[str, tuple[bytes, str]] = {}
    collision_keys: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != expected_names:
            raise SeedError("source archive member set does not match the manifest")
        for member in members:
            _validate_archive_path(member.name)
            collision_key = member.name.casefold()
            if collision_key in collision_keys:
                raise SeedError(f"case-fold archive collision: {member.name}")
            collision_keys.add(collision_key)
            if (
                not member.isfile() or member.linkname or member.uid != 0 or member.gid != 0
                or member.mtime != 0 or member.uname not in {"", None}
                or member.gname not in {"", None} or member.pax_headers
                or member.mode not in {0o644, 0o755}
            ):
                raise SeedError(f"source archive metadata is not normalized: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise SeedError(f"source archive member cannot be read: {member.name}")
            value = extracted.read()
            mode = "100755" if member.mode == 0o755 else "100644"
            if member.name != INNER_MANIFEST_NAME:
                row = expected_rows[member.name]
                if mode != row["mode"] or len(value) != row["size"] or _sha256_bytes(value) != row["sha256"]:
                    raise SeedError(f"source archive member digest mismatch: {member.name}")
                _enforce_source_policy(member.name, value)
            values[member.name] = (value, mode)

    try:
        embedded = json.loads(values[INNER_MANIFEST_NAME][0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SeedError("embedded source manifest is invalid") from error
    if embedded != {key: value for key, value in manifest.items() if key != "archive"}:
        raise SeedError("embedded and adjacent source manifests differ")

    builder_row = expected_rows.get("scripts/build_public_snapshot.py")
    seeder_row = expected_rows.get("scripts/seed_public_repo.py")
    if builder_row is None or seeder_row is None:
        raise SeedError("immutable public release tools are absent from the manifest")
    builder_path = Path(__file__).resolve().with_name("build_public_snapshot.py")
    if _sha256_file(builder_path) != builder_row["sha256"]:
        raise SeedError("running snapshot builder does not match the verified archive")
    if _sha256_file(Path(__file__).resolve()) != seeder_row["sha256"]:
        raise SeedError("running public seeder does not match the verified archive")
    if manifest.get("builder_sha256") != builder_row["sha256"]:
        raise SeedError("source manifest builder identity mismatch")
    return manifest, values


def _extract_values(repository: Path, values: dict[str, tuple[bytes, str]]) -> None:
    resolved_repository = repository.resolve()
    for relative in sorted(values):
        path = _validate_archive_path(relative)
        target = repository.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.resolve(strict=False).relative_to(resolved_repository)
        except ValueError as error:
            raise SeedError(f"archive extraction escapes repository: {relative}") from error
        value, mode = values[relative]
        with target.open("xb") as handle:
            handle.write(value)
        target.chmod(0o755 if mode == "100755" else 0o644)


def _clean_git_environment(scratch: Path) -> tuple[str, dict[str, str]]:
    git = shutil.which("git")
    if git is None:
        raise SeedError("git executable is unavailable")
    git_path = str(Path(git).resolve())
    home = scratch / "home"
    xdg = scratch / "xdg"
    home.mkdir()
    xdg.mkdir()
    return git_path, {
        "PATH": f"{Path(git_path).parent}:/usr/bin:/bin",
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(xdg),
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
    }


def _run_git(
    git: str,
    environment: dict[str, str],
    *arguments: str,
    cwd: Path | None = None,
    input_value: bytes | None = None,
) -> bytes:
    result = subprocess.run(
        [git, "--no-replace-objects", *arguments], cwd=cwd, env=environment,
        input=input_value, check=False, capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise SeedError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout


def _git_inventory(
    repository: Path,
    git: str,
    environment: dict[str, str],
    tree: bool,
) -> tuple[str | None, dict[str, dict[str, object]]]:
    if tree:
        tree_oid = _run_git(git, environment, "-C", str(repository), "rev-parse", "HEAD^{tree}").decode().strip()
        raw = _run_git(git, environment, "-C", str(repository), "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    else:
        tree_oid = None
        raw = _run_git(git, environment, "-C", str(repository), "ls-files", "--stage", "-z")
    inventory: dict[str, dict[str, object]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        if tree:
            mode_value, kind_value, oid_value = metadata.split(b" ", 2)
            if kind_value != b"blob":
                raise SeedError("public tree contains a non-blob")
        else:
            mode_value, oid_value, stage_value = metadata.split(b" ", 2)
            if stage_value != b"0":
                raise SeedError("public index contains a non-zero stage")
        relative = encoded_path.decode("utf-8")
        value = _run_git(git, environment, "-C", str(repository), "cat-file", "blob", oid_value.decode())
        inventory[relative] = {"mode": mode_value.decode(), "sha256": _sha256_bytes(value), "size": len(value)}
    return tree_oid, inventory


def _expected_inventory(values: dict[str, tuple[bytes, str]]) -> dict[str, dict[str, object]]:
    return {
        relative: {"mode": mode, "sha256": _sha256_bytes(value), "size": len(value)}
        for relative, (value, mode) in values.items()
    }


def _verify_repository(
    repository: Path,
    git: str,
    environment: dict[str, str],
    expected: dict[str, dict[str, object]],
) -> tuple[str, str]:
    head_ref = _run_git(git, environment, "-C", str(repository), "symbolic-ref", "HEAD").decode().strip()
    refs = _run_git(
        git, environment, "-C", str(repository), "for-each-ref", "--format=%(refname)"
    ).decode().splitlines()
    count = _run_git(git, environment, "-C", str(repository), "rev-list", "--count", "--all").decode().strip()
    parent_line = _run_git(
        git, environment, "-C", str(repository), "rev-list", "--parents", "--max-count=1", "HEAD"
    ).decode().strip()
    if head_ref != "refs/heads/main" or refs != ["refs/heads/main"] or count != "1" or len(parent_line.split()) != 1:
        raise SeedError("public repository is not exactly one parentless main commit")
    identity = _run_git(
        git, environment, "-C", str(repository), "show", "-s",
        "--format=%an%x00%ae%x00%cn%x00%ce%x00%at%x00%ct", "HEAD",
    ).decode().rstrip("\n").split("\0")
    expected_identity = [
        PUBLIC_AUTHOR_NAME, PUBLIC_AUTHOR_EMAIL, PUBLIC_AUTHOR_NAME, PUBLIC_AUTHOR_EMAIL,
        PUBLIC_COMMIT_TIMESTAMP, PUBLIC_COMMIT_TIMESTAMP,
    ]
    if identity != expected_identity:
        raise SeedError("public repository commit identity or timestamp differs")
    if _run_git(git, environment, "-C", str(repository), "remote").strip():
        raise SeedError("public repository unexpectedly has a remote")
    if _run_git(git, environment, "-C", str(repository), "tag", "--list").strip():
        raise SeedError("public repository unexpectedly has a tag")
    forbidden = (
        repository / ".git/objects/info/alternates", repository / ".git/shallow",
        repository / ".git/info/grafts", repository / ".git/refs/replace",
    )
    if any(path.exists() for path in forbidden):
        raise SeedError("public repository has alternate, shallow, graft, or replace state")
    if _run_git(
        git, environment, "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all"
    ).strip():
        raise SeedError("public repository worktree is not clean")
    tree_oid, tree_inventory = _git_inventory(repository, git, environment, tree=True)
    if tree_inventory != expected:
        raise SeedError("public commit tree differs from the verified archive")
    fsck = _run_git(
        git, environment, "-C", str(repository), "fsck", "--full", "--strict",
        "--no-reflogs", "--unreachable", "--no-progress",
    )
    if fsck.strip():
        raise SeedError("public repository fsck reported unreachable or dangling objects")
    assert tree_oid is not None
    return parent_line, tree_oid


def seed_public_repository(release_dir: Path) -> dict[str, object]:
    raw_release = release_dir if release_dir.is_absolute() else Path.cwd() / release_dir
    if raw_release.is_symlink():
        raise SeedError("release directory must not be a symlink")
    release_dir = raw_release.resolve()
    if not release_dir.is_dir():
        raise SeedError(f"release directory is unavailable: {release_dir}")
    repository = release_dir / PUBLIC_REPOSITORY_DIR
    repository_manifest_path = release_dir / PUBLIC_REPOSITORY_MANIFEST_NAME
    if repository.exists() or repository_manifest_path.exists():
        raise SeedError("public repository output already exists")

    source_manifest, values = _validate_archive(release_dir)
    expected = _expected_inventory(values)
    staged = Path(tempfile.mkdtemp(prefix=".repository-stage-", dir=release_dir))
    scratch = Path(tempfile.mkdtemp(prefix="tlm-public-git-"))
    temporary_manifest = release_dir / f".{PUBLIC_REPOSITORY_MANIFEST_NAME}.tmp"
    published_repository = False
    try:
        _extract_values(staged, values)
        git, environment = _clean_git_environment(scratch)
        empty_template = scratch / "empty-template"
        empty_template.mkdir()
        _run_git(
            git, environment, "init", "--quiet", "--initial-branch=main",
            "--object-format=sha1", f"--template={empty_template}", str(staged),
        )
        _run_git(
            git, environment, "-C", str(staged), "-c", "core.autocrlf=false",
            "-c", "core.filemode=true", "add", "-f", "--all",
        )
        _, index_inventory = _git_inventory(staged, git, environment, tree=False)
        if index_inventory != expected:
            raise SeedError("public Git index differs from the verified archive")
        tree_oid = _run_git(git, environment, "-C", str(staged), "write-tree").decode().strip()
        commit_environment = {
            **environment,
            "GIT_AUTHOR_NAME": PUBLIC_AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": PUBLIC_AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": PUBLIC_AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": PUBLIC_AUTHOR_EMAIL,
            "GIT_AUTHOR_DATE": PUBLIC_COMMIT_DATE,
            "GIT_COMMITTER_DATE": PUBLIC_COMMIT_DATE,
        }
        root_commit = _run_git(
            git, commit_environment, "-C", str(staged), "-c", "commit.gpgSign=false",
            "commit-tree", tree_oid, input_value=PUBLIC_COMMIT_MESSAGE.encode(),
        ).decode().strip()
        _run_git(git, environment, "-C", str(staged), "update-ref", "refs/heads/main", root_commit)
        _run_git(git, environment, "-C", str(staged), "symbolic-ref", "HEAD", "refs/heads/main")
        verified_commit, verified_tree = _verify_repository(staged, git, environment, expected)
        if verified_commit != root_commit or verified_tree != tree_oid:
            raise SeedError("public root commit verification changed identity")

        archive_row = source_manifest["archive"]
        assert isinstance(archive_row, dict)
        repository_manifest = {
            "schema_version": 1,
            "project": "TLM",
            "source_archive_sha256": archive_row["sha256"],
            "source_tree_sha256": source_manifest["source_tree_sha256"],
            "repository_directory": PUBLIC_REPOSITORY_DIR,
            "root_commit": root_commit,
            "root_tree": tree_oid,
            "branch": "main",
            "commit_count": 1,
            "history_is_parentless": True,
            "author_name": PUBLIC_AUTHOR_NAME,
            "author_email": PUBLIC_AUTHOR_EMAIL,
            "commit_date": PUBLIC_COMMIT_DATE,
            "remote_count": 0,
            "tag_count": 0,
            "archive_tree_match": True,
            "independent_policy_rescan": "zero_findings",
            "git_fsck": "passed_without_unreachable_objects",
        }
        temporary_manifest.write_bytes(_json_bytes(repository_manifest))
        staged.rename(repository)
        published_repository = True
        temporary_manifest.replace(repository_manifest_path)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        if published_repository and repository.exists():
            shutil.rmtree(repository)
        temporary_manifest.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return {
        "status": "passed", "repository": str(repository),
        "manifest": str(repository_manifest_path), "root_commit": root_commit,
        "root_tree": tree_oid, "commit_count": 1, "branch": "main",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=Path("dist/public"))
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = seed_public_repository(arguments.release_dir)
    except (SeedError, FileNotFoundError, tarfile.TarError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
