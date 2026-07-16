#!/usr/bin/env python3
"""Build a deterministic, history-free TLM source snapshot for publication."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
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
POLICY_VERSION = "tlm-public-source-v2"

PUBLIC_ROOT_FILES = frozenset(
    {
        ".gitattributes",
        ".gitignore",
        "AGENTS.md",
        "AUTONOMOUS_TRAINING_LOOP.md",
        "CITATION.cff",
        "COPYING",
        "LICENCE",
        "LICENSE",
        "Makefile",
        "README.md",
        "STATUS.md",
        "TASKS.md",
        "docs/PUBLIC_RELEASE.md",
        "pyproject.toml",
        "research/current.yaml",
    }
)
PUBLIC_PREFIXES = (
    ".agents/",
    "configs/",
    "docs/research/",
    "prompts/",
    "research/amendments/",
    "research/authorizations/",
    "research/candidates/",
    "research/experiments/",
    "research/incidents/",
    "research/phase_contracts/",
    "research/receipts/",
    "research/schemas/",
    "research/waivers/",
    "scripts/",
    "src/",
    "tests/",
)
EXCLUDED_PREFIXES = (
    "artifacts/",
    "data/",
    "dist/",
    "docs/social/",
    "docs/visuals/",
    "research/backups/",
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
FORBIDDEN_NAME_PATTERN = re.compile(
    r"(?:^|[._-])(?:credential|credentials|secret|secrets|wallet|wallets)(?:[._-]|$)",
    re.IGNORECASE,
)
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 12 * 1024 * 1024
MAX_FILE_COUNT = 1000

BYTE_PATTERNS = (
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
    (
        "credential_url",
        re.compile(br"[A-Za-z][A-Za-z0-9+.-]*://[^/\s:]+:[^@\s]+@"),
    ),
    (
        "presigned_url",
        re.compile(b"(?i:X-" + b"Amz-(?:Credential|Signature)=|X-" + b"Goog-Signature=)"),
    ),
    (
        "bearer_token",
        re.compile(br"(?i:Authorization\s*:\s*Bearer\s+[A-Za-z0-9._~+/-]{16,})"),
    ),
    (
        "jwt",
        re.compile(br"eyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}"),
    ),
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


class SnapshotError(RuntimeError):
    """Raised when a public snapshot violates the publication policy."""


def _clean_git_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "LC_ALL": "C",
        }
    )
    return environment


def _git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "--no-replace-objects", *arguments],
        cwd=root,
        env=_clean_git_environment(),
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode != 0:
        detail = result.stderr if isinstance(result.stderr, str) else result.stderr.decode()
        raise SnapshotError(f"git {' '.join(arguments)} failed: {detail.strip()}")
    return result.stdout


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _decode_git_path(encoded: bytes) -> str:
    try:
        relative = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SnapshotError("non-UTF-8 Git path") from error
    if unicodedata.normalize("NFC", relative) != relative:
        raise SnapshotError(f"non-NFC Git path: {relative!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in relative):
        raise SnapshotError(f"control character in Git path: {relative!r}")
    if "\\" in relative:
        raise SnapshotError(f"backslash in Git path: {relative!r}")
    return relative


def _tree_entries(root: Path, revision: str) -> list[tuple[str, str, str, str]]:
    raw = _git(root, "ls-tree", "-r", "-z", "--full-tree", revision, binary=True)
    assert isinstance(raw, bytes)
    entries: list[tuple[str, str, str, str]] = []
    collision_keys: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode_value, kind_value, oid_value = metadata.split(b" ", 2)
        relative = _decode_git_path(encoded_path)
        collision_key = relative.casefold()
        prior = collision_keys.get(collision_key)
        if prior is not None and prior != relative:
            raise SnapshotError(f"case-fold path collision: {prior!r}, {relative!r}")
        collision_keys[collision_key] = relative
        entries.append(
            (
                mode_value.decode("ascii"),
                kind_value.decode("ascii"),
                oid_value.decode("ascii"),
                relative,
            )
        )
    return sorted(entries, key=lambda entry: entry[3])


def _read_blob(root: Path, oid: str) -> bytes:
    value = _git(root, "cat-file", "blob", oid, binary=True)
    assert isinstance(value, bytes)
    return value


def _excluded_rule(relative: str) -> str | None:
    for prefix in EXCLUDED_PREFIXES:
        if relative == prefix[:-1] or relative.startswith(prefix):
            return prefix
    return None


def _validate_relative_path(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SnapshotError(f"unsafe repository path: {relative}")
    return path


def _validate_public_path(relative: str) -> None:
    path = _validate_relative_path(relative)
    basename = path.name.casefold()
    suffix = path.suffix.casefold()
    components = {component.casefold() for component in path.parts}
    if basename in FORBIDDEN_BASENAMES or basename.startswith(".env."):
        raise SnapshotError(f"forbidden sensitive filename: {relative}")
    if suffix in FORBIDDEN_SUFFIXES:
        raise SnapshotError(f"forbidden file type: {relative}")
    if components & FORBIDDEN_COMPONENTS or FORBIDDEN_NAME_PATTERN.search(relative):
        raise SnapshotError(f"forbidden sensitive path: {relative}")
    if relative not in PUBLIC_ROOT_FILES and not relative.startswith(PUBLIC_PREFIXES):
        raise SnapshotError(f"path is outside the public allowlist: {relative}")
    if relative not in PUBLIC_ROOT_FILES and suffix not in ALLOWED_TEXT_SUFFIXES:
        raise SnapshotError(f"unapproved public source type: {relative}")


def _scan_text_bytes(relative: str, value: bytes) -> None:
    if len(value) > MAX_FILE_BYTES:
        raise SnapshotError(f"file exceeds {MAX_FILE_BYTES} bytes: {relative}")
    if b"\0" in value:
        raise SnapshotError(f"binary NUL in public source: {relative}")
    try:
        value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SnapshotError(f"non-UTF-8 public source: {relative}") from error
    if value.startswith(LFS_POINTER_PREFIX):
        raise SnapshotError(f"Git LFS pointer is not allowed: {relative}")
    for rule, pattern in BYTE_PATTERNS:
        if pattern.search(value):
            raise SnapshotError(f"sensitive pattern {rule}: {relative}")
    for match in EMAIL_PATTERN.finditer(value):
        email = match.group(0).lower()
        if email in ALLOWED_EMAILS or email.endswith(ALLOWED_FIXTURE_EMAIL_DOMAINS):
            continue
        raise SnapshotError(f"non-public email address: {relative}")


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


def _scanner_policy() -> dict[str, object]:
    return {
        "engine": "tlm-built-in-static-scan-v2",
        "byte_patterns": [
            {"name": name, "pattern": pattern.pattern.decode("latin-1")}
            for name, pattern in BYTE_PATTERNS
        ],
        "email_rule": EMAIL_PATTERN.pattern.decode("latin-1"),
        "allowed_emails": sorted(value.decode("ascii") for value in ALLOWED_EMAILS),
        "allowed_fixture_email_domains": [value.decode("ascii") for value in ALLOWED_FIXTURE_EMAIL_DOMAINS],
        "reject_git_lfs_pointer": True,
    }


def _publication_policy() -> dict[str, object]:
    return {
        "version": POLICY_VERSION,
        "public_root_files": sorted(PUBLIC_ROOT_FILES),
        "public_prefixes": list(PUBLIC_PREFIXES),
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "allowed_text_suffixes": sorted(ALLOWED_TEXT_SUFFIXES),
        "forbidden_suffixes": sorted(FORBIDDEN_SUFFIXES),
        "forbidden_basenames": sorted(FORBIDDEN_BASENAMES),
        "forbidden_components": sorted(FORBIDDEN_COMPONENTS),
        "maximum_file_bytes": MAX_FILE_BYTES,
        "maximum_total_bytes": MAX_TOTAL_BYTES,
        "maximum_file_count": MAX_FILE_COUNT,
    }


def _write_deterministic_archive(archive_path: Path, entries: list[tuple[str, bytes, int]]) -> None:
    with archive_path.open("xb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, compresslevel=9, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
                for relative, value, mode in sorted(entries):
                    info = tarfile.TarInfo(relative)
                    info.size = len(value)
                    info.mode = mode
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(value))


def _verify_written_archive(archive_path: Path, entries: list[tuple[str, bytes, int]]) -> None:
    expected = {relative: (value, mode) for relative, value, mode in entries}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != set(expected):
            raise SnapshotError("written archive member set does not match source inventory")
        for member in members:
            value, mode = expected[member.name]
            extracted = archive.extractfile(member)
            if not member.isfile() or member.mode != mode or extracted is None or extracted.read() != value:
                raise SnapshotError(f"written archive mismatch: {member.name}")


def _replace_release_directory(staged: Path, target: Path) -> None:
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise SnapshotError(f"public output path is not a normal directory: {target}")
    backup = target.parent / f".{target.name}.previous-{os.getpid()}"
    if backup.exists():
        raise SnapshotError(f"stale public output backup requires review: {backup}")
    moved_prior = False
    try:
        if target.exists():
            target.rename(backup)
            moved_prior = True
        staged.rename(target)
    except Exception:
        if moved_prior and not target.exists() and backup.exists():
            backup.rename(target)
        raise
    if moved_prior:
        shutil.rmtree(backup)


def _validate_repository_object_state(root: Path) -> None:
    git_dir = Path(str(_git(root, "rev-parse", "--absolute-git-dir")).strip())
    forbidden = (
        git_dir / "info/grafts",
        git_dir / "objects/info/alternates",
        git_dir / "shallow",
    )
    if any(path.exists() for path in forbidden):
        raise SnapshotError("graft, alternate, or shallow Git state is not allowed")
    replace_refs = str(_git(root, "for-each-ref", "--format=%(refname)", "refs/replace"))
    if replace_refs.strip():
        raise SnapshotError("Git replace refs are not allowed")


def _public_output_path(root: Path, output_dir: Path) -> Path:
    candidate = output_dir if output_dir.is_absolute() else root / output_dir
    candidate = Path(os.path.abspath(candidate))
    expected = root / "dist/public"
    if candidate != expected:
        raise SnapshotError("public output must be exactly <repo>/dist/public")
    if (root / "dist").is_symlink() or expected.is_symlink():
        raise SnapshotError("public output path must not contain symlinks")
    return expected


def build_snapshot(root: Path, output_dir: Path) -> dict[str, object]:
    root = root.resolve()
    if not (root / ".git").exists():
        raise SnapshotError(f"not a Git repository: {root}")
    output_dir = _public_output_path(root, output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _validate_repository_object_state(root)

    revision = str(_git(root, "rev-parse", "HEAD^{commit}")).strip()
    source_tree_oid = str(_git(root, "rev-parse", f"{revision}^{{tree}}")).strip()
    private_identifiers = {revision.encode("ascii"), source_tree_oid.encode("ascii")}

    rows: list[dict[str, object]] = []
    archive_entries: list[tuple[str, bytes, int]] = []
    exported_values: dict[str, bytes] = {}
    excluded_counts = {prefix: 0 for prefix in EXCLUDED_PREFIXES}
    total_bytes = 0
    for mode, kind, oid, relative in _tree_entries(root, revision):
        excluded = _excluded_rule(relative)
        if excluded is not None:
            excluded_counts[excluded] += 1
            continue
        _validate_public_path(relative)
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise SnapshotError(f"non-regular or unsupported Git entry: {relative}")
        value = _read_blob(root, oid)
        _scan_text_bytes(relative, value)
        if any(identifier in value for identifier in private_identifiers):
            raise SnapshotError(f"current source Git identifier exposed in source: {relative}")
        total_bytes += len(value)
        if total_bytes > MAX_TOTAL_BYTES:
            raise SnapshotError(f"public source exceeds {MAX_TOTAL_BYTES} total bytes")
        if len(rows) + 1 > MAX_FILE_COUNT:
            raise SnapshotError(f"public source exceeds {MAX_FILE_COUNT} files")
        rows.append({"mode": mode, "path": relative, "sha256": _sha256_bytes(value), "size": len(value)})
        archive_entries.append((relative, value, 0o755 if mode == "100755" else 0o644))
        exported_values[relative] = value

    builder_value = exported_values.get("scripts/build_public_snapshot.py")
    if builder_value is None or builder_value != Path(__file__).resolve().read_bytes():
        raise SnapshotError("running snapshot builder does not match the immutable exported builder")

    policy = _publication_policy()
    scanner = _scanner_policy()
    source_tree_sha256 = _content_digest(rows)
    license_present = next((name for name in ("LICENSE", "LICENCE", "COPYING") if name in exported_values), None)
    inner_manifest: dict[str, object] = {
        "schema_version": 2,
        "project": "TLM",
        "history_included": False,
        "current_source_head_and_tree_identifiers_included": False,
        "historical_contract_receipts_may_reference_ancestry": True,
        "source_tree_sha256": source_tree_sha256,
        "research_current_sha256": _sha256_bytes(exported_values["research/current.yaml"]),
        "builder_sha256": _sha256_bytes(builder_value),
        "publication_policy": {"version": POLICY_VERSION, "sha256": _sha256_bytes(_canonical_json_bytes(policy))},
        "built_in_scan": {
            "engine": scanner["engine"],
            "ruleset_sha256": _sha256_bytes(_canonical_json_bytes(scanner)),
            "status": "zero_findings",
            "checked_file_count": len(rows),
        },
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "files": rows,
        "exclusions": {"prefix_counts": excluded_counts, "total_tracked_files": sum(excluded_counts.values())},
        "license_status": f"{license_present.lower()}_present" if license_present else "no_license_all_rights_reserved",
    }
    archive_entries.append((INNER_MANIFEST_NAME, _json_bytes(inner_manifest), 0o644))

    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        archive_path = staged / ARCHIVE_NAME
        _write_deterministic_archive(archive_path, archive_entries)
        _verify_written_archive(archive_path, archive_entries)
        outer_manifest = {
            **inner_manifest,
            "archive": {"path": ARCHIVE_NAME, "sha256": _sha256_file(archive_path), "size": archive_path.stat().st_size},
        }
        (staged / MANIFEST_NAME).write_bytes(_json_bytes(outer_manifest))
        _replace_release_directory(staged, output_dir)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise

    return {
        "status": "passed",
        "archive": str(output_dir / ARCHIVE_NAME),
        "manifest": str(output_dir / MANIFEST_NAME),
        "archive_sha256": outer_manifest["archive"]["sha256"],
        "source_tree_sha256": source_tree_sha256,
        "file_count": len(rows),
        "excluded_tracked_file_count": sum(excluded_counts.values()),
        "license_status": inner_manifest["license_status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("dist/public"))
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = build_snapshot(arguments.repo_root, arguments.output_dir)
    except SnapshotError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
