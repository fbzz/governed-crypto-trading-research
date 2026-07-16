# Public release procedure

The canonical TLM repository is a private scientific record. Do **not** push
its `master` history directly to a public remote: historical commits contain
generated checkpoints and prediction/return tables, and commit metadata
contains a personal email address.

The supported public release is a separately initialized, history-free Git
repository built from an immutable source snapshot.

## Build the release

First commit the source and documentation intended for publication. Uncommitted
work is deliberately ignored because the builder reads blobs from immutable
`HEAD`, not from the index or worktree. Then run:

```bash
make public-release
```

This produces four ignored local outputs:

- `dist/public/tlm-public-source.tar.gz`
- `dist/public/tlm-public-source.manifest.json`
- `dist/public/repository/`
- `dist/public/tlm-public-repository.manifest.json`

The snapshot has an explicit text-only allowlist. It includes the reviewed
root documentation, Python source and tests, frozen configs and research
contracts, and repository governance skills. It skips tracked paths under
`artifacts/`, `data/`, `dist/`, `docs/social/`, `docs/visuals/`, and
`research/backups/` without opening their blobs. Any unexpected root, binary,
checkpoint, table, database, key, archive, symlink, Git submodule, Git LFS
pointer, oversized file, sensitive path component, private-home path,
non-public email, or recognized credential pattern fails closed. Failure
messages contain the rule and path, never the matched value.

Archive metadata, file order, modes, ownership, and timestamps are normalized.
Two builds from the same immutable tree are byte-identical. The public manifest
records per-file paths, modes, sizes and SHA-256 hashes, a mode-aware source-tree
digest, policy/scanner hashes, exclusion counts, archive digest, and license
status. It does **not** expose the canonical commit or tree identifier.

The built-in scanner is a defense-in-depth release gate, not a substitute for
repository review. The release also relies on the explicit allowlist and the
separate sensitivity audit described in the project handoff.

## What the seeder guarantees

The seeder verifies every tar member before extracting it, compares the Git
index and committed tree byte-for-byte with the source manifest, and creates a
new repository with:

- exactly one parentless commit on `main`;
- fixed `TLM Research <tlm-research@users.noreply.github.com>` author and
  committer metadata;
- no inherited templates, hooks, signing config, credential helpers, remotes,
  tags, alternates, grafts, replace refs, or unreachable objects;
- a clean worktree and a passing strict `git fsck`.

No software license is currently granted. The public payload is therefore
source-visible but **all rights reserved** until the owner deliberately adds a
`LICENSE` file.

## Verify the generated repository

```bash
git -C dist/public/repository rev-list --count --all
git -C dist/public/repository rev-list --parents --max-count=1 HEAD
git -C dist/public/repository for-each-ref --format='%(refname)'
git -C dist/public/repository fsck --full --strict --no-reflogs --unreachable

python3 -m pip wheel \
  --no-deps --no-build-isolation \
  dist/public/repository \
  --wheel-dir /tmp/tlm-wheel

python3 -m pytest -q \
  dist/public/repository/tests/test_model.py \
  dist/public/repository/tests/test_backtest.py \
  dist/public/repository/tests/test_leakage.py \
  dist/public/repository/tests/test_low_turnover_rank_harness.py \
  dist/public/repository/tests/test_public_snapshot.py
```

The first three Git commands must report one commit, a single-token parent
line, and only `refs/heads/main`. `git fsck` must exit successfully without
printing dangling or unreachable objects.

## Push only the sanitized repository

Do not add a public remote to the canonical repository. Push the generated
repository with an exact refspec and without saving the remote:

```bash
git -C dist/public/repository push --no-follow-tags \
  "$PUBLIC_URL" refs/heads/main:refs/heads/main
```

Never use `--all`, `--tags`, or `--mirror`. A public clone is intentionally not
the canonical artifact store, so full packet-verification and training commands
still require the private hash-bound local inputs.
