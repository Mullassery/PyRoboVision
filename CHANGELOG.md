# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

This file starts from the 2026-09 OSS maturity pass. Earlier history (v1.x-3.1.0)
was not retroactively reconstructed into this format — see `git log` and
[ROADMAP_HONEST.md](./ROADMAP_HONEST.md) for what changed and when up to that point.

## [Unreleased]

### Security

- `pyproject.toml`: bumped dev-only dependency ceilings —
  `black>=23.0,<26.0` -> `<27.0` and `pytest>=7.4,<9.0` -> `<10.0` — to allow
  installing versions with known-CVE fixes. Resolves `black` advisories
  `PYSEC-2026-2121`/`PYSEC-2026-2120` (fixed in `26.3.0`/`26.3.1`; installs
  `26.5.1`) and `pytest` advisory `PYSEC-2026-1845` (fixed in `9.0.3`;
  installs `9.1.1`). Verified: `pip-audit` reports zero known vulnerabilities
  after the bump (previously reported all 3); full test suite still
  278 passed / 7 skipped; `black --check src/ tests/` still flags the same
  32/48 files as before the bump (no new formatting drift introduced by the
  newer black). Both are dev-only tooling, not shipped to downstream users.

### Fixed

- `src/pyrobovision/perception/bbox_3d.py:147` and
  `src/pyrobovision/perception/lidar.py:202`: narrowed bare `except:` clauses
  (ruff `E722`) around `np.linalg.eig()` calls to
  `except np.linalg.LinAlgError:`. These previously swallowed every
  exception, including `KeyboardInterrupt`/`SystemExit`; the only failure
  mode `np.linalg.eig` actually raises is `LinAlgError` on non-convergence,
  so the fallback-to-identity/default behavior is preserved for that case
  while everything else now propagates. Verified: full test suite still
  278 passed / 7 skipped; `ruff check src/pyrobovision --select E722` now
  passes with zero findings (previously 2).
- `pyproject.toml`: removed an invalid `[tool.isort]` setting
  (`multi_line_mode = 3`) that caused `isort` to hard-crash with
  `UnsupportedSettings` instead of running.
- `Makefile`: `test-cov` and `lint` targets referenced the package name
  (`pyrobovision`) instead of the `src/` path, the same silent
  "no data collected" coverage bug already worked around in
  `pyproject.toml`'s pytest config but never mirrored here.
- `.github/workflows/tests.yml`: bumped `actions/setup-python` from `v4`
  (deprecated runtime, flagged by `actionlint`) to `v5`.
- `CONTRIBUTING.md` and `CLAUDE.md`: both still referenced the old
  Proprietary license after the project relicensed to Apache License 2.0;
  corrected.
- `CLAUDE.md` and `README.md`: stale test-count/coverage figures
  (277 tests / ~89%, or 276 passing / 87%) replaced with the actual
  re-measured numbers (278 passed / 7 skipped, 84% coverage).

### Added

- `.github/workflows/tests.yml`: `dependency-audit` job running `pip-audit`
  (informational, `continue-on-error: true` — see
  [ROADMAP_HONEST.md](./ROADMAP_HONEST.md#technical-debt) for why it isn't a
  hard gate yet).
- `.github/ISSUE_TEMPLATE/bug_report.yml`, `.github/ISSUE_TEMPLATE/feature_request.yml`,
  `.github/pull_request_template.md`.
- `ROADMAP_HONEST.md` "Technical Debt" section: concrete, file/line-level
  findings from `black`, `isort`, `ruff`, `mypy`, and `pip-audit`.
- This `CHANGELOG.md`.

### Removed

- `.github/CI_ERRORS.md`: generic boilerplate linking to
  `CI_ERRORS_REPORT.md`/`TROUBLESHOOTING.md`, neither of which exists in this
  repository.

### Renamed

- `ROADMAP.md` -> `ROADMAP_HONEST.md` (content unchanged aside from the new
  Technical Debt section noted above).
