# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

This file starts from the 2026-09 OSS maturity pass. Earlier history (v1.x-3.1.0)
was not retroactively reconstructed into this format — see `git log` and
[ROADMAP_HONEST.md](./ROADMAP_HONEST.md) for what changed and when up to that point.

## [Unreleased]

### Fixed

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
