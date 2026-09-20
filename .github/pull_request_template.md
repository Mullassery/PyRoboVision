## What does this PR do?

<!-- One or two sentences. Link any related issue. -->

## Checklist

- [ ] `pytest tests/ -v` passes locally (with `[dev]` extras installed)
- [ ] New behavior has tests — no exceptions for `tracking/`/`prediction/` (the core)
- [ ] `black src/ tests/`, `isort src/ tests/`, `ruff check src/pyrobovision`, and
      `mypy src/pyrobovision` were run (note: this repo currently has pre-existing
      formatting/lint/type debt not yet cleaned up — see
      [ROADMAP_HONEST.md](../ROADMAP_HONEST.md#technical-debt) — so don't feel
      obligated to fix unrelated pre-existing issues in your diff, but don't add new ones)
- [ ] Any claim in a docstring/README/ROADMAP about what this code does has been
      checked against what it actually does (this project has previously shipped
      inaccurate feature claims — see `ROADMAP_HONEST.md` — and is trying hard not
      to repeat that)

## How was this tested?

<!-- Commands you ran and their real output, not just "tests pass". -->
