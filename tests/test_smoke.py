"""Smoke tests: confirm the compiled extension imports and the Rust engine is reachable.

Real dataset/loader tests arrive with the v0.1 API.
"""

import pyroboframes as prf


def test_version_is_exposed():
    assert isinstance(prf.__version__, str)
    assert prf.__version__


def test_engine_version_matches_package():
    # The Python package version mirrors the Rust engine version.
    assert prf.engine_version() == prf.__version__
