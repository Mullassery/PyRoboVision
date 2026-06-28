"""P6 — Train-Anywhere backend parity: unified output abstraction, fallback chain, conformance."""

import numpy as np
import pytest

import pyroboframes as prf
from pyroboframes import transforms as T
from test_loader import make_dataset


def test_default_framework_mapping():
    assert prf.default_framework("cpu") == "numpy"
    assert prf.default_framework("cuda") == "torch"
    assert prf.default_framework("mps") == "torch"
    assert prf.default_framework("mlx") == "mlx"


def test_to_backend_cpu_is_identity():
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    out = prf.to_backend({"x": x}, device="cpu")
    assert out["x"] is x  # NumPy passes through untouched


def test_to_backend_mlx_roundtrip():
    mx = pytest.importorskip("mlx.core")
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    out = prf.to_backend(x, device="mlx")
    assert isinstance(out, mx.array)
    np.testing.assert_array_equal(np.array(out), x)


def test_transform_backend_fallback_chain():
    # NumPy is always available, so resolution always succeeds.
    assert prf.transforms.resolve_transform_backend("numpy") == "numpy"
    # Preferring an unavailable rung degrades down the chain (cvcuda absent here -> ... -> numpy).
    assert prf.transforms.resolve_transform_backend("cvcuda") in prf.transforms.TRANSFORM_BACKENDS
    assert prf.transforms.resolve_transform_backend("auto") in prf.transforms.TRANSFORM_BACKENDS
    with pytest.raises(ValueError):
        prf.transforms.resolve_transform_backend("nope")


def test_same_script_conformance_cpu_vs_auto(tmp_path):
    # The *same* loop must yield identical batch shapes on cpu and on this machine's auto device.
    make_dataset(str(tmp_path), episodes=2, length=10)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    def shapes(device):
        loader = prf.DataLoader(ds.loader(batch_size=5, shuffle=False), device=device)
        out = []
        for batch in loader:
            out.append({k: tuple(np.asarray(_to_numpy(v)).shape) for k, v in batch.items()})
        return out

    assert shapes("cpu") == shapes("auto")


def _to_numpy(v):
    """Bring an array from any backend back to NumPy for comparison."""
    if hasattr(v, "__array__"):
        return np.asarray(v)
    try:  # torch tensors
        return v.cpu().numpy()
    except AttributeError:
        return np.asarray(v)
