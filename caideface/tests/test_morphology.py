"""Synthetic parity checks, including even-kernel origins and image boundaries."""

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from caideface.morphology import rectangular_dilation


@pytest.mark.parametrize("backend", ["scipy", "separable_cpu", "cuda"])
@pytest.mark.parametrize("kernel", [(1, 1, 1), (2, 4, 6), (3, 5, 7), (14, 14, 14), (30, 2, 3)])
@pytest.mark.parametrize("pattern", ["empty", "full", "corners", "random"])
def test_rectangular_parity(backend, kernel, pattern):
    if backend == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA device unavailable")
    volume = np.zeros((9, 8, 7), dtype=bool)
    if pattern == "full":
        volume[:] = True
    elif pattern == "corners":
        volume[0, 0, 0] = volume[-1, -1, -1] = True
    elif pattern == "random":
        volume = np.random.default_rng(17).random(volume.shape) < 0.04
    structure = np.ones(kernel, dtype=bool)
    actual = rectangular_dilation(volume, structure, backend)
    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, binary_dilation(volume, structure=structure))


def test_noncontiguous_input():
    volume = np.arange(60).reshape(3, 4, 5)[::-1, :, ::-1] % 11 == 0
    structure = np.ones((2, 3, 4), dtype=bool)
    np.testing.assert_array_equal(
        rectangular_dilation(volume, structure, "separable_cpu"),
        binary_dilation(volume, structure=structure),
    )


@pytest.mark.parametrize("backend", ["separable_cpu", "cuda"])
def test_reject_nonrectangular_structure(backend):
    structure = np.ones((3, 3, 3), dtype=bool)
    structure[0, 0, 0] = False
    with pytest.raises(ValueError, match="solid rectangular"):
        rectangular_dilation(np.ones((4, 4, 4)), structure, backend)


def test_cuda_never_silently_falls_back(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        rectangular_dilation(np.ones((3, 3, 3)), np.ones((2, 2, 2)), "cuda")


def test_unknown_backend():
    with pytest.raises(ValueError, match="Unknown dilation"):
        rectangular_dilation(np.ones((3, 3, 3)), np.ones((2, 2, 2)), "typo")


def test_pipeline_forwards_backend(tmp_path, monkeypatch):
    from caideface.pipeline import DefacePipeline
    import caideface.pipeline as pipeline
    seen = {}

    def fake_strip(**kwargs):
        seen.update(kwargs)
        return "done"

    monkeypatch.setattr(pipeline, "skull_strip_batch", fake_strip)
    p = DefacePipeline("brainsfit", "brainsresample", dilation_backend="separable_cpu")
    result = p.run(str(tmp_path), str(tmp_path / "out"), steps="skull_strip")
    assert result["skull_strip_log"] == "done"
    assert set(result["stage_seconds"]) == {"skull_strip"}
    assert result["stage_seconds"]["skull_strip"] >= 0
    assert seen["dilation_backend"] == "separable_cpu"
    assert DefacePipeline("brainsfit", "brainsresample").dilation_backend == "scipy"
