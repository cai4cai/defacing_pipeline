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
    assert seen["dilation_backend"] == "separable_cpu"
    assert DefacePipeline("brainsfit", "brainsresample").dilation_backend == "scipy"


@pytest.mark.parametrize("backend", ["separable_cpu", "cuda"])
def test_random_shapes_and_numeric_inputs(backend):
    from caideface.skull_strip import get_safe_structuring_element
    if backend == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA device unavailable")
    rng = np.random.default_rng(42)
    for _ in range(30):
        shape = (1, int(rng.integers(2, 12)), int(rng.integers(2, 12)))
        volume = np.asfortranarray(rng.integers(-2, 2, size=shape).astype(float))[:, ::-1, :]
        volume[0, 0, 0] = np.nan
        structure = get_safe_structuring_element(tuple(rng.uniform(1, 4, size=3)))
        np.testing.assert_array_equal(
            rectangular_dilation(volume, structure, backend),
            binary_dilation(volume, structure=structure),
        )


@pytest.mark.parametrize("entry", ["pipeline", "batch", "single"])
@pytest.mark.parametrize("failure", ["unavailable", "kernel"])
def test_cuda_preflight_before_extraction(tmp_path, monkeypatch, entry, failure):
    import caideface.skull_strip as ss
    from caideface.pipeline import DefacePipeline
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: failure == "kernel")
    if failure == "kernel":
        def broken_kernel(*args, **kwargs):
            raise RuntimeError("CUDA kernel unavailable")
        monkeypatch.setattr(torch, "ones", broken_kernel)
    def unexpected(*args, **kwargs):
        pytest.fail("Extraction must not be reached")
    monkeypatch.setattr(ss, "_hd_bet_executable", unexpected)
    monkeypatch.setattr(ss, "_extract_brain_hdbet", unexpected)
    with pytest.raises(RuntimeError, match="CUDA"):
        if entry == "pipeline":
            DefacePipeline("fit", "resample", dilation_backend="cuda")
        elif entry == "batch":
            ss.skull_strip_batch(str(tmp_path), str(tmp_path / "out"), dilation_backend="cuda")
        else:
            ss.skull_strip_single(str(tmp_path / "scan.nii.gz"), str(tmp_path),
                                  str(tmp_path / "out"), dilation_backend="cuda")
    assert not (tmp_path / "out").exists()


def test_batch_saves_exact_separable_mask(tmp_path, monkeypatch):
    import nibabel as nib
    import caideface.skull_strip as ss
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    data = np.zeros((8, 9, 10), dtype=np.float32)
    data[2:5, 3:6, 4:7] = 10
    affine = np.diag([1.5, 2, 3, 1])
    scan = inputs / "scan.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), scan)
    monkeypatch.setattr(ss, "_hd_bet_executable", lambda: "stub")
    def extract(source, output, *args):
        nib.save(nib.load(source), output)
    monkeypatch.setattr(ss, "_extract_brain_hdbet", extract)
    actual_dilation = ss.rectangular_dilation
    seen = []
    def dilation(volume, structure, backend):
        seen.append(backend)
        return actual_dilation(volume, structure, backend)
    monkeypatch.setattr(ss, "rectangular_dilation", dilation)
    out = tmp_path / "out"
    result = ss.skull_strip_batch(str(inputs), str(out), device="cpu",
                                dilation_backend="separable_cpu")
    assert seen == ["separable_cpu"]
    assert result.iloc[0]["mask"] == "success"
    masks = list(out.glob("*mask.nii.gz"))
    assert len(masks) == 1
    expected = binary_dilation(data > 0, structure=ss.get_safe_structuring_element((1.5, 2, 3)))
    np.testing.assert_array_equal(nib.load(masks[0]).get_fdata() > 0, expected)
