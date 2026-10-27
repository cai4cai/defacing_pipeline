"""Synthetic parity checks, including even-kernel origins and image boundaries."""

import numpy as np
import pytest
import torch
from scipy.ndimage import binary_dilation

from caideface.morphology import rectangular_dilation, validate_device


def dilate_numpy(volume, structure, device="cpu"):
    tensor = torch.as_tensor(np.ascontiguousarray(volume), device=device)
    result = rectangular_dilation(tensor, structure)
    assert result.device == tensor.device
    assert result.dtype == torch.bool
    return result.cpu().numpy()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kernel", [(1, 1, 1), (2, 4, 6), (3, 5, 7), (14, 14, 14), (30, 2, 3)])
@pytest.mark.parametrize("pattern", ["empty", "full", "corners", "random"])
def test_rectangular_parity(device, kernel, pattern):
    if device == "cuda":
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
    actual = dilate_numpy(volume, structure, device)
    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, binary_dilation(volume, structure=structure))


def test_noncontiguous_input():
    volume = np.arange(60).reshape(3, 4, 5)[::-1, :, ::-1] % 11 == 0
    structure = np.ones((2, 3, 4), dtype=bool)
    np.testing.assert_array_equal(
        dilate_numpy(volume, structure, "cpu"),
        binary_dilation(volume, structure=structure),
    )


def test_reject_nonrectangular_structure():
    structure = np.ones((3, 3, 3), dtype=bool)
    structure[0, 0, 0] = False
    with pytest.raises(ValueError, match="solid rectangular"):
        rectangular_dilation(torch.ones((4, 4, 4)), structure)


@pytest.mark.parametrize("device", ["cpu", "cuda", None])
def test_pipeline_forwards_device(tmp_path, monkeypatch, device):
    from caideface.pipeline import DefacePipeline
    import caideface.pipeline as pipeline
    seen = {}

    def fake_strip(**kwargs):
        seen.update(kwargs)
        return "done"

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    monkeypatch.setattr(pipeline, "get_default_device", lambda: "cpu")
    monkeypatch.setattr(pipeline, "skull_strip_batch", fake_strip)
    p = DefacePipeline("brainsfit", "brainsresample", device=device)
    result = p.run(str(tmp_path), str(tmp_path / "out"), steps="skull_strip")
    assert result["skull_strip_log"] == "done"
    assert seen["device"] == (device or "cpu")
    assert DefacePipeline("brainsfit", "brainsresample", device="cpu").device == "cpu"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_random_shapes_and_numeric_inputs(device):
    from caideface.skull_strip import get_safe_structuring_element
    if device == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA device unavailable")
    rng = np.random.default_rng(42)
    for index in range(30):
        shape = rng.integers(1, 12, size=3)
        if index < 3:
            shape[index] = 1  # Guarantee singleton coverage on each axis.
        volume = np.asfortranarray(rng.integers(-2, 2, size=shape).astype(float))[:, ::-1, :]
        volume[0, 0, 0] = np.nan
        structure = get_safe_structuring_element(tuple(rng.uniform(1, 4, size=3)))
        np.testing.assert_array_equal(
            dilate_numpy(volume, structure, device),
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
            DefacePipeline("fit", "resample", device="cuda")
        elif entry == "batch":
            ss.skull_strip_batch(str(tmp_path), str(tmp_path / "out"), device="cuda")
        else:
            ss.skull_strip_single(str(tmp_path / "scan.nii.gz"), str(tmp_path),
                                  str(tmp_path / "out"), device="cuda")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("selected_device", ["cpu", "cuda", None])
def test_batch_saves_exact_separable_mask(tmp_path, monkeypatch, selected_device):
    import nibabel as nib
    import caideface.skull_strip as ss
    expected_device = selected_device or "cuda"
    if expected_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    monkeypatch.setattr(ss, "get_default_device", lambda: "cuda")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    data = np.zeros((8, 9, 10), dtype=np.float32)
    data[2:5, 3:6, 4:7] = 10
    affine = np.diag([1.5, 2, 3, 1])
    scan = inputs / "scan.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), scan)
    monkeypatch.setattr(ss, "_hd_bet_executable", lambda: "stub")
    def extract(source, output, device, *args):
        assert device == expected_device
        nib.save(nib.load(source), output)
    monkeypatch.setattr(ss, "_extract_brain_hdbet", extract)
    actual_dilation = ss.rectangular_dilation
    seen = []
    def dilation(volume, structure):
        seen.append(volume.device.type)
        return actual_dilation(volume, structure)
    monkeypatch.setattr(ss, "rectangular_dilation", dilation)
    out = tmp_path / "out"
    result = ss.skull_strip_batch(str(inputs), str(out), device=selected_device)
    assert seen == [expected_device]
    assert result.iloc[0]["mask"] == "success"
    masks = list(out.glob("*mask.nii.gz"))
    assert len(masks) == 1
    expected = binary_dilation(data > 0, structure=ss.get_safe_structuring_element((1.5, 2, 3)))
    np.testing.assert_array_equal(nib.load(masks[0]).get_fdata() > 0, expected)


def test_default_cpu_without_cuda(monkeypatch):
    import torch
    from caideface.morphology import validate_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    validate_device("cpu")
    volume = np.zeros((3, 4, 5), dtype=np.uint8)
    volume[1, 2, 3] = 1
    structure = np.ones((2, 3, 4), dtype=np.uint8)
    np.testing.assert_array_equal(dilate_numpy(volume, structure),
                                  binary_dilation(volume, structure=structure))
