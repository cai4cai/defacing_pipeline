"""Recovery and isolation of failures while processing synthetic scans."""
import importlib.util
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.ndimage import binary_dilation

import caideface.skull_strip as ss
import caideface.pipeline as pipeline


def test_pipeline_non_imaging_steps_do_not_check_device(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Must not probe a device for registration or reorientation")
    monkeypatch.setattr(ss, "validate_device", forbidden)
    monkeypatch.setattr(ss, "get_default_device", forbidden)
    monkeypatch.setattr(pipeline, "deface_batch", lambda **kwargs: [])
    monkeypatch.setattr(pipeline, "reorient_batch", lambda *args, **kwargs: "done")
    p = pipeline.DefacePipeline("fit", "resample", device="cuda")
    p.run(str(tmp_path), str(tmp_path / "out"), steps="deface,reorient")


def test_automatic_cuda_failure_uses_cpu(monkeypatch, caplog):
    seen = []
    monkeypatch.setattr(ss, "get_default_device", lambda: "cuda")
    def validate(device):
        seen.append(device)
        if device == "cuda":
            raise RuntimeError("no kernel image")
    monkeypatch.setattr(ss, "validate_device", validate)
    assert ss.resolve_device(None) == "cpu"
    assert seen == ["cuda", "cpu"]
    assert "using CPU" in caplog.text
    with pytest.raises(RuntimeError, match="no kernel image"):
        ss.resolve_device("cuda")


@pytest.mark.parametrize("at_transfer", [False, True])
def test_dilation_oom_retries_cpu(monkeypatch, caplog, at_transfer):
    if not at_transfer and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    mask = np.zeros((8, 9, 10), dtype=np.uint8)
    mask[3, 4, 5] = 1
    structure = np.ones((2, 3, 4), dtype=np.uint8)
    original_dilation = ss.rectangular_dilation
    original_as_tensor = torch.as_tensor
    def dilate(tensor, structure):
        if tensor.device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("simulated pooling OOM")
        return original_dilation(tensor, structure)
    def transfer(data, **kwargs):
        if kwargs.get("device") == "cuda":
            raise torch.cuda.OutOfMemoryError("simulated transfer OOM")
        return original_as_tensor(data, **kwargs)
    if at_transfer:
        monkeypatch.setattr(torch, "as_tensor", transfer)
    else:
        monkeypatch.setattr(ss, "rectangular_dilation", dilate)
    np.testing.assert_array_equal(ss._dilate_mask(mask, structure, "cuda"),
                                  binary_dilation(mask, structure=structure))
    assert "retrying this mask on CPU" in caplog.text


def test_non_oom_errors_are_not_retried(monkeypatch):
    seen = []
    def broken(tensor, structure):
        seen.append(tensor.device.type)
        raise RuntimeError("unexpected kernel failure")
    monkeypatch.setattr(ss, "rectangular_dilation", broken)
    with pytest.raises(RuntimeError, match="unexpected kernel"):
        ss._dilate_mask(np.ones((3, 3, 3)), np.ones((2, 2, 2)), "cpu")
    assert seen == ["cpu"]


def test_batch_preserves_status_after_middle_file_failure(tmp_path, monkeypatch):
    inputs = tmp_path / "in"
    inputs.mkdir()
    out = tmp_path / "out"
    for i in range(3):
        nib.save(nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), np.eye(4)),
                 inputs / f"scan{i}.nii.gz")
    probes = []
    monkeypatch.setattr(ss, "validate_device", lambda device: probes.append(device))
    monkeypatch.setattr(ss, "_hd_bet_executable", lambda: "stub")
    monkeypatch.setattr(ss, "_extract_brain_hdbet",
                        lambda source, output, *args: nib.save(nib.load(source), output))
    original = ss._dilate_mask
    calls = []
    def dilate(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("transient device failure")
        return original(*args)
    monkeypatch.setattr(ss, "_dilate_mask", dilate)
    result = ss.skull_strip_batch(str(inputs), str(out), device="cpu")
    assert probes == ["cpu"]
    assert result["mask"].tolist() == ["success", "failed", "success"]
    assert pd.read_csv(out / "skull_strip_log.csv")["mask"].tolist() == result["mask"].tolist()


@pytest.mark.parametrize("module_name", ["morphology", "skull_strip"])
def test_imaging_module_import_does_not_load_torch(monkeypatch, module_name):
    import importlib
    module_under_test = importlib.import_module("caideface." + module_name)
    monkeypatch.setitem(sys.modules, "torch", None)
    spec = importlib.util.spec_from_file_location("caideface._isolated_" + module_name, module_under_test.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.rectangular_dilation)


def test_transient_probe_error_does_not_change_extraction_device(monkeypatch):
    monkeypatch.setattr(ss, "get_default_device", lambda: "cuda")
    def busy(device):
        raise RuntimeError("CUDA device busy or unavailable")
    monkeypatch.setattr(ss, "validate_device", busy)
    with pytest.raises(RuntimeError, match="busy"):
        ss.resolve_device(None)


def test_completed_resume_does_not_probe_or_require_extraction(tmp_path, monkeypatch):
    inputs = tmp_path / "in"
    outputs = tmp_path / "out"
    inputs.mkdir()
    outputs.mkdir()
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5)), np.eye(4)), inputs / "scan.nii.gz")
    for suffix in ("_brain.nii.gz", "_mask.nii.gz", "_dilated.nii.gz"):
        (outputs / ("scan" + suffix)).touch()
    def forbidden(*args):
        pytest.fail("Completed resume must not probe CUDA or extraction tools")
    monkeypatch.setattr(ss, "resolve_device", forbidden)
    monkeypatch.setattr(ss, "_hd_bet_executable", forbidden)
    result = ss.skull_strip_batch(str(inputs), str(outputs), device="cuda")
    assert result.iloc[0]["mask"] == "skipped"
