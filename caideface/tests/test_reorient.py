"""Tests for the reorientation module."""

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from caideface.reorient import reorient_single, reorient_batch


def _make_nifti(shape=(64, 64, 32), orientation="PSR", dirpath=None):
    """Create a synthetic NIfTI file with a given orientation."""
    from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation

    data = np.random.default_rng(42).random(shape, dtype=np.float32)
    # Start with a RAS affine, then reorient to the desired orientation
    ras_affine = np.diag([1.0, 1.0, 1.5, 1.0])
    ras_img = nib.Nifti1Image(data, ras_affine)

    orig_ornt = io_orientation(ras_img.affine)
    target_ornt = axcodes2ornt(tuple(orientation))
    transform = ornt_transform(orig_ornt, target_ornt)
    img = ras_img.as_reoriented(transform)

    path = os.path.join(dirpath, "test_scan.nii.gz")
    nib.save(img, path)
    return path


class TestReorientSingle:
    def test_reorients_to_ras(self, tmp_path):
        input_path = _make_nifti(orientation="PSR", dirpath=str(tmp_path))
        output_path = os.path.join(str(tmp_path), "output", "reoriented.nii.gz")

        success = reorient_single(input_path, output_path)

        assert success is True
        assert os.path.exists(output_path)
        img = nib.load(output_path)
        assert nib.aff2axcodes(img.affine) == ("R", "A", "S")

    def test_already_ras_is_unchanged(self, tmp_path):
        input_path = _make_nifti(orientation="RAS", dirpath=str(tmp_path))
        output_path = os.path.join(str(tmp_path), "output", "reoriented.nii.gz")

        success = reorient_single(input_path, output_path)

        assert success is True
        orig = nib.load(input_path)
        reoriented = nib.load(output_path)
        assert np.allclose(orig.get_fdata(), reoriented.get_fdata())
        assert np.allclose(orig.affine, reoriented.affine)

    def test_preserves_data_in_world_coordinates(self, tmp_path):
        input_path = _make_nifti(orientation="PIR", dirpath=str(tmp_path))
        output_path = os.path.join(str(tmp_path), "output", "reoriented.nii.gz")

        reorient_single(input_path, output_path)

        orig = nib.load(input_path)
        reoriented = nib.load(output_path)

        # Sample voxels and check world-space intensities match
        rng = np.random.default_rng(0)
        orig_data = orig.get_fdata()
        reori_data = reoriented.get_fdata()

        for _ in range(500):
            vox = rng.integers([0, 0, 0], orig_data.shape)
            world = orig.affine @ np.append(vox, 1)
            vox_reori = np.round(np.linalg.inv(reoriented.affine) @ world).astype(int)[:3]
            if all(0 <= vox_reori[i] < reori_data.shape[i] for i in range(3)):
                assert np.isclose(
                    orig_data[tuple(vox)],
                    reori_data[tuple(vox_reori)],
                    atol=1e-5,
                )

    def test_various_orientations(self, tmp_path):
        orientations = ["RAS", "LPS", "AIL", "SRA", "IPL"]
        for ornt in orientations:
            input_path = _make_nifti(orientation=ornt, dirpath=str(tmp_path))
            output_path = os.path.join(str(tmp_path), f"out_{ornt}.nii.gz")
            success = reorient_single(input_path, output_path)
            assert success is True
            img = nib.load(output_path)
            assert nib.aff2axcodes(img.affine) == ("R", "A", "S"), f"Failed for {ornt}"

    def test_invalid_file_returns_false(self, tmp_path):
        fake_path = os.path.join(str(tmp_path), "nonexistent.nii.gz")
        output_path = os.path.join(str(tmp_path), "output.nii.gz")
        assert reorient_single(fake_path, output_path) is False


class TestReorientBatch:
    def test_processes_all_files(self, tmp_path):
        # Create a directory structure with multiple scans
        for sub in ["sub01", "sub02"]:
            subdir = tmp_path / sub
            subdir.mkdir()
            _make_nifti(orientation="PSR", dirpath=str(subdir))

        output_dir = str(tmp_path / "output")
        df = reorient_batch(str(tmp_path), output_dir)

        assert len(df) == 2
        assert df["success"].all()
        assert os.path.exists(os.path.join(output_dir, "reorientation_log.csv"))

    def test_skips_non_nifti_files(self, tmp_path):
        # Create a non-NIfTI file
        (tmp_path / "notes.txt").write_text("not a scan")
        _make_nifti(orientation="PSR", dirpath=str(tmp_path))

        output_dir = str(tmp_path / "output")
        df = reorient_batch(str(tmp_path), output_dir)

        assert len(df) == 1  # Only the .nii.gz file
