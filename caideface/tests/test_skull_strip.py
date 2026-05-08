"""Tests for the skull-stripping module (unit tests only, no HD-BET needed)."""

import os

import nibabel as nib
import numpy as np
import pytest

from caideface.skull_strip import get_safe_structuring_element, is_valid_3d_volume


class TestStructuringElement:
    def test_isotropic_voxels(self):
        elem = get_safe_structuring_element((1.0, 1.0, 1.0), desired_dilation_mm=14.0)
        assert elem.shape == (14, 14, 14)
        assert elem.dtype == np.uint8
        assert np.all(elem == 1)

    def test_anisotropic_voxels(self):
        elem = get_safe_structuring_element((1.2, 0.9, 0.9), desired_dilation_mm=14.0)
        # 14/1.2 = 11.67 -> 12, 14/0.9 = 15.56 -> 16
        assert elem.shape == (12, 16, 16)

    def test_large_voxels(self):
        elem = get_safe_structuring_element((3.0, 3.0, 3.0), desired_dilation_mm=14.0)
        # 14/3 = 4.67 -> 5
        assert elem.shape == (5, 5, 5)

    def test_very_small_voxels(self):
        elem = get_safe_structuring_element((0.3, 0.3, 0.3), desired_dilation_mm=14.0)
        # 14/0.3 = 46.67 -> 47, but capped at 30
        assert elem.shape == (30, 30, 30)

    def test_minimum_size_is_one(self):
        elem = get_safe_structuring_element((100.0, 100.0, 100.0), desired_dilation_mm=14.0)
        # 14/100 = 0.14 -> rounds to 0 -> clamped to 1
        assert elem.shape == (1, 1, 1)

    def test_custom_max_kernel(self):
        elem = get_safe_structuring_element(
            (0.5, 0.5, 0.5), desired_dilation_mm=14.0, max_kernel_shape=(10, 10, 10)
        )
        # 14/0.5 = 28, capped at 10
        assert elem.shape == (10, 10, 10)

    def test_custom_dilation_mm(self):
        elem = get_safe_structuring_element((1.0, 1.0, 1.0), desired_dilation_mm=7.0)
        assert elem.shape == (7, 7, 7)


class TestIsValid3dVolume:
    def test_valid_3d(self, tmp_path):
        data = np.zeros((64, 64, 32), dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        path = str(tmp_path / "valid_3d.nii.gz")
        nib.save(img, path)
        assert is_valid_3d_volume(path) is True

    def test_2d_rejected(self, tmp_path):
        data = np.zeros((64, 64, 1), dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        path = str(tmp_path / "flat.nii.gz")
        nib.save(img, path)
        assert is_valid_3d_volume(path) is False

    def test_4d_rejected(self, tmp_path):
        data = np.zeros((64, 64, 32, 10), dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        path = str(tmp_path / "timeseries.nii.gz")
        nib.save(img, path)
        assert is_valid_3d_volume(path) is False

    def test_non_nifti_rejected(self, tmp_path):
        path = str(tmp_path / "notes.txt")
        with open(path, "w") as f:
            f.write("not a scan")
        assert is_valid_3d_volume(path) is False

    def test_nonexistent_rejected(self):
        assert is_valid_3d_volume("/tmp/does_not_exist.nii.gz") is False
