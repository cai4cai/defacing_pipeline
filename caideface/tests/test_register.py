"""Tests for the register module (unit tests, no BRAINSFit needed)."""

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from caideface.register import (
    default_template_path,
    default_face_mask_path,
    load_transform,
)


class TestBundledData:
    def test_template_exists_and_loads(self):
        path = default_template_path()
        assert os.path.isfile(path), f"Template not found at {path}"
        img = nib.load(path)
        assert len(img.shape) == 3
        assert all(d > 1 for d in img.shape)

    def test_face_mask_exists_and_loads(self):
        path = default_face_mask_path()
        assert os.path.isfile(path), f"Face mask not found at {path}"
        img = nib.load(path)
        assert len(img.shape) == 3
        assert all(d > 1 for d in img.shape)

    def test_face_mask_is_binary(self):
        img = nib.load(default_face_mask_path())
        data = img.get_fdata()
        unique = np.unique(data)
        # Should contain only 0s and 1s (or close to it)
        assert all(np.isclose(v, 0) or np.isclose(v, 1) for v in unique)


class TestLoadTransform:
    def test_plain_4x4_matrix(self, tmp_path):
        # Create a plain text 4x4 identity matrix
        mat = np.eye(4)
        path = str(tmp_path / "transform.txt")
        np.savetxt(path, mat)

        loaded = load_transform(path)
        assert loaded.shape == (4, 4)
        assert np.allclose(loaded, mat)

    def test_plain_affine_matrix(self, tmp_path):
        # Create a non-trivial affine
        mat = np.array([
            [0.9, 0.1, 0.0, 5.0],
            [-0.1, 0.9, 0.0, -3.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        path = str(tmp_path / "affine.txt")
        np.savetxt(path, mat)

        loaded = load_transform(path)
        assert np.allclose(loaded, mat)


class TestMaskCombination:
    def test_face_plus_brain_mask(self):
        """Verify face mask + brain mask combination clips to [0, 1]."""
        face = np.array([0, 1, 1, 0, 0], dtype=float)
        brain = np.array([0, 0, 1, 1, 0], dtype=float)
        combined = np.clip(face + brain, 0, 1)

        expected = np.array([0, 1, 1, 1, 0], dtype=float)
        assert np.array_equal(combined, expected)

    def test_defacing_preserves_brain(self):
        """Verify defacing keeps brain voxels and sets others to background."""
        scan = np.array([100, 200, 300, 400, 500], dtype=float)
        mask = np.array([0, 1, 1, 1, 0], dtype=float)
        background = 0

        defaced = np.where(mask > 0, scan, background)

        assert defaced[0] == background  # outside mask -> background
        assert defaced[1] == 200         # inside mask -> preserved
        assert defaced[4] == background  # outside mask -> background
        assert np.array_equal(defaced[1:4], [200, 300, 400])  # brain preserved

    def test_4d_defacing(self):
        """Verify defacing works on 4D volumes."""
        scan = np.random.default_rng(0).random((4, 4, 4, 3))
        mask = np.zeros((4, 4, 4))
        mask[1:3, 1:3, 1:3] = 1
        background = 0

        defaced = np.where(mask[..., np.newaxis] > 0, scan, background)

        # Inside mask: preserved
        assert np.allclose(defaced[1, 1, 1, :], scan[1, 1, 1, :])
        # Outside mask: background
        assert np.all(defaced[0, 0, 0, :] == background)
