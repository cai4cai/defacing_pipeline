"""Step 2: Skull-stripping with HD-BET and dynamic mask dilation."""

import os
import subprocess
import sys
import logging

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)


def get_default_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"


def _hd_bet_executable() -> str:
    """Return the path to hd-bet that belongs to the current Python env."""
    env_bin = os.path.dirname(sys.executable)
    env_hdbet = os.path.join(env_bin, "hd-bet")
    if os.path.isfile(env_hdbet):
        return env_hdbet
    # Fall back to PATH
    result = subprocess.run(["which", "hd-bet"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    raise EnvironmentError(
        "hd-bet not found. Install it with: pip install hd-bet\n"
        "See: https://github.com/MIC-DKFZ/HD-BET"
    )


def get_safe_structuring_element(
    voxel_sizes: tuple,
    desired_dilation_mm: float = 14.0,
    max_kernel_shape: tuple = (30, 30, 30),
) -> np.ndarray:
    """Compute a 3D structuring element adapted to voxel sizes.

    This ensures a consistent physical dilation distance in mm regardless
    of voxel resolution.

    Parameters
    ----------
    voxel_sizes : tuple of float
        Voxel spacing (mm) along each axis.
    desired_dilation_mm : float
        Desired physical dilation size in mm.
    max_kernel_shape : tuple of int
        Maximum allowed structuring element size per axis.

    Returns
    -------
    np.ndarray
        Binary structuring element.
    """
    shape = tuple(
        min(mk, max(1, int(round(desired_dilation_mm / vs))))
        for vs, mk in zip(voxel_sizes, max_kernel_shape)
    )
    logger.debug("Structuring element shape: %s", shape)
    return np.ones(shape, dtype=np.uint8)


def is_valid_3d_volume(filepath: str) -> bool:
    """Return True if the NIfTI file is a 3D volume (not 2D or 4D)."""
    if not filepath.endswith(".nii.gz"):
        return False
    try:
        img = nib.load(filepath)
        shape = img.shape
        return len(shape) == 3 and all(d > 1 for d in shape)
    except Exception:
        return False


def skull_strip_single(
    input_file: str,
    input_root: str,
    output_dir: str,
    device: str = "cpu",
    disable_tta: bool = True,
    desired_dilation_mm: float = 14.0,
) -> dict:
    """Run HD-BET on a single NIfTI file and produce dilated brain mask.

    Parameters
    ----------
    input_file : str
        Path to a reoriented NIfTI file.
    input_root : str
        Root input directory (used to compute relative paths).
    output_dir : str
        Root output directory.
    device : str
        'cpu' or 'cuda'.
    disable_tta : bool
        Disable test-time augmentation for faster processing.
    desired_dilation_mm : float
        Physical dilation size in mm for mask expansion.

    Returns
    -------
    dict
        Status dict with keys: input, output, hd_bet, mask, dilated.
    """
    filename = os.path.basename(input_file)
    stem = filename.replace(".nii.gz", "")
    subfolder = os.path.relpath(os.path.dirname(input_file), start=input_root)

    hd_bet_file = os.path.join(output_dir, subfolder, f"{stem}_brain.nii.gz")
    mask_file = os.path.join(output_dir, subfolder, f"{stem}_mask.nii.gz")
    dilated_file = os.path.join(output_dir, subfolder, f"{stem}_dilated.nii.gz")

    os.makedirs(os.path.dirname(hd_bet_file), exist_ok=True)

    status = {
        "hd_bet": "skipped" if os.path.exists(hd_bet_file) else "pending",
        "mask": "skipped" if os.path.exists(mask_file) else "pending",
        "dilated": "skipped" if os.path.exists(dilated_file) else "pending",
    }

    try:
        # --- HD-BET ---
        if not os.path.exists(hd_bet_file):
            hdbet_bin = _hd_bet_executable()
            cmd = [hdbet_bin, "-i", input_file, "-o", hd_bet_file, "-device", device]
            if disable_tta:
                cmd.append("--disable_tta")
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                status["hd_bet"] = "success"
            except subprocess.CalledProcessError as e:
                logger.error("hd-bet failed for %s: %s", input_file, e.stderr)
                status["hd_bet"] = "failed"
                return {"input": input_file, "output": hd_bet_file, **status, "error": e.stderr}

        # --- Binary mask + dilation ---
        if not os.path.exists(mask_file):
            img = nib.load(hd_bet_file)
            data = img.get_fdata()
            voxel_sizes = img.header.get_zooms()[:3]

            struct_elem = get_safe_structuring_element(voxel_sizes, desired_dilation_mm)

            binary_volume = (data > 0).astype(np.uint8)
            dilated_data = binary_dilation(binary_volume, structure=struct_elem)

            nib.save(nib.Nifti1Image(dilated_data, img.affine, img.header), mask_file)
            status["mask"] = "success" if os.path.exists(mask_file) else "failed"

        # --- Apply dilated mask to original scan ---
        if not os.path.exists(dilated_file):
            mask_img = nib.load(mask_file)
            mask_data = (mask_img.get_fdata() > 0).astype(np.uint8)

            original_img = nib.load(input_file)
            original_data = original_img.get_fdata()

            dilated_volume = original_data * mask_data
            nib.save(
                nib.Nifti1Image(dilated_volume, original_img.affine, original_img.header),
                dilated_file,
            )
            status["dilated"] = "success" if os.path.exists(dilated_file) else "failed"

    except Exception as e:
        logger.exception("Error processing %s", input_file)
        for k in status:
            if status[k] == "pending":
                status[k] = "failed"

    return {"input": input_file, "output": hd_bet_file, **status}


def skull_strip_batch(
    input_dir: str,
    output_dir: str,
    device: str | None = None,
    disable_tta: bool = True,
    desired_dilation_mm: float = 14.0,
) -> pd.DataFrame:
    """Run HD-BET skull-stripping on all 3D NIfTI volumes under *input_dir*.

    Scans that are not 3D volumes (2D slices or 4D time series) are skipped.

    Parameters
    ----------
    input_dir : str
        Root directory with reoriented NIfTI files.
    output_dir : str
        Root output directory for HD-BET results.
    device : str or None
        'cpu' or 'cuda'. Auto-detected if None.
    disable_tta : bool
        Disable test-time augmentation.
    desired_dilation_mm : float
        Physical dilation in mm.

    Returns
    -------
    pd.DataFrame
        Processing log.
    """
    # Validate hd-bet is available (will raise if not found)
    _hd_bet_executable()

    if device is None:
        device = get_default_device()

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_data = []

    for root, _dirs, files in os.walk(input_dir):
        for fname in files:
            input_file = os.path.join(root, fname)

            if not is_valid_3d_volume(input_file):
                continue

            logger.info("Processing: %s", input_file)
            result = skull_strip_single(
                input_file, input_dir, output_dir, device, disable_tta, desired_dilation_mm
            )
            log_data.append(result)

    df = pd.DataFrame(log_data)
    log_path = os.path.join(output_dir, "hd_bet_log.csv")

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(log_path, index=False)
    logger.info("HD-BET log saved at %s", log_path)

    return df
