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


# ---------------------------------------------------------------------------
# Brain extraction backends
# ---------------------------------------------------------------------------

def _extract_brain_hdbet(
    input_file: str, output_file: str, device: str = "cpu", disable_tta: bool = True
) -> None:
    """Run HD-BET brain extraction (MRI)."""
    hdbet_bin = _hd_bet_executable()
    cmd = [hdbet_bin, "-i", input_file, "-o", output_file, "-device", device]
    if disable_tta:
        cmd.append("--disable_tta")
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _extract_brain_totalseg(
    input_file: str, output_file: str, binary_mask_file: str, device: str = "cpu"
) -> None:
    """Run TotalSegmentator brain extraction (CT).

    Saves both a brain-extracted volume (*output_file*) and the raw binary
    brain mask (*binary_mask_file*).  The mask is needed because CT brain-
    extracted volumes have negative values, so ``data > 0`` cannot be used
    to recover the mask (unlike MRI).

    Requires ``pip install caideface[ct]``.
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
        import totalsegmentator.config as _ts_config
    except ImportError:
        raise ImportError(
            "TotalSegmentator is required for CT skull-stripping.\n"
            "Install it with: pip install caideface[ct]"
        )

    # Disable anonymous usage statistics — no data should leave the machine.
    # Patch send_usage_stats to a no-op so no network calls are made.
    _orig_send = _ts_config.send_usage_stats
    _ts_config.send_usage_stats = lambda *a, **kw: None

    ts_device = "cpu" if device == "cpu" else "gpu"

    try:
        input_img = nib.load(input_file)
        brain_mask = totalsegmentator(
            input=input_img,
            task="total",
            roi_subset=["brain"],
            device=ts_device,
            quiet=True,
        )
    finally:
        _ts_config.send_usage_stats = _orig_send

    # Save the raw binary brain mask
    mask_data = brain_mask.get_fdata().astype(np.uint8)
    nib.save(
        nib.Nifti1Image(mask_data, input_img.affine, input_img.header),
        binary_mask_file,
    )

    # Apply brain mask to get brain-extracted volume
    data = input_img.get_fdata()
    brain_data = data * mask_data
    nib.save(
        nib.Nifti1Image(brain_data, input_img.affine, input_img.header),
        output_file,
    )


# ---------------------------------------------------------------------------
# Single / batch skull-stripping
# ---------------------------------------------------------------------------

def skull_strip_single(
    input_file: str,
    input_root: str,
    output_dir: str,
    modality: str = "mri",
    device: str = "cpu",
    disable_tta: bool = True,
    desired_dilation_mm: float = 14.0,
) -> dict:
    """Extract brain, create dilated mask, and apply it to a single NIfTI file.

    Parameters
    ----------
    input_file : str
        Path to a reoriented NIfTI file.
    input_root : str
        Root input directory (used to compute relative paths).
    output_dir : str
        Root output directory.
    modality : str
        ``'mri'`` (uses HD-BET) or ``'ct'`` (uses TotalSegmentator).
    device : str
        'cpu' or 'cuda'.
    disable_tta : bool
        Disable HD-BET test-time augmentation (MRI only).
    desired_dilation_mm : float
        Physical dilation size in mm for mask expansion.

    Returns
    -------
    dict
        Status dict with keys: input, output, brain_extract, mask, dilated.
    """
    filename = os.path.basename(input_file)
    stem = filename.replace(".nii.gz", "")
    subfolder = os.path.relpath(os.path.dirname(input_file), start=input_root)

    brain_file = os.path.join(output_dir, subfolder, f"{stem}_brain.nii.gz")
    mask_file = os.path.join(output_dir, subfolder, f"{stem}_mask.nii.gz")
    dilated_file = os.path.join(output_dir, subfolder, f"{stem}_dilated.nii.gz")

    os.makedirs(os.path.dirname(brain_file), exist_ok=True)

    status = {
        "brain_extract": "skipped" if os.path.exists(brain_file) else "pending",
        "mask": "skipped" if os.path.exists(mask_file) else "pending",
        "dilated": "skipped" if os.path.exists(dilated_file) else "pending",
    }

    try:
        # --- Brain extraction (modality-dependent) ---
        # For CT, a separate binary mask file is saved alongside the
        # brain-extracted volume because CT values include negatives,
        # making ``data > 0`` unreliable for mask recovery.
        ct_binary_mask_file = os.path.join(
            output_dir, subfolder, f"{stem}_brain_binary_mask.nii.gz"
        )
        if not os.path.exists(brain_file):
            if modality == "ct":
                logger.info("Running TotalSegmentator brain extraction...")
                _extract_brain_totalseg(input_file, brain_file, ct_binary_mask_file, device)
            else:
                logger.info("Running HD-BET brain extraction...")
                try:
                    _extract_brain_hdbet(input_file, brain_file, device, disable_tta)
                except subprocess.CalledProcessError as e:
                    logger.error("HD-BET failed for %s: %s", input_file, e.stderr)
                    status["brain_extract"] = "failed"
                    return {"input": input_file, "output": brain_file, **status, "error": e.stderr}
            status["brain_extract"] = "success"

        # --- Binary mask + dilation ---
        if not os.path.exists(mask_file):
            if modality == "ct" and os.path.exists(ct_binary_mask_file):
                # Use the raw TotalSegmentator binary mask directly
                img = nib.load(ct_binary_mask_file)
                binary_volume = (img.get_fdata() > 0).astype(np.uint8)
            else:
                # MRI: threshold the brain-extracted volume
                img = nib.load(brain_file)
                binary_volume = (img.get_fdata() > 0).astype(np.uint8)

            voxel_sizes = img.header.get_zooms()[:3]
            struct_elem = get_safe_structuring_element(voxel_sizes, desired_dilation_mm)
            dilated_data = binary_dilation(binary_volume, structure=struct_elem)

            nib.save(nib.Nifti1Image(dilated_data, img.affine, img.header), mask_file)
            status["mask"] = "success" if os.path.exists(mask_file) else "failed"

        # --- Apply dilated mask to original scan ---
        if not os.path.exists(dilated_file):
            mask_img = nib.load(mask_file)
            mask_data = (mask_img.get_fdata() > 0).astype(np.uint8)

            original_img = nib.load(input_file)
            original_data = original_img.get_fdata()

            # Detect background so outside-mask voxels keep the correct
            # intensity (e.g. ~-1000 HU for CT instead of 0).
            from .background import detect_background_value
            bg = detect_background_value(original_data, modality)

            dilated_volume = np.where(mask_data, original_data, bg)
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

    return {"input": input_file, "output": brain_file, **status}


def skull_strip_batch(
    input_dir: str,
    output_dir: str,
    modality: str = "mri",
    device: str | None = None,
    disable_tta: bool = True,
    desired_dilation_mm: float = 14.0,
) -> pd.DataFrame:
    """Run skull-stripping on all 3D NIfTI volumes under *input_dir*.

    Uses HD-BET for MRI or TotalSegmentator for CT, based on *modality*.
    Scans that are not 3D volumes (2D slices or 4D time series) are skipped.

    Parameters
    ----------
    input_dir : str
        Root directory with reoriented NIfTI files.
    output_dir : str
        Root output directory for skull-stripping results.
    modality : str
        ``'mri'`` (HD-BET) or ``'ct'`` (TotalSegmentator).
    device : str or None
        'cpu' or 'cuda'. Auto-detected if None.
    disable_tta : bool
        Disable test-time augmentation (MRI/HD-BET only).
    desired_dilation_mm : float
        Physical dilation in mm.

    Returns
    -------
    pd.DataFrame
        Processing log.
    """
    # Validate backend is available
    if modality == "mri":
        _hd_bet_executable()
    elif modality == "ct":
        try:
            import totalsegmentator  # noqa: F401
        except ImportError:
            raise ImportError(
                "TotalSegmentator is required for CT skull-stripping.\n"
                "Install it with: pip install caideface[ct]"
            )

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
                input_file, input_dir, output_dir,
                modality=modality,
                device=device,
                disable_tta=disable_tta,
                desired_dilation_mm=desired_dilation_mm,
            )
            log_data.append(result)

    df = pd.DataFrame(log_data)
    log_path = os.path.join(output_dir, "skull_strip_log.csv")

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(log_path, index=False)
    logger.info("Skull-stripping log saved at %s", log_path)

    return df
