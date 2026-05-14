"""Step 1: Reorientation of NIfTI scans to RAS using nibabel."""

import os
import logging

import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation
import pandas as pd

logger = logging.getLogger(__name__)

# RAS matches the MNI152 standard, fslreorient2std, HD-BET, and the CT brain atlas.
TARGET_ORIENTATION = ("R", "A", "S")


def reorient_single(input_file: str, output_file: str) -> bool:
    """Reorient a single NIfTI file to RAS orientation.

    This is equivalent to FSL's ``fslreorient2std`` and matches the
    MNI152 template orientation used by both the MRI and CT pipelines.

    Parameters
    ----------
    input_file : str
        Path to the input NIfTI (.nii.gz) file.
    output_file : str
        Path where the reoriented file will be saved.

    Returns
    -------
    bool
        True if reorientation succeeded, False otherwise.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        img = nib.load(input_file)
        orig_ornt = io_orientation(img.affine)
        target_ornt = axcodes2ornt(TARGET_ORIENTATION)
        transform = ornt_transform(orig_ornt, target_ornt)
        reoriented = img.as_reoriented(transform)
        nib.save(reoriented, output_file)
    except Exception as e:
        logger.error("Failed to reorient %s: %s", input_file, e)
        return False

    return os.path.exists(output_file)


def reorient_batch(input_dir: str, output_dir: str) -> pd.DataFrame:
    """Reorient all NIfTI files found recursively under *input_dir* to RAS.

    The directory structure is mirrored under *output_dir*.

    Parameters
    ----------
    input_dir : str
        Root directory containing NIfTI files.
    output_dir : str
        Root directory where reoriented files will be saved,
        preserving the subdirectory structure.

    Returns
    -------
    pd.DataFrame
        Log with columns ``input``, ``output``, ``success``.
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    logger.info("Target orientation: RAS")

    log_rows = []
    for root, _dirs, files in os.walk(input_dir):
        for fname in files:
            if not fname.endswith(".nii.gz"):
                continue

            input_path = os.path.join(root, fname)
            rel = os.path.relpath(root, input_dir)
            out_path = os.path.join(output_dir, rel, fname)

            logger.info("Reorienting %s", input_path)
            success = reorient_single(input_path, out_path)
            log_rows.append({"input": input_path, "output": out_path, "success": success})

            if success:
                logger.info("OK: %s", fname)
            else:
                logger.warning("FAILED: %s", fname)

    df = pd.DataFrame(log_rows)

    # Save log
    log_path = os.path.join(output_dir, "reorientation_log.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(log_path, index=False)
    logger.info("Reorientation log saved at %s", log_path)

    return df
