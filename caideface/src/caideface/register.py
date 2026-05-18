"""Step 3: Affine registration with BRAINSFit, face mask warping, and defacing."""

import os
import csv
import logging
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.linalg import inv
from natsort import natsorted

from .background import detect_background_value

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bundled data helpers
# ---------------------------------------------------------------------------

def _data_dir() -> str:
    """Return the path to the bundled data directory."""
    return os.path.join(os.path.dirname(__file__), "data")


def default_template_path(modality: str = "mri") -> str:
    """Return the default template path for the given modality."""
    if modality == "ct":
        return default_ct_template_path()
    return os.path.join(_data_dir(), "mni_icbm152_t1_tal_nlin_sym_55_ext_brain_only.nii.gz")


def default_face_mask_path(modality: str = "mri") -> str:
    """Return the default face mask path for the given modality."""
    if modality == "ct":
        return os.path.join(_data_dir(), "ct_face_mask.nii.gz")
    return os.path.join(_data_dir(), "t1_mask.nii.gz")


def default_ct_template_path() -> str:
    """Return the CT brain atlas from the TotalSegmentator installation."""
    try:
        import totalsegmentator
        ts_dir = os.path.dirname(totalsegmentator.__file__)
        ct_atlas = os.path.join(ts_dir, "resources", "ct_brain_atlas_1mm.nii.gz")
        if os.path.isfile(ct_atlas):
            return ct_atlas
    except ImportError:
        pass
    raise FileNotFoundError(
        "CT brain atlas not found. Install TotalSegmentator with: pip install caideface[ct]"
    )


# ---------------------------------------------------------------------------
# Transform I/O (supports both plain 4x4 text and ITK/Slicer formats)
# ---------------------------------------------------------------------------

def read_slicer_transform_as_forward_in_RAS(txtfile_path: str) -> np.ndarray:
    """Read an ITK/3D-Slicer affine transform file and return it as a
    forward-direction 4x4 matrix in RAS coordinates."""
    import SimpleITK as sitk

    sitk_tfm = sitk.AffineTransform(sitk.ReadTransform(txtfile_path))

    A = np.array(sitk_tfm.GetMatrix()).reshape(3, 3)
    t = np.array(sitk_tfm.GetTranslation())
    c = np.array(sitk_tfm.GetCenter())

    # T(x) = A(x-c) + t + c  =>  translation = -A@c + t + c
    t_aff = -A @ c + t + c

    mat = np.eye(4)
    mat[:3, :3] = A
    mat[:3, 3] = t_aff

    # LPS -> RAS conversion
    RAS_to_LPS = np.diag([-1, -1, 1, 1])
    mat_RAS = RAS_to_LPS @ mat @ RAS_to_LPS

    # Return forward (inverse of resampling direction)
    return inv(mat_RAS)


def load_transform(path: str) -> np.ndarray:
    """Load a transform from a text file, auto-detecting ITK vs plain format."""
    with open(path, "r") as f:
        first_line = f.readline()

    if "#Insight Transform File" in first_line:
        return read_slicer_transform_as_forward_in_RAS(path)
    return np.loadtxt(path)


# ---------------------------------------------------------------------------
# Apply affine: warp floating image sform using a registration transform
# ---------------------------------------------------------------------------

def apply_affine_to_sform(
    floating_path: str,
    reference_path: str,
    transform_path: str,
    output_dir: str,
) -> str:
    """Apply a registration transform to the sform of a floating image.

    Saves a warped NIfTI whose data is unchanged but whose affine encodes
    the registration to the reference.

    Returns the path to the warped image.
    """
    flo_nii = nib.load(floating_path)
    reg_transform = load_transform(transform_path)

    warped_affine = inv(reg_transform) @ flo_nii.affine

    flo_name = os.path.basename(floating_path).replace(".nii.gz", "")
    warped_path = os.path.join(output_dir, flo_name + "_warped.nii.gz")

    warped_nii = nib.Nifti1Image(flo_nii.get_fdata(), warped_affine)
    warped_nii.header["qform_code"] = 0
    nib.save(warped_nii, warped_path)

    return warped_path


# ---------------------------------------------------------------------------
# BRAINSFit registration
# ---------------------------------------------------------------------------

def run_brainsfit(
    fixed_path: str,
    moving_path: str,
    output_volume_path: str,
    output_transform_path: str,
    brainsfit_path: str,
    background_value: float = 0,
) -> int:
    """Run BRAINSFit affine registration and return the exit code."""
    cmd = (
        f'"{brainsfit_path}" '
        f'--fixedVolume "{fixed_path}" '
        f'--movingVolume "{moving_path}" '
        f'--outputVolume "{output_volume_path}" '
        f'--outputTransform "{output_transform_path}" '
        f'--samplingPercentage 0.1 '
        f'--splineGridSize 14,10,12 '
        f'--initializeTransformMode useMomentsAlign '
        f'--useRigid --useAffine '
        f'--maskProcessingMode NOMASK '
        f'--medianFilterSize 0,0,0 '
        f'--removeIntensityOutliers 0 '
        f'--outputVolumePixelType float '
        f'--backgroundFillValue {background_value} '
        f'--interpolationMode Linear '
        f'--numberOfIterations 1500 '
        f'--maximumStepLength 0.05 '
        f'--minimumStepLength 0.001 '
        f'--relaxationFactor 0.5 '
        f'--translationScale 1000 '
        f'--reproportionScale 1 '
        f'--skewScale 1 '
        f'--maxBSplineDisplacement 0 '
        f'--fixedVolumeTimeIndex 0 '
        f'--movingVolumeTimeIndex 0 '
        f'--numberOfHistogramBins 50 '
        f'--numberOfMatchPoints 10 '
        f'--costMetric MMI '
        f'--maskInferiorCutOffFromCenter 1000 '
        f'--ROIAutoDilateSize 0 '
        f'--ROIAutoClosingSize 9 '
        f'--numberOfSamples 0 '
        f'--failureExitCode -1 '
        f'--numberOfThreads -1 '
        f'--debugLevel 0 '
        f'--costFunctionConvergenceFactor 2e+13 '
        f'--projectedGradientTolerance 1e-05 '
        f'--maximumNumberOfEvaluations 900 '
        f'--maximumNumberOfCorrections 25 '
        f'--metricSamplingStrategy Random '
        f'>> /dev/null'
    )
    return os.system(cmd)


def run_brainsresample(
    input_volume: str,
    reference_volume: str,
    output_volume: str,
    transform_path: str,
    brainsresample_path: str,
    inverse: bool = True,
) -> int:
    """Run BRAINSResample and return the exit code."""
    cmd = (
        f'"{brainsresample_path}" '
        f'--inputVolume "{input_volume}" '
        f'--referenceVolume "{reference_volume}" '
        f'--outputVolume "{output_volume}" '
        f'--warpTransform "{transform_path}" '
        f'{"--inverseTransform " if inverse else ""}'
        f'--interpolationMode NearestNeighbor'
    )
    return os.system(cmd)


# ---------------------------------------------------------------------------
# Defacing logic
# ---------------------------------------------------------------------------

def _build_scan_mapping(
    floating_imgs: list[str],
    reoriented_dir: str,
    skullstripped_dir: str,
) -> dict:
    """Map dilated skull-stripped scans to their reoriented originals and brain masks.

    Returns a dict: {floating_path: (reoriented_path, brain_mask_path)}
    """
    reoriented_files = natsorted(glob(os.path.join(reoriented_dir, "**", "*.nii.gz"), recursive=True))
    brain_masks = natsorted(glob(os.path.join(skullstripped_dir, "**", "*_mask.nii.gz"), recursive=True))

    mapping = {}
    for fimg in floating_imgs:
        stem = os.path.basename(fimg).replace("_dilated.nii.gz", "")
        reoriented = next((r for r in reoriented_files if os.path.basename(r) == f"{stem}.nii.gz"), None)
        mask = next((m for m in brain_masks if os.path.basename(m) == f"{stem}_mask.nii.gz"), None)
        if reoriented and mask:
            mapping[fimg] = (reoriented, mask)
        else:
            logger.warning("No match for %s (reoriented=%s, mask=%s)", fimg, reoriented, mask)
    return mapping


def deface_single(
    floating_path: str,
    reoriented_path: str,
    brain_mask_path: str,
    results_dir: str,
    target_path: str,
    face_mask_path: str,
    brainsfit_path: str,
    brainsresample_path: str,
    modality: str = "mri",
    existing_transform: str | None = None,
    background_value: float | None = None,
) -> bool:
    """Register, warp face mask, and deface a single scan.

    Returns True on success, False on failure.
    """
    os.makedirs(results_dir, exist_ok=True)

    base = os.path.basename(floating_path)
    masked_path = os.path.join(results_dir, base.replace(".nii.gz", "_masked.nii.gz"))

    # Skip if already defaced
    if os.path.isfile(masked_path):
        logger.info("Already defaced, skipping: %s", masked_path)
        return True

    # --- Background value detection (always runs per volume) ---
    floating_nii = nib.load(reoriented_path)
    floating_data = floating_nii.get_fdata()

    detected_bg = detect_background_value(floating_data, modality)
    if background_value is not None:
        logger.info(
            "Detected background: %.1f, using override: %.1f",
            detected_bg, background_value,
        )
    else:
        background_value = detected_bg
        logger.info("Using detected background: %.1f", background_value)

    out_affine = os.path.join(results_dir, base.replace(".nii.gz", ".txt"))
    out_resampled = os.path.join(results_dir, base.replace(".nii.gz", "_resampled.nii.gz"))

    # --- Registration ---
    if existing_transform and os.path.isfile(existing_transform):
        logger.info("Using existing transform: %s", existing_transform)
        out_affine = existing_transform
    else:
        logger.info("Running BRAINSFit registration...")
        exit_code = run_brainsfit(
            target_path, floating_path, out_resampled, out_affine, brainsfit_path,
            background_value=background_value,
        )
        if exit_code != 0:
            logger.error("BRAINSFit failed (exit %d) for %s", exit_code, floating_path)
            return False
        logger.info("BRAINSFit completed successfully")

    # --- Apply affine to warp face mask into scan space ---
    apply_affine_to_sform(face_mask_path, floating_path, out_affine, results_dir)

    # --- Resample face mask ---
    face_mask_resampled = os.path.join(
        results_dir,
        os.path.basename(face_mask_path).replace(".nii.gz", "_resampled.nii.gz"),
    )
    run_brainsresample(
        face_mask_path, floating_path, face_mask_resampled,
        out_affine, brainsresample_path, inverse=True,
    )

    # --- Combine masks and deface ---
    face_data = nib.load(face_mask_resampled).get_fdata()
    brain_data = nib.load(brain_mask_path).get_fdata()

    mask_data = np.clip(face_data + brain_data, 0, 1)

    if floating_data.ndim == 4:
        defaced = np.where(mask_data[..., np.newaxis] > 0, floating_data, background_value)
    else:
        defaced = np.where(mask_data > 0, floating_data, background_value)

    nib.save(nib.Nifti1Image(defaced, floating_nii.affine), masked_path)
    logger.info("Saved defaced scan: %s", masked_path)
    return True


def deface_batch(
    reoriented_dir: str,
    skullstripped_dir: str,
    output_dir: str,
    brainsfit_path: str,
    brainsresample_path: str,
    modality: str = "mri",
    target_path: str | None = None,
    face_mask_path: str | None = None,
    background_value: float | None = None,
) -> list[str]:
    """Run the full defacing pipeline (Step 3) on all dilated scans.

    Parameters
    ----------
    reoriented_dir : str
        Directory with reoriented scans from Step 1.
    skullstripped_dir : str
        Directory with skull-stripped outputs from Step 2.
    output_dir : str
        Where defaced scans will be saved.
    brainsfit_path : str
        Path to the BRAINSFit executable.
    brainsresample_path : str
        Path to the BRAINSResample executable.
    modality : str
        Image modality: ``'mri'`` or ``'ct'``.
    target_path : str or None
        Path to skull-stripped MNI152 template. Uses bundled if None.
    face_mask_path : str or None
        Path to face mask in MNI152 space. Uses bundled if None.
    background_value : float or None
        Value to fill defaced regions. Auto-detected per volume when None.

    Returns
    -------
    list[str]
        Paths of scans that failed to deface.
    """
    if target_path is None:
        target_path = default_template_path(modality)
    if face_mask_path is None:
        face_mask_path = default_face_mask_path(modality)

    skullstripped_dir = os.path.abspath(skullstripped_dir)
    reoriented_dir = os.path.abspath(reoriented_dir)
    output_dir = os.path.abspath(output_dir)

    floating_imgs = natsorted(
        glob(os.path.join(skullstripped_dir, "**", "*_dilated.nii.gz"), recursive=True)
    )

    if not floating_imgs:
        logger.warning("No dilated skull-stripped scans found in %s", skullstripped_dir)
        return []

    logger.info("Found %d scans to deface", len(floating_imgs))

    # Build mapping: floating -> (reoriented, brain_mask)
    mapping = _build_scan_mapping(floating_imgs, reoriented_dir, skullstripped_dir)

    # Check for pre-existing transforms
    existing_transforms = {}
    for fimg in floating_imgs:
        tfm_path = os.path.join(os.path.dirname(fimg), "Transform_to_template.txt")
        if os.path.isfile(tfm_path):
            existing_transforms[fimg] = tfm_path

    failed = []
    for fimg in floating_imgs:
        if fimg not in mapping:
            logger.warning("Skipping %s: no matching reoriented scan / mask", fimg)
            failed.append(fimg)
            continue

        reoriented_path, brain_mask_path = mapping[fimg]
        rel = os.path.relpath(os.path.dirname(fimg), skullstripped_dir)
        results_dir = os.path.join(output_dir, rel)

        try:
            ok = deface_single(
                floating_path=fimg,
                reoriented_path=reoriented_path,
                brain_mask_path=brain_mask_path,
                results_dir=results_dir,
                target_path=target_path,
                face_mask_path=face_mask_path,
                brainsfit_path=brainsfit_path,
                brainsresample_path=brainsresample_path,
                modality=modality,
                existing_transform=existing_transforms.get(fimg),
                background_value=background_value,
            )
            if not ok:
                failed.append(fimg)
        except Exception:
            logger.exception("Error defacing %s", fimg)
            failed.append(fimg)

    # Write failure log
    if failed:
        csv_path = os.path.join(output_dir, "not_defaced_scans.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scan_path"])
            for p in failed:
                writer.writerow([p])
        logger.warning("%d scans failed. See %s", len(failed), csv_path)
    else:
        logger.info("All scans successfully defaced.")

    return failed
