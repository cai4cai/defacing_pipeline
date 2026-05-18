"""Full defacing pipeline orchestrator: reorient -> skull-strip -> register & deface."""

import logging
import os

import pandas as pd

from .reorient import reorient_batch
from .skull_strip import skull_strip_batch
from .register import deface_batch

logger = logging.getLogger(__name__)


class DefacePipeline:
    """Orchestrates the three-step defacing pipeline.

    Parameters
    ----------
    brainsfit_path : str
        Path to the BRAINSFit executable (from 3D Slicer).
    brainsresample_path : str
        Path to the BRAINSResample executable (from 3D Slicer).
    modality : str
        Image modality: ``'mri'`` or ``'ct'``.
    device : str or None
        'cpu' or 'cuda' for HD-BET. Auto-detected if None.
    disable_tta : bool
        Disable HD-BET test-time augmentation for faster processing.
    desired_dilation_mm : float
        Physical dilation in mm for brain mask expansion.
    background_value : float or None
        Value for defaced voxels. Auto-detected per volume when None.
    target_path : str or None
        Custom MNI152 skull-stripped template. Uses bundled if None.
    face_mask_path : str or None
        Custom face mask in MNI152 space. Uses bundled if None.
    """

    def __init__(
        self,
        brainsfit_path: str,
        brainsresample_path: str,
        modality: str = "mri",
        device: str | None = None,
        disable_tta: bool = True,
        desired_dilation_mm: float = 14.0,
        background_value: float | None = None,
        target_path: str | None = None,
        face_mask_path: str | None = None,
    ):
        self.brainsfit_path = brainsfit_path
        self.brainsresample_path = brainsresample_path
        self.modality = modality
        self.device = device
        self.disable_tta = disable_tta
        self.desired_dilation_mm = desired_dilation_mm
        self.background_value = background_value
        self.target_path = target_path
        self.face_mask_path = face_mask_path

    def run(
        self,
        input_dir: str,
        output_dir: str,
        steps: str = "all",
    ) -> dict:
        """Run the defacing pipeline.

        Parameters
        ----------
        input_dir : str
            Directory containing raw NIfTI (.nii.gz) files.
        output_dir : str
            Root output directory. Subdirectories will be created:
            ``reoriented/``, ``skullstripped/``, ``defaced/``.
        steps : str
            Which steps to run: 'all', 'reorient', 'skull_strip', 'deface',
            or comma-separated combination like 'reorient,skull_strip'.

        Returns
        -------
        dict
            Summary with keys: reorient_log, skull_strip_log, failed_defacing.
        """
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)

        reoriented_dir = os.path.join(output_dir, "reoriented")
        skullstripped_dir = os.path.join(output_dir, "skullstripped")
        defaced_dir = os.path.join(output_dir, "defaced")

        run_steps = set(s.strip() for s in steps.split(",")) if steps != "all" else {"reorient", "skull_strip", "deface"}

        results = {}

        # Step 1: Reorientation
        if "reorient" in run_steps:
            logger.info("=" * 60)
            logger.info("STEP 1: Reorientation to RAS")
            logger.info("=" * 60)
            reorient_log = reorient_batch(input_dir, reoriented_dir)
            results["reorient_log"] = reorient_log
        else:
            logger.info("Skipping Step 1 (reorientation)")

        # Step 2: Skull-stripping
        if "skull_strip" in run_steps:
            logger.info("=" * 60)
            backend = "TotalSegmentator" if self.modality == "ct" else "HD-BET"
            logger.info("STEP 2: Skull-stripping with %s", backend)
            logger.info("=" * 60)
            skull_strip_log = skull_strip_batch(
                input_dir=reoriented_dir,
                output_dir=skullstripped_dir,
                modality=self.modality,
                device=self.device,
                disable_tta=self.disable_tta,
                desired_dilation_mm=self.desired_dilation_mm,
            )
            results["skull_strip_log"] = skull_strip_log
        else:
            logger.info("Skipping Step 2 (skull-stripping)")

        # Step 3: Registration & Defacing
        if "deface" in run_steps:
            logger.info("=" * 60)
            logger.info("STEP 3: Registration & Defacing")
            logger.info("=" * 60)
            failed = deface_batch(
                reoriented_dir=reoriented_dir,
                skullstripped_dir=skullstripped_dir,
                output_dir=defaced_dir,
                brainsfit_path=self.brainsfit_path,
                brainsresample_path=self.brainsresample_path,
                modality=self.modality,
                target_path=self.target_path,
                face_mask_path=self.face_mask_path,
                background_value=self.background_value,
            )
            results["failed_defacing"] = failed
        else:
            logger.info("Skipping Step 3 (registration & defacing)")

        logger.info("Pipeline finished.")
        return results
