"""caideface - MRI defacing and text anonymisation pipeline from cai4cai.

A three-step pipeline for anonymising head MRI scans:
1. Reorientation to MNI152 atlas reference (nibabel)
2. Skull-stripping with HD-BET and dynamic dilation
3. Affine registration and defacing (BRAINSFit)

Plus standalone text anonymisation via NER + HIPS (Hiding in Plain Sight).
"""

__version__ = "0.3.2"

from .pipeline import DefacePipeline
from .reorient import reorient_batch, reorient_single
from .skull_strip import skull_strip_batch, skull_strip_single
from .register import deface_batch, deface_single
from .anonymize import anonymize_batch, anonymize_single, default_ner_model_path
from .background import detect_background_value

__all__ = [
    "DefacePipeline",
    "reorient_batch",
    "reorient_single",
    "skull_strip_batch",
    "skull_strip_single",
    "deface_batch",
    "deface_single",
    "anonymize_batch",
    "anonymize_single",
    "default_ner_model_path",
    "detect_background_value",
]
