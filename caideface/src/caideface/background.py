"""Background value detection for MRI and CT volumes.

CT volumes use different background encodings depending on how the scanner
or PACS exports them.  This module detects the correct background value
from the image histogram so it can be used for mask dilation padding,
BRAINSFit ``--backgroundFillValue``, and the final defacing fill.
"""

import logging

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# Known CT background anchors and their tolerance (HU)
_CT_ANCHORS = {
    "hu": -1000,
    "rescaled": 0,
    "offset": 1024,
}
_ANCHOR_TOLERANCE = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_lowest_peak(volume: np.ndarray, n_bins: int = 256) -> tuple[float, float]:
    """Find the lowest prominent peak in the volume histogram.

    Parameters
    ----------
    volume : np.ndarray
        3D image data.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    tuple[float, float]
        ``(peak_value, fraction)`` where *fraction* is the approximate
        proportion of voxels near the peak.
    """
    flat = volume.ravel()

    # Clip to 0.1–99.9 percentile to suppress extreme outliers
    lo, hi = np.percentile(flat, [0.1, 99.9])
    clipped = flat[(flat >= lo) & (flat <= hi)]

    counts, bin_edges = np.histogram(clipped, bins=n_bins)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Normalise counts for prominence calculation
    counts_norm = counts / counts.max() if counts.max() > 0 else counts

    peak_indices, properties = find_peaks(counts_norm, prominence=0.05)

    if len(peak_indices) == 0:
        # Fallback: use the bin with the highest count
        peak_idx = int(np.argmax(counts))
        peak_val = float(bin_centres[peak_idx])
        fraction = float(counts[peak_idx]) / len(flat)
        return peak_val, fraction

    # Take the peak with the lowest bin centre value
    lowest_idx = peak_indices[np.argmin(bin_centres[peak_indices])]
    peak_val = float(bin_centres[lowest_idx])
    fraction = float(counts[lowest_idx]) / len(flat)

    return peak_val, fraction


def _classify_ct_encoding(peak: float, dtype: np.dtype) -> str:
    """Classify CT encoding based on the background peak and dtype.

    Returns
    -------
    str
        One of ``'hu'``, ``'rescaled'``, ``'offset'``, or ``'unknown'``.
    """
    for name, anchor in _CT_ANCHORS.items():
        if abs(peak - anchor) <= _ANCHOR_TOLERANCE:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_background_value(volume: np.ndarray, modality: str) -> float:
    """Detect the background fill value for a volume.

    For **MRI** the background is always ``0.0`` (no signal outside the body).
    A histogram check is performed and a warning is logged if the lowest
    prominent peak is far from zero.

    For **CT** the function builds a histogram, finds the lowest prominent
    peak (which should correspond to air), classifies the encoding, performs
    consistency checks, and returns the *detected* peak value (not the
    theoretical anchor).

    Parameters
    ----------
    volume : np.ndarray
        3D image data.
    modality : str
        ``'mri'`` or ``'ct'``.

    Returns
    -------
    float
        Detected background value.

    Raises
    ------
    ValueError
        If *modality* is invalid, or if a dtype/value inconsistency is
        detected (e.g. unsigned dtype with a negative peak).
    """
    modality = modality.lower().strip()
    if modality not in ("mri", "ct"):
        raise ValueError(f"modality must be 'mri' or 'ct', got '{modality}'")

    # ------------------------------------------------------------------
    # MRI
    # ------------------------------------------------------------------
    if modality == "mri":
        peak, _ = _find_lowest_peak(volume)
        if abs(peak) > 50:
            logger.warning(
                "MRI background peak at %.1f is far from 0. "
                "Using 0.0 anyway — pass --background-value to override.",
                peak,
            )
        return 0.0

    # ------------------------------------------------------------------
    # CT
    # ------------------------------------------------------------------
    peak, fraction = _find_lowest_peak(volume)
    dtype = volume.dtype
    encoding = _classify_ct_encoding(peak, dtype)

    # Dtype consistency: unsigned dtype cannot have negative values
    if np.issubdtype(dtype, np.unsignedinteger) and peak < 0:
        raise ValueError(
            f"CT volume has unsigned dtype ({dtype}) but histogram peak "
            f"at {peak:.1f} suggests negative values. This is inconsistent — "
            f"check the input data."
        )

    # Volume fraction check: air should dominate in a head scan
    if fraction < 0.005:
        logger.warning(
            "CT background peak at %.1f represents only %.1f%% of voxels "
            "(expected >5%%). Detection may be unreliable.",
            peak,
            fraction * 100,
        )

    if encoding == "unknown":
        logger.warning(
            "CT background peak at %.1f does not match expected anchors "
            "(HU≈-1000, rescaled≈0, offset≈1024). Using detected value.",
            peak,
        )
    else:
        logger.info(
            "Detected CT encoding: %s (background = %.1f)", encoding, peak
        )

    return float(peak)
