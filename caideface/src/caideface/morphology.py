"""Exact binary dilation for the pipeline's rectangular structuring elements."""

import numpy as np


def validate_device(device):
    """Exercise dilation on the selected device before expensive extraction."""
    import torch

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")
    probe = torch.ones((2, 3, 4), device=device, dtype=torch.bool)
    rectangular_dilation(probe, np.ones((2, 3, 4), dtype=bool)).any().item()


def rectangular_dilation(volume, structure):
    """Return a boolean tensor on the input tensor's device.

    Three 1D boolean-OR passes match SciPy's default origin and zero boundaries
    for solid rectangular structures, including even-sized kernels.
    """
    import torch

    structure = np.asarray(structure)
    if (volume.ndim != 3 or structure.ndim != 3 or
            not all(volume.shape) or not all(structure.shape) or
            not np.all(structure)):
        raise ValueError("Separable dilation requires a nonempty 3D volume and solid rectangular structure")

    with torch.no_grad():
        result = volume != 0
        for axis, size in enumerate(structure.shape):
            out = torch.zeros_like(result)
            length = result.shape[axis]
            # SciPy reflects the structure: even kernels extend further right.
            for offset in range(-((size - 1) // 2), size // 2 + 1):
                if abs(offset) >= length:
                    continue
                source = [slice(None)] * 3
                target = [slice(None)] * 3
                if offset >= 0:
                    source[axis] = slice(offset, length)
                    target[axis] = slice(0, length - offset)
                else:
                    source[axis] = slice(0, length + offset)
                    target[axis] = slice(-offset, length)
                out[tuple(target)].logical_or_(result[tuple(source)])
            result = out
        return result
