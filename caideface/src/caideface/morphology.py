"""Exact binary dilation for the pipeline's rectangular structuring elements."""

import numpy as np


def validate_device(device):
    """Exercise pooling on the selected device before expensive extraction."""
    import torch
    from torch.nn import functional as F

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")
    with torch.inference_mode():
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        probe = torch.ones((1, 1, 2, 2, 2), device=device, dtype=dtype)
        F.max_pool3d(probe, 2).sum().item()


def rectangular_dilation(volume, structure):
    """Return a boolean tensor on the input tensor's device.

    Three 1D pooling passes match SciPy's default origin and zero boundaries
    for solid rectangular structures, including even-sized kernels.
    """
    import torch
    from torch.nn import functional as F

    structure = np.asarray(structure)
    if (volume.ndim != 3 or structure.ndim != 3 or
            not all(volume.shape) or not all(structure.shape) or
            not np.all(structure)):
        raise ValueError("Separable dilation requires a nonempty 3D volume and solid rectangular structure")

    with torch.inference_mode():
        dtype = torch.float16 if volume.device.type == "cuda" else torch.float32
        tensor = (volume != 0).to(dtype=dtype)[None, None]
        for axis, size in enumerate(structure.shape):
            kernel = [1, 1, 1]
            kernel[axis] = size
            pads = [0] * 6
            offset = 2 * (2 - axis)
            # SciPy reflects the structure: even kernels need extra right padding.
            pads[offset] = (size - 1) // 2
            pads[offset + 1] = size // 2
            if size % 2:
                padding = [0, 0, 0]
                padding[axis] = size // 2
                tensor = F.max_pool3d(tensor, kernel, stride=1, padding=padding)
            else:
                tensor = F.max_pool3d(F.pad(tensor, pads, value=0), kernel, stride=1)
        return tensor[0, 0] > 0
