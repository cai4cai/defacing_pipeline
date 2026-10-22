"""Exact binary dilation for the pipeline's rectangular structuring elements."""

import numpy as np

BACKENDS = ("cpu", "cuda")


def validate_dilation_backend(backend):
    """Reject an unusable CUDA backend before starting expensive extraction."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown dilation backend: {backend}")
    if backend == "cuda":
        import torch
        from torch.nn import functional as F

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA dilation requested but CUDA is unavailable")
        # Exercise the actual kernel and synchronise to catch device/build errors.
        with torch.inference_mode():
            probe = torch.ones((1, 1, 2, 2, 2), device="cuda")
            F.max_pool3d(probe, 2).sum().item()


def rectangular_dilation(volume, structure, backend="cpu"):
    """Dilate once with SciPy's default origin and zero outside the image.

    A solid rectangle is separable into three one-dimensional operations. CUDA
    returns a NumPy boolean array, synchronising the device-to-host transfer.
    Both CPU and CUDA use the same PyTorch pooling implementation.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown dilation backend: {backend}")

    volume = np.asarray(volume)
    structure = np.asarray(structure)
    if (volume.ndim != 3 or structure.ndim != 3 or
            not all(volume.shape) or not all(structure.shape) or
            not np.all(structure)):
        raise ValueError("Separable dilation requires a nonempty 3D volume and solid rectangular structure")

    import torch
    from torch.nn import functional as F

    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA dilation requested but CUDA is unavailable")
    with torch.inference_mode():
        tensor = torch.as_tensor(
            np.ascontiguousarray(volume != 0), device=backend, dtype=torch.float32,
        )[None, None]
        for axis, size in enumerate(structure.shape):
            kernel = [1, 1, 1]
            kernel[axis] = size
            pads = [0] * 6
            offset = 2 * (2 - axis)
            # SciPy reflects the structure: even kernels need extra right padding.
            pads[offset] = (size - 1) // 2
            pads[offset + 1] = size // 2
            tensor = F.max_pool3d(F.pad(tensor, pads, value=0), kernel, stride=1)
        return (tensor[0, 0] > 0).cpu().numpy()
