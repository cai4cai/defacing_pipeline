# Exact rectangular dilation backends

| Matched dilation-only benchmark, six fixed masks | Batch time |
|---|---:|
| Separable CPU | 25.87 s |
| CUDA | 2.66 s |

CUDA was **9.74× faster than separable CPU** in this comparison. Times are means
of two repetitions on the same six saved masks; extraction, registration and
file I/O are excluded. Both implementations reproduced the original SciPy
masks exactly. This is not a whole-pipeline speedup or a matched timing comparison
against the original SciPy implementation. Results depend on image/kernel sizes
and hardware (RTX 5090 for CUDA).

For a 30×30×30 box, three separable passes reduce nominal neighbourhood work
from 27,000 positions to 30 + 30 + 30 = 90: a **300× theoretical work reduction**,
not a measured runtime speedup. Earlier brain-to-mask intervals included I/O
and must not be divided by a dilation-only timer to claim a kernel speedup.

`scipy` remains the default. `separable_cpu` and `cuda` implement solid rectangular
3D dilation with the same even-kernel origin and zero-boundary convention.
The CUDA implementation uses PyTorch max-pooling and returns a NumPy boolean
mask. CUDA is opt-in and fails if unavailable; there is no silent CPU fallback.
HD-BET, BRAINSFit, serial processing and existing pipeline result fields remain
unchanged. Exactness refers to dilation of the same input mask, not equivalence
of different extraction or registration runs.

A separate six-mask comparison used two warm-ups and 30 randomised rounds
(540 calls): CUDA max-pool 2.8853 s, Gaussian convolution thresholded at >0
2.8794 s, and box-average pooling thresholded at >0 2.8398 s. All tested masks
matched, but the 0.2–1.6% timing differences were immaterial. Positive Gaussian
thresholds changed masks. The implementation retains exact max-pool dilation.

Synthetic tests cover odd/even kernels, kernels larger than the image, empty/full
masks, corner boundaries, random masks, noncontiguous input, unsupported structures,
CUDA unavailability, and CLI/Python backend forwarding. GPU availability controls
whether CUDA parity tests run; skipped GPU checks are not successful GPU tests.
