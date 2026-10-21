# Exact rectangular dilation backends

| Method | Batch time | Speedup relative to main |
|---|---:|---:|
| Original full 3D SciPy dilation (main) | 2,209.94 s | 1.00× |
| Separable CPU (three 1D passes) | 26.20 s | 84.34× |
| Separable CUDA (three 1D passes) | 2.95 s | 749.46× |

Six precomputed binary masks, one timed pass per method, identical 14 mm dilation settings
and per-input rectangular kernels. Backend order rotates between inputs. Batch
time is the sum of serial dilation calls, excluding image loading, kernel
construction and output comparisons; CUDA includes host/device transfers, after
a small untimed warm-up. Ratios use unrounded timings. Intel Core Ultra 9 285K,
four-CPU quota / 12 GiB RAM; RTX 5090. Other workloads
were active on the shared workstation, so these are indicative measurements,
not repeated-trial confidence estimates or whole-pipeline speedups.

All six outputs matched main exactly for both accelerated backends.

“Separable” replaces one full 3D box dilation with three successive 1D dilations:
X, then Y, then Z. For a solid rectangular box, the result is identical when
kernel centering and boundaries match; this does not generally apply to spheres.

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

Benchmark versions: NumPy 1.26.4, SciPy 1.15.3, PyTorch 2.10.0+cu128.
