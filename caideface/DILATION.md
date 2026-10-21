# Exact rectangular dilation backends

| Method | Total dilation time (6 masks) | Speedup relative to main |
|---|---:|---:|
| Original full 3D SciPy dilation (main) | 2,210 s | 1× |
| Separable CPU (three 1D passes) | 26.2 s | about 80× |
| Separable CUDA (three 1D passes) | 2.95 s | about 750× |

These are indicative dilation-only measurements, not whole-pipeline speedups.
Six precomputed binary masks were measured once per method on a shared workstation,
with backend order rotated between inputs. Settings were identical: the existing
14 mm parameter and per-input rectangular kernels capped at 30 voxels per axis.
All six outputs matched main exactly. The inputs are not distributed, and their
shapes, spacings and kernel sizes are not reported here, so this table is not a
reproducible public benchmark or a prediction for other volumes. Runtime depends
strongly on those dimensions and concurrent load.

Times sum serial dilation calls, excluding image loading, kernel construction,
output comparisons, torch import and CUDA initialisation. CUDA transfers are
included after a small untimed warm-up. Cold-process startup adds overhead that
was not measured in this benchmark. Hardware: Intel Core Ultra 9 285K, four-CPU
quota / 12 GiB RAM, RTX 5090; other workloads were active. Versions: NumPy 1.26.4,
SciPy 1.15.3, PyTorch 2.10.0+cu128. These timings predate the change to transfer
boolean results back from CUDA; that change has not been rebenchmarked.

“Separable” replaces one full 3D box dilation with three successive 1D dilations:
X, then Y, then Z. For a solid rectangular box, the result is identical when
kernel centering and boundaries match; this does not generally apply to spheres.

At 1 mm isotropic spacing, the default parameter produces a 14×14×14 box:
nominal neighbourhood work falls from 2,744 to 42 positions (about 65×).
At the 30×30×30 kernel cap, it falls from 27,000 to 90 (300×). With the default
parameter, that cap is reached at approximately 0.47 mm spacing or smaller along
each axis. These are theoretical work reductions, not measured runtime speedups.

`scipy` remains the default. `separable_cpu` and `cuda` preserve even-kernel origins
and zero boundaries. CUDA uses float32 max-pooling and returns a NumPy boolean
mask, transferring one byte per voxel to the host. Intermediate and padded tensors
still require substantially more GPU memory than the input boolean mask; allow
several float32 volume copies plus padding and memory used by other stages.
The CUDA backend uses the current PyTorch device, respecting `CUDA_VISIBLE_DEVICES`,
independently of the brain-extraction `--device` option.

CUDA is opt-in. Pipeline construction and skull-stripping entry points check CUDA
availability and execute a small pooling operation before extraction. Failures
propagate instead of falling back to CPU. This does not guarantee sufficient
memory for a full volume: later per-file failures retain existing pipeline error
handling and must be checked in the processing log.

HD-BET, BRAINSFit and serial processing remain unchanged. Exactness concerns
dilation of the same input mask, not equivalence of different extraction or
registration runs. Synthetic tests cover odd/even and oversized kernels, image
boundaries, numeric/noncontiguous inputs, singleton axes, anisotropic uint8
structures, early CUDA failures and CLI forwarding and batch-to-saved-mask integration. CUDA parity
tests skip when no GPU is available; skipped checks are not successful GPU tests.

For a public synthetic comparison, run from the `caideface/` directory with the
package installed:

```bash
python benchmarks/dilation.py --shape 96 96 64 --kernel 14 14 14 --repeats 3 --pattern ellipsoid --cuda
```

The `ellipsoid` pattern is a centred solid blob with radii one third of each
image dimension. The default `random` pattern sets each voxel with 1% probability.
Mask geometry affects runtime; neither synthetic pattern represents every scan.

Omit `--cuda` for CPU-only testing. The script reports exact dimensions, individual
call times and preflight time (including torch import/CUDA initialisation when
selected), checks equality outside the timer, and warms each backend before
randomising timed call order. It does not reproduce the private six-mask table.
