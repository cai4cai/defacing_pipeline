# Exact rectangular dilation

## Current implementation: public synthetic comparison

| Method | Median dilation time (one mask, 3 rounds) | Speedup relative to main |
|---|---:|---:|
| Original full 3D SciPy dilation | 0.531 s | 1× |
| PyTorch separable CPU | 0.0319 s | about 17× |
| PyTorch separable CUDA | 0.00198 s | about 270× |

A solid ellipsoid in a 96×96×64 array, radii one third of each dimension,
14×14×14 kernel, RTX 5090 / Intel Core Ultra 9 285K. Three randomised timed
rounds after warm-up; every output matched SciPy exactly. CPU quota: two cores;
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`. CUDA times ranged from 1.34–3.53 ms.
These are indicative dilation-only timings on a shared workstation, not
whole-pipeline speedups. Transfers are included; import, initialisation, warm-up,
input construction and equality checks are excluded. Preflight took 0.19 s.
Versions: NumPy 1.26.4, SciPy 1.15.3, PyTorch 2.10.0+cu128.

CPU thread settings matter: an initial run with unrestricted thread counts under
the same two-core quota took about 0.70 s for PyTorch CPU versus 0.53 s for SciPy.
Match thread counts to available resources when benchmarking; the library does
not change process-wide thread settings.

## Historical six-mask comparison

| Method | Total dilation time (6 masks) | Speedup relative to main |
|---|---:|---:|
| Original full 3D SciPy dilation (main) | 2,210 s | 1× |
| Historical SciPy separable CPU (three 1D passes) | 26.2 s | about 80× |
| Separable CUDA (three 1D passes) | 2.95 s | about 750× |

These are indicative dilation-only measurements, not whole-pipeline speedups.
Six precomputed binary masks were measured once per method on a shared workstation,
with method order rotated between inputs. Settings were identical: the existing
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
SciPy 1.15.3, PyTorch 2.10.0+cu128. These timings predate the boolean CUDA return-transfer change and the switch
to PyTorch CPU pooling. They do not measure the current CPU implementation.

“Separable” replaces one full 3D box dilation with three successive 1D dilations:
X, then Y, then Z. For a solid rectangular box, the result is identical when
kernel centering and boundaries match; this does not generally apply to spheres.

At 1 mm isotropic spacing, the default parameter produces a 14×14×14 box:
nominal neighbourhood work falls from 2,744 to 42 positions (about 65×).
At the 30×30×30 kernel cap, it falls from 27,000 to 90 (300×). With the default
parameter, that cap is reached at approximately 0.47 mm spacing or smaller along
each axis. These are theoretical work reductions, not measured runtime speedups.

The existing `--device cpu|cuda` option controls both brain extraction and dilation.
If omitted, the pipeline selects CUDA when available, otherwise CPU. The Python
API uses `device`; there is no separate dilation selector. The pooling function
accepts a tensor and returns a boolean tensor on that same device. The pipeline
transfers the boolean mask to the host for saving (one byte per voxel).

Both devices use the same separable float32 pooling implementation. Intermediate
and padded tensors require substantially more memory than a boolean mask; allow
several float32 volume copies plus padding and memory used by other stages.
CUDA respects `CUDA_VISIBLE_DEVICES`. The tensor operation has no CPU/CUDA
allowlist, but other devices such as MPS are not validated for the complete
extraction pipeline; the CLI retains its existing CPU/CUDA choices.

Pipeline construction and skull-stripping entry points exercise a small pooling
operation on the selected device before extraction. Failures propagate rather
than falling back to CPU. This does not guarantee sufficient memory for a full
volume: later per-file failures retain existing pipeline error handling and must
be checked in the processing log.

HD-BET, BRAINSFit and serial processing remain unchanged. Exactness concerns
dilation of the same input mask, not equivalence of different extraction or
registration runs. Synthetic tests cover odd/even and oversized kernels, image
boundaries, numeric/noncontiguous inputs, singleton axes, anisotropic uint8
structures, early CUDA failures and CLI forwarding and batch-to-saved-mask integration. CUDA parity
tests skip when no GPU is available; skipped checks are not successful GPU tests.

For a public synthetic comparison, run from the `caideface/` directory with the
package installed:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python benchmarks/dilation.py --shape 96 96 64 --kernel 14 14 14 --repeats 3 --pattern ellipsoid --cuda
```

The `ellipsoid` pattern is a centred solid blob with radii one third of each
image dimension. The default `random` pattern sets each voxel with 1% probability.
Mask geometry affects runtime; neither synthetic pattern represents every scan.

Omit `--cuda` for CPU-only testing. The script reports exact dimensions, individual
call times and device preflight time (excluding torch import), checks equality outside the timer, and warms each method before
randomising timed call order. It does not reproduce the private six-mask table.
