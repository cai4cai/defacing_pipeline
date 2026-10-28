# Exact rectangular dilation

## Boolean implementation

Dilation now uses exact shifted boolean OR operations on the input tensor's device.
This avoids max pooling's floating-point volumes and internal indices on CPU.

A fixed-input comparison across 687 precomputed masks (four CPU threads, RTX 5090,
PyTorch 2.10.0) measured the following totals, excluding input loading and including
CUDA transfers. All outputs matched the separable SciPy reference exactly.
These private inputs are not distributed; the results are indicative, not a
reproducible public benchmark or a whole-pipeline speedup.

| Method | Total dilation time | Peak CUDA allocation |
|---|---:|---:|
| Separable SciPy maximum filter | 66.4 s | — |
| Previous PyTorch CPU max pooling | 619 s | — |
| Boolean PyTorch CPU | 39.6 s | — |
| Previous PyTorch CUDA max pooling | 3.78 s | 1,043 MiB |
| Boolean PyTorch CUDA | 3.89 s | 210 MiB |

In a 12-input pipeline comparison with registration unchanged, separable SciPy
dilation accounted for 1.16 seconds of approximately 800 seconds total. CUDA saved
less than one second in this stage. The major benefit over main remains replacing
full 3D dilation with separable passes, not moving already-separable CPU dilation
to CUDA. The boolean implementation keeps a shared PyTorch path with lower memory
use and faster CPU execution on this workload. Other workloads may rank differently.

## Public synthetic comparison (current boolean implementation)

| Method | Median dilation time (one mask, 5 rounds) | Speedup relative to main |
|---|---:|---:|
| Original full 3D SciPy dilation | 0.393 s | 1× |
| Boolean PyTorch CPU | 0.00213 s | about 180× |
| Boolean PyTorch CUDA | 0.000542 s | about 730× |

Solid ellipsoid, 96×96×64 array, radii one third of each dimension, 14³ kernel;
RTX 5090, two CPU cores and two torch threads (24 inter-op threads).
PyTorch 2.10.0+cu128, NumPy 1.26.4, SciPy 1.15.3. Five randomised rounds after
warm-up; all outputs matched. Transfers included; loading, imports, initialisation
and equality checks excluded. Preflight: 0.154 s. CPU times ranged from 0.00094 to
0.116 s, including a scheduling outlier; CUDA ranged from 0.00043 to 0.00073 s.
These are indicative dilation-only measurements on a shared machine, not
whole-pipeline speedups or precise performance guarantees. Thread settings and
mask geometry matter; the benchmark records both and does not alter library defaults.

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
API uses `device`; there is no separate dilation selector. The dilation function
accepts a tensor and returns a boolean tensor on that same device. The pipeline
transfers the boolean mask to the host for saving (one byte per voxel).

Both devices use three separable boolean-OR passes over shifted slices. No floating
point conversion, pooling indices or padded volume is needed. Even-sized kernels
retain SciPy's asymmetric origin. The result is an ordinary mutable boolean tensor.

CUDA respects `CUDA_VISIBLE_DEVICES`. The tensor operation has no CPU/CUDA
allowlist, but other devices such as MPS are not validated for the complete
extraction pipeline; the CLI retains its existing CPU/CUDA choices.

Device resolution and a tiny dilation probe happen once at skull-stripping batch entry,
not during pipeline construction or once per file. Automatic CUDA selection falls
back to CPU with a warning if the probe reports a recognised missing-kernel/build capability failure; explicit CUDA requests fail early. Probe out-of-memory errors fail early even with automatic selection, so a temporary allocation failure does not move brain extraction to CPU.
A CUDA out-of-memory error during mask transfer or dilation retries that mask on
CPU with a warning, using the retained host mask and the same boolean algorithm.
Unused CUDA cache is released after dilation and before a CPU retry, once temporary tensors are gone. The CUDA context and other live tensors remain allocated. Later masks still attempt their selected device because their memory requirements may differ.
Other errors remain per-file failures, allowing the batch to continue and write
its status CSV. A CPU retry still requires enough host RAM. Direct single-file
calls use their supplied device and retain per-file exception handling.

PyTorch imports in the dilation and skull-stripping modules are deferred until
imaging work needs them. Other package dependencies may independently load torch.

HD-BET, BRAINSFit and serial processing remain unchanged. Exactness concerns
dilation of the same input mask, not equivalence of different extraction or
registration runs. Synthetic tests cover odd/even and oversized kernels, image
boundaries, numeric/noncontiguous inputs, singleton axes, anisotropic uint8
structures, early CUDA failures and CLI forwarding and batch-to-saved-mask integration. CUDA parity
tests skip when no GPU is available; skipped checks are not successful GPU tests.

For a public synthetic comparison, run from the `caideface/` directory with the
package installed:

```bash
python benchmarks/dilation.py --shape 96 96 64 --kernel 14 14 14 --repeats 5 --pattern ellipsoid --threads 2 --cuda
```

The `ellipsoid` pattern is a centred solid blob with radii one third of each
image dimension. The default `random` pattern sets each voxel with 1% probability.
Mask geometry affects runtime; neither synthetic pattern represents every scan.

Omit `--cuda` for CPU-only testing. The script reports exact dimensions, individual
call times and device preflight time (excluding torch import), checks equality outside the timer, and warms each method before
randomising timed call order. It does not reproduce the private six-mask table.

## Process sharing and CPU quotas

CUDA dilation creates a context in the parent process; HD-BET uses a separate
process. Clearing the allocator cache does not release that context. The combined
CUDA workflow requires a device that permits both processes, with sufficient
memory for both contexts. Exclusive-process compute mode is not validated or
supported by this workflow. Use `--device cpu` on such systems, or arrange a GPU
allocation that permits multiple processes. CPU-only dilation with GPU extraction
is not currently a separate CLI option. The local GPUs used for validation were
in Default mode; the earlier HD-BET CUDA-access failure has not been attributed
to exclusive mode. No compute-mode settings were changed for testing.

Set CPU thread limits explicitly in quota-limited containers, for example
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 caideface run ... --device cpu` with four
allocated CPU cores. In embedded Python, call `torch.set_num_threads(4)` before
imaging work. Choose the count for the actual allocation; PyTorch defaults may
reflect the host rather than the container quota. The library does not change
process-wide thread settings. The public benchmark accepts `--threads` and
records effective settings. Its reported timing outlier occurred with two threads
and a two-core quota, so it does not itself prove thread oversubscription.

Automatic CPU fallback is limited to recognised missing-kernel/build capability
errors. Out-of-memory, device-busy and unrecognised runtime errors propagate at
startup instead of moving extraction to CPU. Fully completed batch resumes skip
device probing and extraction-tool checks, following the existing output-existence
resume convention; this does not validate the contents of existing outputs.
