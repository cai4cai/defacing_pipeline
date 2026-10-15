# GPU worker throughput comparison

Initial six-series research comparison, 2026-10-15. The same six frozen series
were processed in seven fresh configurations: 42 successful attempts, no failed
series or nonzero worker exits. Automated output comparison and upstream
verification are complete; successful processing alone is not visual review.

## End-to-end downstream throughput

| Total CPU / RAM budget | Configuration | Batch wall time | Series/hour | Speedup vs one 5090 worker at the same budget |
| --- | --- | ---: | ---: | ---: |
| 4 CPUs / 16 GiB | One 5090 worker | 665.42 s | 32.46 | 1.00x |
| 4 CPUs / 16 GiB | One 4090 worker | 672.56 s | 32.12 | 0.99x |
| 4 CPUs / 16 GiB | Two workers sharing the 5090 | 574.49 s | 37.60 | 1.16x |
| 4 CPUs / 16 GiB | One worker per GPU | 526.84 s | 41.00 | 1.26x |
| 8 CPUs / 32 GiB | One 5090 worker | 484.35 s | 44.60 | 1.00x |
| 8 CPUs / 32 GiB | Two workers sharing the 5090 | 353.66 s | 61.07 | 1.37x |
| 8 CPUs / 32 GiB | One worker per GPU | 353.49 s | 61.10 | 1.37x |

At the larger resource budget, one GPU with two workers and two GPUs with one
worker each were indistinguishable at the measurement resolution. Two shared
5090 workers achieved 1.88x the four-CPU single-worker throughput, but that
comparison also doubles the CPU/RAM budget. It is not a GPU-only speedup.

The practical starting configuration supported by this experiment is two
processes sharing one GPU, with four CPUs and 16 GiB RAM allocated per worker.
Multi-GPU assignment remains useful for allocation flexibility; it did not
improve the larger-budget result here. Registration remains CPU BRAINSFit.

## Method and limits

- Pinned downstream implementation: `e547cb136654d0ba29dea0bfa7613cd704bcf89f`.
- Imaging image: `sha256:1e26d601b798a61f0a5ee5f4481094652427a845d6b6205c93a19f1b060a2f4d`.
- CUDA HD-BET and CUDA rectangular dilation; frozen weights/reference, TTA off,
  14 mm dilation, unchanged registration and native geometry bounds.
- Offline, unprivileged, read-only containers; fresh processing/candidate roots.
  Reviewed outputs were preserved. GPU UUIDs and runtime roots remain private.
- Two-worker arms partition the frozen selection into disjoint queues, balanced
  using earlier per-series timings. Their union is the complete baseline set.
- CPU and RAM budgets are fixed within each group and split equally between
  workers. Each worker has a 1,024 process/thread limit, fresh-process GPU health
  checks and a 2,400-second outer deadline. No automatic retry was needed.
- Wall time runs from worker launch until all workers are observed complete. It
  includes startup, export and validation, and a small completion-polling delay.
  Throughput uses that full batch time, not summed overlapping worker timings.
- One measured batch per configuration: these are initial observations, not
  repeated-trial confidence estimates or a sustained full-cohort guarantee.
  Input/model caches were not deliberately flushed. The selection is small;
  queue balance, registration variability and background contention can matter.
- Sampled memory/utilisation evidence is retained privately. Samples can miss
  peaks or short GPU bursts; device totals include unrelated resident/display
  allocations. The 4090 had a small background allocation at the initial check.
- Both GPUs passed actual CUDA computation and 3D convolution checks. The focused
  downstream GPU/guard suite subsequently passed 38 tests on each GPU.

## Upstream contribution

A separate caideface branch based on
`81a360cff48b7171465e7826fd1139d96da80eb5` ports separable dilation into the package
and adds explicit process-isolated GPU batch execution through CLI and Python
APIs. It preserves the existing serial path and defaults. The batch option uses
fresh per-volume outputs, GPU visibility set before child Python starts,
per-job deadlines and failure propagation. Stage wall timings are also exposed.

The table above measures the downstream DICOM workflow, not that new upstream
scheduler. Its synthetic and real-data smoke verification is reported
separately below. Broader use on the user's cases remains the next validation step
before upstream PR submission. No PR has been published.

## Output comparison and upstream verification

Each arm compared all 1,107 DICOM instances against the previously accepted GPU
reference. Across all seven arms there were zero header/geometry differences,
zero mutually retained pixel differences, zero IOD validation failures, and zero
removed voxels inside that reference's dilated protection mask. There were no GPU
health-check failures. The reviewed CPU configuration checksum is unchanged.

All 5090-only arms reproduced the reference extraction foreground and dilated
protection masks exactly. The 4090-only arm differed at 218 extraction-foreground
voxels and 246 dilated-mask voxels; mixed-GPU arms differed at 96 foreground voxels
and 76 or 80 dilated-mask voxels. Final native keep masks varied in every arm,
including the fresh single-worker controls (aggregate differences ranged from
1,259,874 to 1,771,004 voxels). This is not full-pipeline voxel identity. These
checks use the accepted GPU reference and do not revise the earlier CPU/GPU
numerical findings or the user's acceptance of that reference. New benchmark
outputs retain their normal review status; no additional visual review is claimed.

For the recommended shared-5090 configuration at eight CPUs, sampled device memory
peaked at 5,615 MiB (5.48 GiB), and aggregate container memory at 9.07 GiB. These
are sampled totals, not allocator high-water marks or guaranteed capacity limits.

The upstream package passed 138 tests on the 5090; 82 focused morphology and
scheduler tests also passed on the 4090. Each invocation reported one existing
third-party deprecation warning. Tests cover exact dilation at odd/even kernels
and boundaries, device visibility, distinct outputs for repeated filenames,
worker failures, and timeout termination of child process groups.

A separate real-data smoke comparison exercised the new upstream scheduler on
two frozen NIfTI inputs, at eight CPUs and 32 GiB total:

| Upstream scheduler mode | Volumes completed | Launcher wall time |
| --- | ---: | ---: |
| One 5090 worker, eight threads | 2/2 | 127.97 s |
| Two 5090 workers, four threads each | 2/2 | 100.71 s |
| One worker per GPU, four threads each | 2/2 | 97.96 s |

These are small functional smoke timings for NIfTI processing, not the six-series
DICOM benchmark above. Both 5090 modes reproduced reference protection masks
exactly; the mixed-GPU smoke differed at 40 protection-mask voxels. All compared
mask and defaced-image grids matched. Defaced pixel arrays varied, as in the
repeat controls. This smoke check does not establish full-cohort output acceptance.

The first upstream smoke launch failed because the local imaging container's
HD-BET wrapper required `IMAGING_THREADS`. The harness was corrected and all three
modes rerun into fresh destinations; the failed evidence is preserved. No
numerical algorithm change was needed, and the workers correctly returned failure.
The first comparison-launch attempt also rejected overlapping input/output roots;
comparison reports were subsequently written to separate destinations. Neither
launcher correction altered or reran the timed downstream benchmark arms.

All task-owned containers exited. Detailed case-level evidence, failed-launch
receipts and the private operational issue log remain outside Git/GitHub.
