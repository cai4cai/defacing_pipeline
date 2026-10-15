# caideface throughput PR preparation

- Upstream base: `81a360cff48b7171465e7826fd1139d96da80eb5`.
- Branch: `performance/dilation-and-batch-workers`; no PR opened or pushed.
- User requested benchmark-supported parallelism/multi-GPU options for a future PR.

Ported rectangular separable CPU/CUDA dilation from the tested downstream
prototype into the pipeline, skull-strip API and CLI with SciPy default retained.
Prepared an opt-in process scheduler with explicit GPU visibility, fresh isolated
per-volume outputs, bounded job deadlines and failure propagation. Added synthetic
parity and subprocess-scheduler tests. Container verification passed: 138 tests on the 5090 and 82 focused tests on
the 4090. Three real-data scheduler smoke modes each completed two volumes.
The downstream six-series benchmark found 1.37x throughput from two workers at
the same eight-CPU budget; shared-GPU and multi-GPU results were effectively
identical. Evidence and limits are in `caideface/THROUGHPUT.md`. The first smoke
launch lacked a required local image-wrapper environment variable; corrected
harness runs passed without numerical code changes. Upstream verification is
reported separately from downstream timings. Private images, runtime issues and review evidence
remain outside this checkout.
