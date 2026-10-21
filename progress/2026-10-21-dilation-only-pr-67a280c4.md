# Narrow PR #29 to exact rectangular dilation

Preserved the complete previous implementation at
`0992846eae7a5ccc64c35b473989105ca024b1e6` on
`performance/batch-workers-follow-up`, with a separate local checkout.
Batch scheduling is tracked in [issue #31](https://github.com/cai4cai/defacing_pipeline/issues/31);
registration alternatives remain in #30.

Removed batch APIs/CLI/options/tests, stage timers and unrelated CLI failure
handling from this PR. Kept exact separable CPU/CUDA dilation, SciPy defaults,
backend forwarding and dedicated tests. Replaced broad throughput documentation
with `caideface/DILATION.md`; earlier task notes remain on the preserved branch.
No changes to HD-BET, BRAINSFit, serial result fields or processing parameters.

Verification: full synthetic package suite in the retained offline container,
RTX 5090, two CPU quota / 6 GiB RAM: **126 passed**, one existing third-party
Click/spaCy deprecation warning. CUDA tests ran, none skipped. No patient inputs
mounted and no performance benchmark rerun under the existing GPU workload.
`git diff --check` passed. PR remains draft; no merge or bulk processing.
