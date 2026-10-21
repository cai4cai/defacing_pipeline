# Scope throughput evidence to the existing method

Keep PR #29 focused on exact rectangular dilation and configurable process-isolated
workers. HD-BET and BRAINSFit remain the processing method. Move registration
candidate discussion into [issue #30](https://github.com/cai4cai/defacing_pipeline/issues/30),
with separate six-input iterative and 24-input pretrained comparisons, timing
scope and outstanding quality qualifications.

Updated `caideface/THROUGHPUT.md` with retained-method 2/4/8-worker NIfTI scaling
and fixed-mask Gaussian/box-convolution comparisons. Removed misleading ratios
between the original brain-to-mask interval and dilation-only timers; retained
the 300× theoretical neighbourhood-work calculation and matched 9.74× result.
PR description separates package NIfTI timings from downstream DICOM timings.
No project-specific locations, identifiers, images or record-level evidence added.

Documentation only; implementation and defaults unchanged. Cross-checked figures
against retained aggregate reports and the prior PR body. `git diff --check`
passed. Earlier 138/82-test verification is reported as historical; no GPU jobs
or software tests rerun for this documentation change. Registration and persistent
model-loading changes remain separate follow-ups. PR remains draft; no merge.
