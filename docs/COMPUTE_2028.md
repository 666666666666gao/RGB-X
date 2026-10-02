# SSH 2028 environment ledger

### env: rgbx-risk@433a922a

how: existing conda prefix `/data/gb/conda/envs/rgbx-risk` on SSH alias 2028
spec: `configs/env-spec.json`, canonical sorted JSON SHA256 prefix 433a922a
tier: 3 RTX 3090 GPUs (24 GiB each), 4 CPU compute threads per rank, 5 loader workers per rank
project: `/data/gb/rgbx-risk`
weights: existing OSTrack initializer symlink; 370179249 bytes;
SHA256 `8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41`
validated: 2026-10-02; all three seeded CUDA witnesses and the documented real-data DDP invocation passed.
review: fresh gpt-6-astra reviewer, max effort; original WARN about missing explicit report fields,
then supplemental PASS after reviewing gradient assertions, validation logs, and checkpoint audit;
review_independence=same-family, acceptance_status=provisional.
gotcha: PyTorch 1.13.1 CUDA 11.7 is installed; do not replace it with the unrelated existing 1.12 environment.
gotcha: pandas 1.5.3 preserves the author's read_csv squeeze argument.
gotcha: LasHeR needs ordinal sorted frame paths; validation uses the existing XTrackProcessing class.
gotcha: torch.distributed.launch supplies the author's --local_rank argument; use the documented launcher.
gotcha: fail_safe=False is required for exceptions to produce a failed process exit.

Exact validation commands and expected results are in `docs/RUN_20261002.md`.
Do not launch formal training during the independent documented validation.
