# Training monitoring notes

Scope: monitor the authorized 65-epoch XTrack-B author AdamW run on 2028,
including progress, numerical health, process liveness, and final checkpoint.
Startup and finite short-run evidence do not prove full training completion.

## Loss interpretation

The author's `lib/train/actors/xtrack.py` uses:

`total = 2 * giou + 5 * l1 + location + balance + 0.01 * classification`.

Classification is summed across the backbone's routing outputs. The displayed
classification value around 17.3 contributes about 0.173 to total loss.
For the observed epoch-1 step-835 row:

`2 * 0.27009 + 5 * 0.02663 + 0.44894 + 0.07966 + 0.01 * 17.30622 = 1.37500`,

consistent with logged total 1.37498 after component rounding. Thus this row's
component scale and loss weighting agree with the unmodified author actor.
No training loss or model behavior was changed during monitoring.

Source observation for later controlled work: `vit_ce_adapter.py` appends
`logits_attn` twice to `logits_prompt` (lines 218 and 219). The actor therefore
receives duplicated attention routing logits in the original implementation.
This source behavior is retained in the running author baseline; changing it
would require a separate paired experiment and must not be mixed into a claim
about the proposed optimizer's gains.

## Completion evidence still required

- Actual launcher termination with exit code zero.
- A loadable epoch-65 checkpoint from this run, with finite model tensors.
- 65 finite epoch training-loss entries and the expected optimizer-step count
  (2,500 updates per epoch; 162,500 total, checked on parameters used every step).
- 13 finite validation-loss entries, matching the configured validation interval
  of five epochs across the 65-epoch run.
- The final run receipt and final epoch logs agree with the saved checkpoint.

The persistent monitor checks exit status, checkpoint epoch, finite weights,
and training loss history. The final manual audit must also inspect optimizer
state and final epoch logs. Do not mark the monitoring goal complete before
these terminal artifacts exist and pass inspection.
