# Runbook

## No alerts for 24 hours

This is first because it is the dangerous one. A safety system that has
failed and a site with no violations produce the same thing, which is
silence.

1. Check detections per hour, not alerts. Zero means the input broke. A
   site can genuinely have no violations for a day. It cannot have no
   people.
2. Check `last_frame_ts` per camera. A dead RTSP stream is silent, and
   the bounded queue in `src/capture.py` hides the backlog by design.
3. Run `scripts/test_walk.py`. If no alert arrives, the chain is broken
   somewhere between the lens and the notification.
4. Check the engine file hash against the deployed release. A checkpoint
   that shipped without a re-export means the device is running the old
   model while every dashboard describes the new one.

## Too many false alerts

1. Group dismissals by reason with `src.review.hard_negatives.triage`.
   The fix differs for each, and only one of the three is a model problem.
   - `wrong_detection`, retrain with the mined negatives
   - `outside_zone`, fix the polygon, do NOT retrain
   - `authorised`, fix the rule, do NOT retrain
2. Only after that, consider raising `conf_threshold`. Check
   `artifacts/threshold_sweep.csv` for what it costs in recall.

Retraining on `outside_zone` dismissals teaches the model to miss real
people in exactly the area you most need it to see them.

## Frame budget exceeded

1. Read the stage table from `src.budget.report()`.
2. Largest stage first. It is frequently not inference. NMS runs on the
   CPU and does not quantise, so it grows as a share of the frame every
   time the network gets faster.
3. Before optimising inference, check
   `src.budget.amdahl_ceiling()`. At a 53.7 per cent share the best
   possible whole-frame win is 2.16x, however fast the network becomes.

## A change to letterboxing

Re-export the engine and re-run `tests/test_export_parity.py`. The
padding value and the geometry are part of the model contract, and a
mismatch reads as a slightly worse model rather than as a bug.

## The confidence threshold appears to do nothing

Check `src/tracker.py`. Its high-confidence gate derives from
`settings.conf_threshold` for a reason: when it held its own constant,
the stricter of the two silently won and the whole sweep came out flat.
