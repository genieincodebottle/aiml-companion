# CLAUDE.md

## What this is
Two site cameras at 15 fps into one edge device. YOLO detects PPE
violations, ByteTrack keeps identity between detections, zone logic
tests a polygon, a dwell timer suppresses flicker.

## The budget
2 cameras x 15 fps = 30 frames/s = 33.33 ms per frame. Everything is
measured against that number.

## Rules
- Detection runs on every SECOND frame. The tracker fills the gap. Do
  not "fix" a dropped frame by detecting on every frame.
- NMS runs on the CPU and does not quantise. Count it in the budget.
- Any change to letterboxing invalidates the exported engine. Re-export
  and re-run the parity test.

## Running without a GPU, a camera, or TensorRT
The whole repository runs offline on numpy alone. `src/sim/` supplies a
stand-in engine and a synthetic shift, so `make test` and `make demo`
need no model weights, no video, and no network.

- `cv2` and `shapely` are OPTIONAL. `src/preprocess.py` falls back to a
  numpy resize and `src/zones.py` uses numpy ray casting. Never import
  either at module level.
- The TensorRT path in `scripts/export.py` is real code that requires
  `trtexec`. It is exercised by a test against a recorded log rather
  than by running it.

## Two numbers that must stay in step
`src/budget.py` carries a measured stage table AND a modelled one. The
modelled table is what the README, the model card and the teaching
material quote. If you change a stage cost, `tests/test_budget.py`
fails, and that is deliberate.
