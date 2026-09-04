# CLAUDE.md

## What this is
Visual defect triage for a production line. A ViT backbone produces one
embedding per image. That embedding feeds three consumers, a linear classifier,
a FAISS retrieval index, and a drift monitor.

## Commands
```bash
pip install -r requirements.txt
python -m scripts.make_synthetic_data     # offline demo data
python -m src.run_pipeline                # full pipeline, writes artifacts/
python -m pytest tests/ -q
uvicorn api.main:app --port 8000
```

## Rules
- The backbone runs ONCE per image. If you add a consumer, it reads the cached
  embedding. Never add a second forward pass.
- Calibration is fitted on validation only. Fitting it on test is the bug that
  makes every threshold meaningless.
- Any change to transforms invalidates the FAISS index. Rebuild it.
- Mined review data may train. It must never enter the evaluation set.

## Layout
- `src/` pure logic, numpy only, no torch import at module level
- `src/models/`, `src/data/transforms.py` the only places torch is imported
- `api/` FastAPI, imports the model lazily so the module loads without torch
- `tests/` must pass with numpy alone; torch-dependent tests use importorskip
