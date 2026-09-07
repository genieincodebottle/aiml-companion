"""End-to-end demo. Split, train, calibrate, index, evaluate, route, report.

Runs on the synthetic embeddings from scripts/make_synthetic_data.py, so it needs
no GPU, no downloads and no real images. Every number in artifacts/ comes from
this script.
"""
import json
from pathlib import Path

import numpy as np

from src.calibrate import fit_temperature
from src.config import settings
from src.data.splits import assert_no_batch_leak, split_by_batch
from src.gate import route
from src.metrics.calibration import expected_calibration_error, reliability_table
from src.metrics.gate_sim import simulate, sweep
from src.metrics.slices import ceilings_sum_to_error_budget, slice_report
from src.report import write_csv, write_report
from src.retrieval.index import build
from src.retrieval.service import RetrievalService
from src.schemas import CLASSES, Route
from src.train import train

LABEL_BUDGET = 3000

DATA = Path("data")
ART = Path("artifacts")
RUN = ART / "run"


def main() -> None:
    rows = [json.loads(l) for l in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    embeddings = np.load(DATA / "embeddings.npy")
    labels = np.load(DATA / "labels.npy")
    by_id = {r["image_id"]: i for i, r in enumerate(rows)}

    # 1. Split by BATCH, never by image, and prove it.
    splits = split_by_batch(rows, seed=13)
    assert_no_batch_leak(splits)
    idx = {k: np.array([by_id[r["image_id"]] for r in v]) for k, v in splits.items()}
    print(f"split: train {len(idx['train'])}, val {len(idx['val'])}, test {len(idx['test'])}")

    # 2. Only part of the train split is labelled. Labels are the budget, so the
    #    head is fitted on LABEL_BUDGET examples and the rest stay unlabelled
    #    production traffic that the index and the drift monitor still use.
    labelled = idx["train"][:LABEL_BUDGET]
    print(f"labelling budget: {len(labelled)} of {len(idx['train'])} train images carry a label")
    head = train(embeddings[labelled], labels[labelled], len(CLASSES))

    # 3. Fit the temperature on VALIDATION, never on test.
    val_logits = head.forward(embeddings[idx["val"]])
    temperature = fit_temperature(val_logits, labels[idx["val"]])
    print(f"fitted temperature: {temperature:.3f}")

    # 4. Evaluate on test, before and after calibration.
    test_logits = head.forward(embeddings[idx["test"]])
    test_labels = labels[idx["test"]]

    raw = head.probabilities(embeddings[idx["test"]], 1.0)
    cal = head.probabilities(embeddings[idx["test"]], temperature)
    pred = np.argmax(cal, axis=1)
    correct = pred == test_labels
    conf = cal[np.arange(len(pred)), pred]

    overall = float(correct.mean())
    ece_raw = expected_calibration_error(raw[np.arange(len(pred)), np.argmax(raw, axis=1)], correct)
    ece_cal = expected_calibration_error(conf, correct)
    print(f"accuracy {overall:.3f} | ECE {ece_raw:.3f} -> {ece_cal:.3f}")

    # Temperature scaling must not change a single prediction. Assert it.
    assert (np.argmax(raw, axis=1) == pred).all(), "temperature changed a prediction"

    # 5. Slices, ranked by ceiling rather than by error rate.
    records = [{"correct": bool(c), "class": CLASSES[l].value,
                "line": rows[i]["line_id"], "shift": rows[i]["shift"]}
               for c, l, i in zip(correct, test_labels, idx["test"])]
    by_class = slice_report(records, "class")
    assert ceilings_sum_to_error_budget(by_class, overall), "ceilings must sum to the error budget"

    # 6. The gate.
    gate = simulate(conf, correct, settings.accept_above, settings.reject_below)
    routes = [route(CLASSES[p], float(c)) for p, c in zip(pred, conf)]
    reviewed = sum(1 for r in routes if r is Route.REVIEW)
    print(f"gate: accept {gate['auto_accept_share']:.1%}, review {gate['review_share']:.1%}, "
          f"escaped {gate['escaped_errors']}")
    print(f"policy routes to review: {reviewed / len(routes):.1%} "
          f"(higher than the gate share, because two classes never auto-accept)")

    # 7. Retrieval over the same vectors, no second forward pass.
    index, ids = build(embeddings, [r["image_id"] for r in rows])
    meta = {r["image_id"]: {"final_ruling": r["label"], "batch_id": r["batch_id"]} for r in rows}
    service = RetrievalService(index, ids, meta,
                               built_with="vitb16-augreg2-v3", version="vitb16-augreg2-v3")
    n = service.neighbours(embeddings[idx["test"]][0], k=settings.top_k)
    print(f"retrieval returns {len(n)} neighbours, top similarity {n[0].similarity:.3f}")

    # 8. Write everything down.
    RUN.mkdir(parents=True, exist_ok=True)
    head.save(RUN / "head.npz")
    (RUN / "manifest.json").write_text(
        json.dumps({"temperature": temperature, "embedding_version": "vitb16-augreg2-v3"}), encoding="utf-8")
    np.save(RUN / "index_vectors.npy", embeddings.astype("float32"))
    np.save(RUN / "index_ids.npy", np.array(ids))
    (RUN / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    write_report(RUN, overall, by_class, gate, ece_cal, temperature)
    write_csv(ART / "slice_report.csv", [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in by_class])
    write_csv(ART / "threshold_sweep.csv", [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        for r in sweep(conf, correct)])
    write_csv(ART / "reliability.csv", reliability_table(conf, correct))
    (ART / "report.md").write_text((RUN / "report.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {ART}/report.md, slice_report.csv, threshold_sweep.csv, reliability.csv")


if __name__ == "__main__":
    main()
