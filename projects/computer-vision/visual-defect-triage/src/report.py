"""One markdown report per run, so two checkpoints can be diffed."""
from pathlib import Path


def write_report(run_dir, overall: float, by_class: list[dict], gate: dict,
                 ece: float, temperature: float) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Run {run_dir.name}",
        "",
        f"Overall accuracy {overall:.3f}",
        f"ECE {ece:.3f} at temperature {temperature:.2f}",
        "",
        "## By class, sorted by ceiling",
        "",
        "| class | n | share | accuracy | ceiling |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in by_class:
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['share']:.1%} "
            f"| {r['accuracy']:.3f} | {r['ceiling']:.4f} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        f"auto-accept {gate['auto_accept_share']:.1%}, "
        f"auto-reject {gate['auto_reject_share']:.1%}, "
        f"review {gate['review_share']:.1%}",
        "",
        f"escaped {gate['escaped_errors']} errors "
        f"({gate['escape_rate']:.2%} of accepted)",
        "",
    ]
    out = run_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_csv(path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    cols = list(rows[0])
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
