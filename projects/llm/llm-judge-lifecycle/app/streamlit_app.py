"""Four tabs, one per lifecycle phase.

    streamlit run app/streamlit_app.py

The tab order is the argument: a judge is born, tuned, deployed, and then
maintained, and the fourth tab is the one most teams never build. Every number
shown here carries its provenance, because a metric without its caveats is how
a class-balanced alignment score ends up in a slide as a production defect rate.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client as api

st.set_page_config(page_title="LLM Judge Lifecycle", page_icon="⚖️", layout="wide")


def show_provenance(payload: dict) -> None:
    """The caveats travel with the numbers, always.

    Two of these change what a result MEANS, and the person reading the screen
    is usually not the person who chose the config.
    """
    prov = payload.get("provenance", {})
    if prov.get("offline_stub_run"):
        st.warning(
            "**Offline run.** Every role is the deterministic rule engine. "
            "These numbers measure the rules written into the rubric, not any "
            "model's judgement. Use them as the baseline to beat.",
            icon="⚙️",
        )
    elif prov.get("single_model_config"):
        st.info(
            "**Generator and judge are the same model.** Self-preference bias "
            "is uncontrolled: the judge may be rewarding its own house style "
            "rather than quality. Point one role at another provider to "
            "measure the gap.",
            icon="🪞",
        )
    with st.expander("Provenance and cost", expanded=False):
        st.json(prov)


def metric_row(metrics: dict) -> None:
    """Point estimate and interval together. The interval is not an ornament -
    on this benchmark it is regularly wider than the differences people want to
    read as improvements."""
    columns = st.columns(4)
    for column, key, label in zip(
        columns,
        ("specificity", "recall", "reasoning_agreement"),
        ("Specificity (catches bad)", "Recall (keeps good)", "Reasoning agreement"),
    ):
        value = metrics.get(key)
        low, high = metrics.get("ci95", {}).get(key, [0, 0])
        column.metric(
            label,
            "n/a" if value is None else f"{value:.3f}",
            f"95% CI [{low:.2f}, {high:.2f}]",
            delta_color="off",
        )
    columns[3].metric("Weighted score", f"{metrics.get('weighted', 0):.3f}")


try:
    health = api.health()
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

st.title("The lifecycle of an LLM judge")
st.caption(
    f"domain: **{health['domain']}** · "
    "a judge is not a benchmark score, it is a system that has to be built, "
    "tuned, deployed and kept aligned"
)

CRITERIA = [c["id"] for c in health["criteria"]]
GATE = [c["id"] for c in health["criteria"] if c["must_have"]]

birth, training, deployment, monitoring = st.tabs(
    [
        "I · Birth",
        "II · Training (RART)",
        "III · Deployment",
        "IV · Monitoring",
    ]
)

# ---------------------------------------------------------------- Phase I
with birth:
    st.subheader("The benchmark, and what it can and cannot tell you")
    st.markdown(
        "Held near 50/50 per criterion **on purpose**. Real defect rates are a "
        "few percent, so a naturally-sampled benchmark would be ~95% PASS and a "
        "judge answering PASS to everything would score 95% while catching "
        "nothing. The cost of balancing is that **nothing here estimates the "
        "live defect rate** - that number comes from Phase IV."
    )
    if st.button("Load benchmark report", key="bench"):
        payload = api.benchmark_report()
        rows = payload["data"]["splits"]["criteria"]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "criterion": cid,
                        "gate": row["must_have"],
                        "n": row["n"],
                        "fail fraction": row["fail_fraction"],
                        "train": row["splits"]["train"],
                        "validation": row["splits"]["validation"],
                        "test": row["splits"]["test"],
                    }
                    for cid, row in rows.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        show_provenance(payload)

    st.divider()
    st.markdown("**Score a rubric on one split**")
    columns = st.columns([2, 2, 1])
    criterion = columns[0].selectbox("criterion", CRITERIA, key="eval_c")
    split = columns[1].selectbox("split", ["train", "validation", "test"], key="eval_s")
    if columns[2].button("Evaluate", key="eval_go"):
        payload = api.evaluate(criterion, split)
        data = payload["data"]
        metric_row(data["metrics"])
        st.caption(
            "Test splits here hold five or six examples. A perfect score over "
            "three of them is not evidence of a perfect judge, which is what "
            "the intervals are saying."
        )
        left, right = st.columns(2)
        left.markdown("**False passes** - bad work that would reach users")
        left.write(data["false_passes"] or "none")
        right.markdown("**False fails** - good work that would be regenerated")
        right.write(data["false_fails"] or "none")
        show_provenance(payload)

# --------------------------------------------------------------- Phase II
with training:
    st.subheader("Reasoning-Aligned Rubric Tuning")
    st.markdown(
        "No gradients and no fine-tuning. **The rubric text is the parameter** "
        "and a reflector model is the optimiser: score the rubric, collect the "
        "errors, ask for a better rubric, keep it only if validation improves."
    )
    columns = st.columns([2, 2, 1])
    criterion = columns[0].selectbox("criterion", GATE, key="tune_c")
    aligned = columns[1].checkbox(
        "reasoning alignment",
        value=True,
        help="Off reproduces the paper's vanilla arm: the reflector sees only "
        "label mismatches, so a right-verdict-wrong-reason case never reaches "
        "it.",
    )
    if columns[2].button("Tune", key="tune_go"):
        with st.spinner("running RART..."):
            payload = api.tune(criterion, aligned)
        data = payload["data"]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "iteration": it["index"],
                        "weighted": it["weighted"],
                        "specificity": it["validation"]["specificity"],
                        "recall": it["validation"]["recall"],
                        "reasoning": it["validation"]["reasoning_agreement"],
                        "focus set": it["focus_size"],
                        "kept": "*" if it["accepted"] else "",
                    }
                    for it in data["iterations"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"stopped: {data['stopped_because']}")
        if data["improved_on_seed"]:
            st.success(
                f"Best rubric came from iteration {data['best_iteration']}. It is "
                "**staged**, not live - promotion is a separate, deliberate step."
            )
        else:
            st.info(
                "No improvement on the seed rubric. Read the specificity before "
                "calling that a success: high means the human guideline was "
                "already at ceiling, low means the optimiser could not reach the "
                "headroom that is there. Both are results worth reporting."
            )
        with st.expander("The tuned rubric"):
            st.code(data["best_rubric"], language="markdown")
        show_provenance(payload)

# -------------------------------------------------------------- Phase III
with deployment:
    st.subheader("Gate and critic, in one judge")
    st.markdown(
        "`generate → judge → revise`, bounded. The judge **rejects** on a "
        "must-have failure, and its reason becomes the writer's instruction for "
        "the next attempt. When the budget runs out the artefact is **dropped**: "
        "a bad one reaching a user cannot be recalled, a missing one costs an "
        "opportunity."
    )
    if st.button("Pass rate vs retry budget", key="curve"):
        with st.spinner("serving the catalogue..."):
            payload = api.retry_curve(max_k=6)
        frame = pd.DataFrame(payload["data"]["curve"]).set_index("k")
        st.line_chart(frame[["cumulative_pass_rate"]])
        st.caption(payload["data"]["reading_guide"])
        show_provenance(payload)

# --------------------------------------------------------------- Phase IV
with monitoring:
    st.subheader("The drift band")
    st.markdown(
        "`judge >= mean(raters) - 2 * sd(raters)`. The threshold **floats**: on a "
        "week the raters found hard, sd widens and so does the band, so the "
        "judge is not punished for finding hard what people also found hard. A "
        "fixed threshold fires every hard week until somebody mutes it."
    )
    weeks = api.monitoring_weeks()["data"]["weeks"]
    if not weeks:
        st.info("No rated weeks found under domains/<domain>/hitl/.")
    else:
        week = st.selectbox("week", weeks, index=len(weeks) - 1)
        if st.button("Check the band", key="drift"):
            payload = api.monitoring_check(week)
            data = payload["data"]
            for report in data["reports"]:
                st.markdown(f"**criterion: {report['criterion']}**")
                rows = [
                    {"scope": scope, **{k: v for k, v in check.items()}}
                    for scope, checks in (
                        ("overall", report["overall"]),
                        ("new items", report["new_items"]),
                    )
                    for check in checks
                ]
                st.dataframe(
                    pd.DataFrame(rows), use_container_width=True, hide_index=True
                )
                for note in report["notes"]:
                    st.warning(note)
            if data["alert"]:
                st.error(f"DRIFT ALERT - {data['action']['note']}")
                for command in data["action"].get("commands", []):
                    st.code(command, language="bash")
            else:
                st.success(data["action"]["note"])
            show_provenance(payload)
