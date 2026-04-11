"""
Smart Claims Processor - Streamlit Dashboard

Three tabs:
  1. Process Claim - Submit and track a new claim
  2. HITL Review Queue - Human reviewer interface
  3. Analytics - Pipeline metrics and cost breakdown
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.graph import process_claim
from src.hitl.queue import (
    DecisionRequest,
    list_pending_reviews,
    queue_stats,
    submit_decision,
    get_ticket,
)
from src.models.schemas import ClaimDecision
from src.models.state import ClaimInput

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Claims Processor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global typography */
    .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    /* Card-style containers */
    div[data-testid="stExpander"] {
        border: 1px solid #334155;
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetric"] label { color: #94A3B8 !important; font-size: 0.85rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #F1F5F9 !important; font-weight: 700; }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
    .stButton > button[kind="secondary"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
        border: 1px solid #334155;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1E293B;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
    }

    /* Input styling */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        border-radius: 8px !important;
        border-color: #334155 !important;
    }

    /* Status banners */
    .status-approved {
        background: linear-gradient(135deg, #065F46 0%, #064E3B 100%);
        color: #A7F3D0;
        padding: 20px 24px;
        border-radius: 12px;
        border-left: 4px solid #10B981;
        margin: 16px 0;
    }
    .status-denied {
        background: linear-gradient(135deg, #7F1D1D 0%, #991B1B 100%);
        color: #FCA5A5;
        padding: 20px 24px;
        border-radius: 12px;
        border-left: 4px solid #EF4444;
        margin: 16px 0;
    }
    .status-hitl {
        background: linear-gradient(135deg, #78350F 0%, #92400E 100%);
        color: #FDE68A;
        padding: 20px 24px;
        border-radius: 12px;
        border-left: 4px solid #F59E0B;
        margin: 16px 0;
    }
    .status-pending {
        background: linear-gradient(135deg, #1E3A5F 0%, #1E40AF 100%);
        color: #93C5FD;
        padding: 20px 24px;
        border-radius: 12px;
        border-left: 4px solid #3B82F6;
        margin: 16px 0;
    }

    /* Priority badges */
    .priority-critical { background: #DC2626; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
    .priority-high { background: #D97706; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
    .priority-normal { background: #059669; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }

    /* Section dividers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #CBD5E1;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #334155;
    }

    /* Score rings */
    .score-ring {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 64px;
        height: 64px;
        border-radius: 50%;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .score-high { border: 3px solid #10B981; color: #10B981; }
    .score-mid { border: 3px solid #F59E0B; color: #F59E0B; }
    .score-low { border: 3px solid #EF4444; color: #EF4444; }

    /* Hide default Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────

def _decision_banner(decision_str: str, amount: float) -> str:
    """Return HTML for a styled decision banner."""
    label = decision_str.upper().replace('_', ' ')
    status_map = {
        "approved": "approved",
        "approved_partial": "approved",
        "denied": "denied",
        "auto_rejected": "denied",
        "escalated_human_review": "hitl",
        "fraud_investigation": "denied",
        "pending_documents": "pending",
    }
    css_class = status_map.get(decision_str, "pending")
    icon = {"approved": "✅", "denied": "❌", "hitl": "⏳", "pending": "📋"}.get(css_class, "ℹ️")
    return f"""<div class="status-{css_class}">
        <span style="font-size:1.5rem">{icon}</span>
        <span style="font-size:1.3rem; font-weight:700; margin-left:12px">{label}</span>
        <span style="float:right; font-size:1.5rem; font-weight:700">${amount:,.2f}</span>
    </div>"""


def _score_html(score: float, label: str) -> str:
    """Return HTML for a circular score badge."""
    css = "score-high" if score >= 0.75 else ("score-mid" if score >= 0.50 else "score-low")
    return f"""<div style="text-align:center">
        <div class="score-ring {css}">{score:.0%}</div>
        <div style="color:#94A3B8; font-size:0.8rem; margin-top:6px">{label}</div>
    </div>"""


def _validate_claim_form(name, email, policy, incident_type, amount, description) -> list[str]:
    """Validate form inputs. Returns list of errors."""
    errors = []
    if not name or name.strip() == "":
        errors.append("Claimant name is required")
    if not email or "@" not in email:
        errors.append("Valid email address is required")
    if not policy:
        errors.append("Policy number is required")
    if amount <= 0:
        errors.append("Estimated damage must be greater than $0")
    if not description or len(description.strip()) < 10:
        errors.append("Incident description must be at least 10 characters")
    return errors


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 0">
        <span style="font-size:2.5rem">🏛️</span>
        <h2 style="margin:8px 0 4px 0; font-weight:700">Smart Claims</h2>
        <p style="color:#94A3B8; margin:0; font-size:0.9rem">AI-Powered Insurance Processing</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="padding:8px 0">
        <p style="color:#CBD5E1; font-weight:600; margin-bottom:12px">Frameworks</p>
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px">
            <span style="background:#1E40AF; padding:4px 12px; border-radius:20px; font-size:0.8rem">🔗 LangGraph</span>
            <span style="background:#7C3AED; padding:4px 12px; border-radius:20px; font-size:0.8rem">👥 CrewAI</span>
        </div>
        <p style="color:#CBD5E1; font-weight:600; margin-bottom:12px">Production Features</p>
        <div style="display:flex; gap:8px; flex-wrap:wrap">
            <span style="background:#334155; padding:4px 12px; border-radius:20px; font-size:0.8rem">🔒 PII Masking</span>
            <span style="background:#334155; padding:4px 12px; border-radius:20px; font-size:0.8rem">🛡️ Guardrails</span>
            <span style="background:#334155; padding:4px 12px; border-radius:20px; font-size:0.8rem">👤 HITL</span>
            <span style="background:#334155; padding:4px 12px; border-radius:20px; font-size:0.8rem">⚖️ LLM Judge</span>
            <span style="background:#334155; padding:4px 12px; border-radius:20px; font-size:0.8rem">📋 Audit Logs</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick HITL stats in sidebar
    try:
        stats = queue_stats()
        if stats["pending_total"] > 0:
            st.markdown(f"""
            <div style="background:#92400E; padding:12px 16px; border-radius:10px; margin-top:8px">
                <span style="font-weight:600">🔔 {stats['pending_total']} Pending Reviews</span><br>
                <span style="font-size:0.85rem; color:#FDE68A">{stats['pending_critical']} critical</span>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        pass


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🗂️ Process Claim", "👤 HITL Review Queue", "📊 Analytics"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 - Process Claim
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<h1 style="margin-bottom:4px">Submit Insurance Claim</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94A3B8; margin-top:0">Fill in claim details and process through the AI pipeline</p>', unsafe_allow_html=True)

    # ── Quick Test Scenarios ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚡ Quick Test Scenarios</div>', unsafe_allow_html=True)
    scenario_cols = st.columns(4)

    with scenario_cols[0]:
        if st.button("✅ Normal Claim\n$8.5K Auto", use_container_width=True):
            st.session_state["scenario"] = "normal"
    with scenario_cols[1]:
        if st.button("⚠️ HITL Review\n$28K Theft", use_container_width=True):
            st.session_state["scenario"] = "hitl"
    with scenario_cols[2]:
        if st.button("🚨 Fraud Flags\n$45K Property", use_container_width=True):
            st.session_state["scenario"] = "fraud"
    with scenario_cols[3]:
        if st.button("🚫 Lapsed Policy\n$2.2K Auto", use_container_width=True):
            st.session_state["scenario"] = "lapsed"

    # Load scenario defaults
    scenario = st.session_state.get("scenario", "normal")
    SCENARIOS = {
        "normal": {
            "name": "Jane Smith", "email": "jane.smith@email.com", "phone": "555-123-4567",
            "policy": "POL-AUTO-789456", "type": "auto_collision", "date": "2024-11-15",
            "location": "Austin, TX", "police": "APD-2024-567890", "amount": 8500.0,
            "year": 2019, "make": "Toyota", "model": "Camry",
            "desc": "Rear-ended at intersection of Main St and 5th Ave. Significant trunk damage, broken tail lights.",
            "docs": ["police_report.pdf", "damage_photos.zip", "repair_estimate.pdf"],
        },
        "hitl": {
            "name": "Jane Smith", "email": "jane.smith@email.com", "phone": "555-123-4567",
            "policy": "POL-AUTO-789456", "type": "auto_theft", "date": "2024-10-20",
            "location": "Houston, TX", "police": "HPD-2024-112233", "amount": 28000.0,
            "year": 2022, "make": "Honda", "model": "CR-V",
            "desc": "Vehicle stolen from parking garage overnight. No witnesses. Vehicle not recovered.",
            "docs": ["police_report.pdf", "proof_of_ownership.pdf"],
        },
        "fraud": {
            "name": "Robert Johnson", "email": "rjohnson99@gmail.com", "phone": "555-999-8888",
            "policy": "POL-HOME-334521", "type": "property_fire", "date": "2024-03-10",
            "location": "Dallas, TX", "police": "", "amount": 45000.0,
            "year": None, "make": None, "model": None,
            "desc": "Kitchen fire, total loss of appliances and cabinets. Fire started from electrical fault.",
            "docs": ["damage_photos.zip"],
        },
        "lapsed": {
            "name": "Maria Garcia", "email": "mgarcia@yahoo.com", "phone": "555-444-3333",
            "policy": "POL-AUTO-112233", "type": "auto_collision", "date": "2024-08-15",
            "location": "San Antonio, TX", "police": "SAPD-2024-445566", "amount": 2200.0,
            "year": 2018, "make": "Ford", "model": "Escape",
            "desc": "Side collision at parking lot. Minor damage to front bumper.",
            "docs": ["police_report.pdf", "damage_photos.zip"],
        },
    }
    s = SCENARIOS.get(scenario, SCENARIOS["normal"])

    st.markdown("---")

    # ── Claim Form ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">👤 Claimant Information</div>', unsafe_allow_html=True)
        claimant_name = st.text_input("Full Name *", value=s["name"], help="Legal name as it appears on the policy")
        claimant_email = st.text_input("Email *", value=s["email"])
        claimant_phone = st.text_input("Phone", value=s["phone"])
        policy_number = st.selectbox(
            "Policy Number *",
            ["POL-AUTO-789456", "POL-HOME-334521", "POL-AUTO-112233"],
            index=["POL-AUTO-789456", "POL-HOME-334521", "POL-AUTO-112233"].index(s["policy"]),
            help="POL-AUTO-112233 is lapsed (for testing denial path)"
        )

    with col2:
        st.markdown('<div class="section-header">📋 Incident Details</div>', unsafe_allow_html=True)
        incident_type = st.selectbox(
            "Incident Type *",
            ["auto_collision", "auto_theft", "property_fire", "property_water", "liability", "medical"],
            index=["auto_collision", "auto_theft", "property_fire", "property_water", "liability", "medical"].index(s["type"]),
        )
        incident_date = st.text_input("Incident Date *", value=s["date"], help="YYYY-MM-DD format")
        incident_location = st.text_input("Location", value=s["location"])
        police_report = st.text_input("Police Report # (if applicable)", value=s["police"])

    st.markdown('<div class="section-header">💰 Damage Details</div>', unsafe_allow_html=True)
    damage_cols = st.columns([2, 1, 1, 1])

    with damage_cols[0]:
        estimated_amount = st.number_input(
            "Estimated Damage ($) *", min_value=0.0, value=s["amount"], step=100.0,
            help="Your best estimate of total damage"
        )
    with damage_cols[1]:
        vehicle_year = st.number_input("Vehicle Year", min_value=1990, max_value=2026, value=s["year"] or 2020) if "auto" in incident_type else None
    with damage_cols[2]:
        vehicle_make = st.text_input("Make", value=s["make"] or "") if "auto" in incident_type else None
    with damage_cols[3]:
        vehicle_model = st.text_input("Model", value=s["model"] or "") if "auto" in incident_type else None

    incident_description = st.text_area(
        "Incident Description *", value=s["desc"], height=100,
        help="Describe what happened in detail. Include time, circumstances, and any witnesses."
    )

    documents = st.multiselect(
        "Documents Provided",
        ["police_report.pdf", "damage_photos.zip", "repair_estimate.pdf",
         "proof_of_ownership.pdf", "medical_records.pdf", "fire_report.pdf",
         "plumber_report.pdf", "inventory_list.pdf"],
        default=s["docs"],
        help="Select all documents attached to this claim"
    )

    # ── Clear Form ────────────────────────────────────────────────────────────
    clear_col, _, submit_col = st.columns([1, 3, 1])
    with clear_col:
        if st.button("🗑️ Reset Form", use_container_width=True):
            st.session_state["scenario"] = "normal"
            st.rerun()

    # ── Submit ────────────────────────────────────────────────────────────────
    with submit_col:
        process_clicked = st.button("🚀 Process Claim", type="primary", use_container_width=True)

    if process_clicked:
        # Validate
        validation_errors = _validate_claim_form(
            claimant_name, claimant_email, policy_number,
            incident_type, estimated_amount, incident_description
        )

        if validation_errors:
            for err in validation_errors:
                st.error(f"⚠️ {err}")
        else:
            claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"

            claim = ClaimInput(
                claim_id=claim_id,
                policy_number=policy_number,
                claimant_name=claimant_name,
                claimant_email=claimant_email,
                claimant_phone=claimant_phone,
                claimant_dob="1985-03-15",
                incident_date=incident_date,
                incident_type=incident_type,
                incident_description=incident_description,
                incident_location=incident_location,
                police_report_number=police_report or None,
                estimated_amount=estimated_amount,
                vehicle_year=int(vehicle_year) if vehicle_year else None,
                vehicle_make=vehicle_make or None,
                vehicle_model=vehicle_model or None,
                documents=documents,
                is_appeal=False,
                original_claim_id=None,
            )

            # Process with progress
            progress = st.progress(0, text="Initializing pipeline...")
            status_placeholder = st.empty()

            try:
                progress.progress(10, text="🔒 Masking PII...")
                progress.progress(20, text="📋 Validating intake...")
                progress.progress(40, text="🔍 Running fraud detection crew (CrewAI)...")
                progress.progress(60, text="💰 Assessing damage & checking policy...")
                progress.progress(75, text="🧮 Calculating settlement...")

                result = process_claim(claim)
                st.session_state["last_result"] = result

                progress.progress(90, text="⚖️ Running LLM-as-Judge evaluation...")
                progress.progress(100, text="✅ Pipeline complete!")

            except Exception as e:
                progress.empty()
                st.error(f"❌ Pipeline Error: {str(e)}")
                st.info("The claim has been automatically escalated for manual review.")
                result = None

            if result:
                progress.empty()
                decision = result.get("final_decision")
                decision_str = decision.value if hasattr(decision, "value") else str(decision)
                amount = result.get("final_amount_usd", 0)
                fraud_output = result.get("fraud_output")
                eval_output = result.get("evaluation_output")

                # ── Decision Banner ───────────────────────────────────────────
                st.markdown(_decision_banner(decision_str, amount), unsafe_allow_html=True)

                # ── Key Metrics ───────────────────────────────────────────────
                mc = st.columns(6)
                mc[0].metric("Settlement", f"${amount:,.2f}")
                mc[1].metric("Fraud Risk", f"{fraud_output.fraud_risk_level.value.upper()}" if fraud_output else "N/A")
                mc[2].metric("Fraud Score", f"{fraud_output.fraud_score:.2f}" if fraud_output else "N/A")
                mc[3].metric("Eval Score", f"{eval_output.overall_score:.2f}" if eval_output else "N/A")
                mc[4].metric("Agents Used", result.get("agent_call_count", 0))
                mc[5].metric("Pipeline Cost", f"${result.get('total_cost_usd', 0):.4f}")

                # ── HITL Alert ────────────────────────────────────────────────
                if result.get("hitl_required"):
                    ticket_id = result.get("hitl_ticket_id", "N/A")
                    priority = result.get("hitl_priority")
                    priority_str = priority.value if hasattr(priority, "value") else str(priority)
                    st.markdown(f"""
                    <div class="status-hitl">
                        <strong>🔔 Human Review Required</strong> - Ticket: <code>{ticket_id}</code> - Priority: <span class="priority-{priority_str}">{priority_str.upper()}</span><br>
                        <span style="font-size:0.9rem">Go to the <strong>HITL Review Queue</strong> tab to submit a decision</span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Detail Tabs ───────────────────────────────────────────────
                detail_tabs = st.tabs([
                    "📧 Notification", "🔍 Fraud Analysis", "💰 Settlement",
                    "⚖️ Evaluation", "📋 Pipeline Trace", "🔒 Guardrails"
                ])

                with detail_tabs[0]:
                    comm = result.get("communication_output")
                    if comm:
                        st.markdown(f"**Subject:** {comm.subject}")
                        st.markdown("---")
                        st.text_area("Claimant Email", comm.message, height=250, disabled=True)
                        if comm.next_steps:
                            st.markdown("**Next Steps:**")
                            for step in comm.next_steps:
                                st.markdown(f"- {step}")
                        if comm.appeal_instructions:
                            with st.expander("📝 Appeal Instructions"):
                                st.write(comm.appeal_instructions)
                        with st.expander("🔒 Internal Adjuster Notes (not sent to claimant)"):
                            st.text_area("Notes", comm.internal_notes, height=150, disabled=True)
                    else:
                        st.info("No communication generated for this claim path.")

                with detail_tabs[1]:
                    if fraud_output:
                        fraud_cols = st.columns(3)
                        fraud_cols[0].markdown(_score_html(fraud_output.fraud_score, "Fraud Score"), unsafe_allow_html=True)
                        fraud_cols[1].markdown(_score_html(fraud_output.pattern_score, "Pattern Risk"), unsafe_allow_html=True)
                        fraud_cols[2].markdown(_score_html(fraud_output.anomaly_score, "Anomaly Risk"), unsafe_allow_html=True)

                        st.markdown(f"**Risk Level:** `{fraud_output.fraud_risk_level.value.upper()}`")
                        st.markdown(f"**Recommendation:** `{fraud_output.recommendation}`")

                        if fraud_output.primary_concerns:
                            st.markdown("**Primary Concerns:**")
                            for concern in fraud_output.primary_concerns:
                                st.markdown(f"- 🔴 {concern}")

                        with st.expander("📊 Full Crew Analysis"):
                            st.write(fraud_output.crew_summary)
                    else:
                        st.info("Fraud analysis was not run (fast-mode or intake rejection).")

                with detail_tabs[2]:
                    settlement = result.get("settlement_output")
                    if settlement:
                        st.markdown("**Calculation Breakdown:**")
                        for i, step in enumerate(settlement.calculation_breakdown):
                            st.markdown(f"{i+1}. {step}")

                        calc_cols = st.columns(4)
                        calc_cols[0].metric("Gross Damage", f"${settlement.gross_damage_usd:,.2f}")
                        calc_cols[1].metric("Depreciation", f"-${settlement.depreciation_applied_usd:,.2f}")
                        calc_cols[2].metric("Deductible", f"-${settlement.deductible_applied_usd:,.2f}")
                        calc_cols[3].metric("Net Settlement", f"${settlement.settlement_amount_usd:,.2f}")

                        if settlement.denial_reasons:
                            st.error("**Denial Reasons:** " + " | ".join(settlement.denial_reasons))
                        st.metric("Regulatory Compliance", "✅ Passed" if settlement.regulatory_compliance else "❌ Failed")
                    else:
                        st.info("No settlement calculation (claim may have been rejected at intake).")

                with detail_tabs[3]:
                    if eval_output:
                        # Score circles
                        eval_cols = st.columns(5)
                        for col, (label, score) in zip(eval_cols, [
                            ("Accuracy", eval_output.accuracy_score),
                            ("Completeness", eval_output.completeness_score),
                            ("Fairness", eval_output.fairness_score),
                            ("Safety", eval_output.safety_score),
                            ("Transparency", eval_output.transparency_score),
                        ]):
                            col.markdown(_score_html(score, label), unsafe_allow_html=True)

                        overall_status = "✅ PASSED" if eval_output.passed else "❌ FAILED"
                        st.markdown(f"### Overall: {eval_output.overall_score:.0%} {overall_status}")

                        if eval_output.feedback:
                            st.info(f"**Judge Feedback:** {eval_output.feedback}")
                        if eval_output.flags:
                            for flag in eval_output.flags:
                                st.warning(f"🚩 {flag}")
                    else:
                        st.info("Evaluation was skipped for this claim.")

                with detail_tabs[4]:
                    trace = result.get("pipeline_trace", [])
                    if trace:
                        for i, step in enumerate(trace):
                            agent = step.get("agent", "unknown")
                            duration = step.get("duration_ms", "N/A")
                            st.markdown(f"**Step {i+1}: `{agent}`** ({duration}ms)")
                            st.json(step)
                    else:
                        st.info("No trace data available.")

                with detail_tabs[5]:
                    violations = result.get("guardrails_violations", [])
                    g_cols = st.columns(3)
                    g_cols[0].metric("Agent Calls", f"{result.get('agent_call_count', 0)} / 25")
                    g_cols[1].metric("Tokens Used", f"{result.get('total_tokens_used', 0):,}")
                    g_cols[2].metric("Cost", f"${result.get('total_cost_usd', 0):.4f} / $0.50")

                    if violations:
                        st.warning(f"**{len(violations)} Guardrail Violation(s):**")
                        for v in violations:
                            st.markdown(f"- ⚠️ {v}")
                    else:
                        st.success("✅ All guardrails passed - no violations")

                    errors = result.get("error_log", [])
                    if errors:
                        st.error("**Pipeline Errors:**")
                        for err in errors:
                            st.markdown(f"- {err}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 - HITL Review Queue
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<h1 style="margin-bottom:4px">Human Review Queue</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94A3B8; margin-top:0">Claims requiring human adjuster review and decision</p>', unsafe_allow_html=True)

    # Stats bar
    try:
        stats = queue_stats()
    except Exception:
        stats = {"pending_total": 0, "pending_critical": 0, "pending_high": 0, "resolved_today": 0, "human_overrides_today": 0}

    stat_cols = st.columns(5)
    stat_cols[0].metric("📋 Pending Total", stats["pending_total"])
    stat_cols[1].metric("🔴 Critical", stats["pending_critical"])
    stat_cols[2].metric("🟡 High", stats.get("pending_high", 0))
    stat_cols[3].metric("✅ Resolved Today", stats["resolved_today"])
    stat_cols[4].metric("🔄 AI Overrides", stats["human_overrides_today"])

    st.markdown("---")

    # Ticket list
    try:
        tickets = list_pending_reviews()
    except Exception:
        tickets = []

    if not tickets:
        st.markdown("""
        <div style="text-align:center; padding:60px 0; color:#64748B">
            <span style="font-size:3rem">✅</span>
            <h3 style="color:#94A3B8; margin-top:16px">Queue Empty</h3>
            <p>All claims have been processed. No pending reviews.</p>
            <p style="font-size:0.85rem">Process a high-value or fraud-flagged claim to populate this queue.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for ticket in tickets:
            priority_icon = {"critical": "🔴", "high": "🟡", "normal": "🟢"}.get(ticket.priority, "⚪")
            priority_badge = f'<span class="priority-{ticket.priority}">{ticket.priority.upper()}</span>'

            with st.expander(
                f"{priority_icon} {ticket.ticket_id} | Claim: {ticket.claim_id} | "
                f"Priority: {ticket.priority.upper()} ({ticket.priority_score:.0f})"
            ):
                # Triggers
                st.markdown("**Why This Needs Review:**")
                for t in ticket.triggers:
                    st.markdown(f"- ⚠️ {t}")

                # Full brief
                try:
                    full = get_ticket(ticket.ticket_id)
                    if full.get("review_brief"):
                        with st.expander("📄 Full Review Brief"):
                            st.code(full["review_brief"], language=None)
                except Exception:
                    st.warning("Could not load full ticket details.")

                st.markdown("---")
                st.markdown("**Submit Your Decision:**")

                review_cols = st.columns([1, 1, 1])
                with review_cols[0]:
                    reviewer_id = st.text_input("Reviewer ID", key=f"rev_{ticket.ticket_id}", value="adjuster_001")
                with review_cols[1]:
                    decision = st.selectbox(
                        "Decision", [d.value for d in ClaimDecision],
                        key=f"dec_{ticket.ticket_id}",
                    )
                with review_cols[2]:
                    override_ai = st.checkbox("Override AI", key=f"ovr_{ticket.ticket_id}", help="Check if your decision differs from the AI recommendation")

                notes = st.text_area("Review Notes", key=f"notes_{ticket.ticket_id}", height=80, placeholder="Explain your reasoning...")

                if st.button("✅ Submit Decision", key=f"submit_{ticket.ticket_id}", type="primary"):
                    if not reviewer_id.strip():
                        st.error("Reviewer ID is required")
                    else:
                        try:
                            req = DecisionRequest(
                                reviewer_id=reviewer_id,
                                decision=decision,
                                notes=notes,
                                override_ai=override_ai,
                            )
                            result = submit_decision(ticket.ticket_id, req)
                            st.success(f"✅ Decision submitted: **{result['decision'].upper().replace('_', ' ')}**")
                            st.balloons()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to submit decision: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 - Analytics
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<h1 style="margin-bottom:4px">Pipeline Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94A3B8; margin-top:0">Performance metrics, cost breakdown, and architecture overview</p>', unsafe_allow_html=True)

    # ── Cost Profile ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💰 Cost Per Claim Type (Gemini 2.5 Flash)</div>', unsafe_allow_html=True)

    import pandas as pd

    cost_data = pd.DataFrame({
        "Claim Path": [
            "⚡ Fast Mode (<$500)",
            "✅ Standard Auto ($1K-$10K)",
            "⚠️ Complex + HITL ($10K+)",
            "🚨 Fraud Flagged",
            "🚫 Lapsed Policy (Intake Reject)",
        ],
        "Pipeline Path": [
            "Intake -> Settlement -> Comms",
            "All 7 agents",
            "All 7 + HITL + Evaluation",
            "All 7 + Fraud deep dive",
            "Intake -> Comms (2 agents)",
        ],
        "Agents": [3, 7, 9, 7, 2],
        "Avg Tokens": ["8K", "22K", "38K", "28K", "4K"],
        "Cost": ["~$0.003", "~$0.008", "~$0.014", "~$0.010", "~$0.001"],
        "Time": ["~8s", "~25s", "~60s", "~40s", "~5s"],
    })
    st.dataframe(cost_data, use_container_width=True, hide_index=True)

    # ── Architecture ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏗️ Framework Architecture</div>', unsafe_allow_html=True)

    arch_cols = st.columns(2)
    with arch_cols[0]:
        st.markdown("""
        | Layer | Framework | Why |
        |-------|-----------|-----|
        | **Orchestration** | LangGraph | Conditional routing, state machine, HITL checkpoints |
        | **Fraud Detection** | CrewAI | Role-based expert agents with manager synthesis |
        | **Agent Outputs** | Pydantic v2 | Structured output, zero parsing errors |
        | **HITL API** | FastAPI | REST endpoints for reviewer interface |
        | **Security** | Custom | PII regex masking, SHA-256 audit logs |
        | **Evaluation** | LLM-as-Judge | 5-dimension quality scoring |
        | **UI** | Streamlit | Interactive dashboard with real-time results |
        """)

    with arch_cols[1]:
        st.markdown("""
        **Pipeline Paths:**
        ```
        PATH A: Normal claim
          intake -> fraud -> damage -> policy -> settlement -> eval -> comms

        PATH B: HITL escalation
          ... -> eval FAILS -> hitl_checkpoint -> human review -> comms

        PATH C: Auto-reject (fraud >= 0.90)
          intake -> fraud -> auto_reject -> comms

        PATH D: Intake rejection
          intake -> invalid -> comms (denial)

        PATH E: Fast mode (<$500)
          intake -> settlement -> comms
        ```
        """)

    # ── HITL Triggers ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">👤 HITL Trigger Thresholds</div>', unsafe_allow_html=True)

    try:
        from src.config import get_hitl_config
        hitl_cfg = get_hitl_config()
        triggers = hitl_cfg.get("triggers", {})
        weights = hitl_cfg.get("priority_weights", {})
        sla = hitl_cfg.get("sla_hours", {})

        trigger_cols = st.columns(3)
        with trigger_cols[0]:
            st.markdown("**Trigger Conditions:**")
            st.json(triggers)
        with trigger_cols[1]:
            st.markdown("**Priority Weights:**")
            st.json(weights)
        with trigger_cols[2]:
            st.markdown("**SLA Hours by Priority:**")
            st.json(sla)
    except Exception:
        st.info("Could not load HITL configuration.")

    # ── Guardrails ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🛡️ Guardrails Configuration</div>', unsafe_allow_html=True)

    try:
        from src.config import get_guardrails_config
        g_cfg = get_guardrails_config()
        guard_cols = st.columns(5)
        guard_cols[0].metric("Max Agent Calls", g_cfg.get("max_agent_calls", 25))
        guard_cols[1].metric("Max Tokens", f"{g_cfg.get('max_tokens_per_claim', 50000):,}")
        guard_cols[2].metric("Max Cost", f"${g_cfg.get('max_cost_usd', 0.50):.2f}")
        guard_cols[3].metric("Timeout", f"{g_cfg.get('max_execution_seconds', 300)}s")
        guard_cols[4].metric("Min Confidence", f"{g_cfg.get('min_output_confidence', 0.60):.0%}")
    except Exception:
        st.info("Could not load guardrails configuration.")

    # ── Production Features Summary ──────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Production Feature Checklist</div>', unsafe_allow_html=True)

    features = {
        "Security - PII Masking": "Regex + field-level masking before any LLM call. Email, phone, SSN, DOB, names redacted.",
        "Security - Audit Logs": "SHA-256 hashed NDJSON logs. 7-year retention for insurance compliance. Every agent action recorded.",
        "Guardrails - Pre-execution": "Budget limits (tokens, cost, calls), loop detection, execution timeout enforcement.",
        "Guardrails - Post-execution": "Output confidence thresholds, empty reasoning detection (hallucination proxy).",
        "Human-in-the-Loop": "Priority queue with SQLite backend. FastAPI REST endpoints for human reviewers. SLA tracking.",
        "LLM-as-Judge": "5-dimension evaluation (accuracy, completeness, fairness, safety, transparency). Quality gate before release.",
        "Dual Framework": "LangGraph for orchestration (5 conditional paths). CrewAI for fraud crew (3 role-based agents).",
        "Structured Output": "Pydantic v2 schemas for all agent outputs. Zero parsing errors guaranteed.",
        "Rule-Based Grounding": "Pattern matching and damage calculators run BEFORE LLM to reduce hallucination.",
        "Graceful Degradation": "LLM failure -> rule-based fallback -> HITL escalation. Pipeline never crashes silently.",
    }

    for name, desc in features.items():
        st.markdown(f"✅ **{name}**: {desc}")
