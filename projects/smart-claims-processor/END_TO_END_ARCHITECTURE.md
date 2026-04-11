# End-to-End Architecture - Smart Claims Processor

## System Overview

The Smart Claims Processor is a **multi-agent insurance claims processing system** that combines two AI orchestration frameworks:

- **LangGraph** - State machine orchestration with 5 conditional routing paths
- **CrewAI** - Role-based fraud detection crew with 3 specialist agents

The system processes insurance claims through a pipeline of 7 specialized agents, with production-grade security (PII masking, audit logs), guardrails (budget enforcement, hallucination detection), human-in-the-loop (priority queue with REST API), and automated evaluation (LLM-as-judge quality scoring).

---

## Data Flow

```
[Claimant submits claim]
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SECURITY LAYER                                                          │
│  PIIMasker.mask_claim() -> replaces email, phone, SSN, DOB, names      │
│  Original claim stored in state.claim (never sent to LLM)              │
│  Masked version stored in state.masked_claim (used in all LLM calls)   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LANGGRAPH STATE MACHINE                                                 │
│                                                                         │
│  ClaimsState (TypedDict):                                               │
│    - claim: ClaimInput (raw)                                            │
│    - masked_claim: dict (PII-safe)                                      │
│    - intake_output: IntakeValidationOutput                              │
│    - fraud_output: FraudAssessmentOutput                                │
│    - damage_output: DamageAssessmentOutput                              │
│    - policy_output: PolicyCheckOutput                                   │
│    - settlement_output: SettlementOutput                                │
│    - evaluation_output: EvaluationOutput                                │
│    - communication_output: CommunicationOutput                          │
│    - hitl_*: HITL fields (triggers, priority, ticket, human decision)   │
│    - pipeline_trace: Annotated[list[dict], operator.add]  (append-only) │
│    - error_log: Annotated[list[str], operator.add]        (append-only) │
│    - guardrails_*: Budget tracking (calls, tokens, cost, violations)    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Details

### 1. Intake Agent (LangGraph Node)

**File:** `src/agents/intake_agent.py`

```
Input: Raw claim + policy number
Steps:
  1. PII mask the claim
  2. Look up policy in database (no LLM)
  3. Verify policy was active on incident date (no LLM)
  4. Check required documents (no LLM)
  5. Call LLM for nuanced eligibility assessment
Output: IntakeValidationOutput (is_valid, claim_type, confidence, flags)
```

**Key design:** Steps 1-4 are rule-based (free, instant). LLM is only called if basic checks pass, saving tokens on obviously invalid claims.

### 2. Fraud Detection Crew (CrewAI)

**File:** `src/agents/fraud_crew.py`

```
Framework: CrewAI (Process.sequential with manager synthesis)

Agent 1: Pattern Analyst
  Role: "Insurance Fraud Pattern Analyst"
  Tools: [check_fraud_patterns_tool]
  Max iterations: 3
  Backstory: 15-year veteran fraud investigator

Agent 2: Anomaly Detector
  Role: "Statistical Anomaly Detection Specialist"
  Tools: [anomaly_detection_tool, claim_baseline_tool]
  Max iterations: 3
  Backstory: PhD in actuarial science

Agent 3: Social Validator
  Role: "Claim Consistency Validator"
  Tools: [] (reasons from claim text only)
  Max iterations: 2
  Backstory: Former investigative journalist

Synthesis:
  1. Rule-based pattern check runs first (grounding)
  2. CrewAI agents run sequentially
  3. Composite score = pattern(0.40) + anomaly(0.35) + consistency(0.25)
  4. Classify: LOW (<0.35), MEDIUM (0.35-0.65), HIGH (0.65-0.90), CONFIRMED (>0.90)

Output: FraudAssessmentOutput (fraud_score, risk_level, primary_concerns)
```

### 3. Damage Assessor (LangGraph Node)

**File:** `src/agents/damage_assessor.py`

```
Pre-compute (no LLM):
  - Vehicle ACV (Actual Cash Value) via depreciation model
  - Total loss check (repair > 75% of ACV?)
  - Repair estimate range lookup

LLM:
  - Receives pre-computed data as grounding
  - Produces independent assessment (may differ from claimant's estimate)
  - Decides: repair | replace | total_loss

Output: DamageAssessmentOutput (assessed_damage_usd, line_items, confidence)
```

### 4. Policy Checker (LangGraph Node)

**File:** `src/agents/policy_checker.py`

```
Tool lookup:
  - Coverage type for this claim
  - Coverage limit
  - Deductible
  - Exclusions

LLM:
  - Applies policy terms to claim facts
  - Identifies applicable exclusions
  - Calculates net covered amount

Output: PolicyCheckOutput (coverage_status, covered_amount, exclusions, compliance_flags)
```

### 5. Settlement Calculator (LangGraph Node)

**File:** `src/agents/settlement_calculator.py`

```
Pre-compute:
  - Apply depreciation to assessed damage
  - Calculate: assessed - depreciation - deductible = settlement

LLM:
  - Validates calculation
  - Produces step-by-step breakdown
  - Checks regulatory compliance

Safety caps (applied AFTER LLM, overriding if needed):
  - settlement <= assessed_damage * 1.15 (never overpay)
  - settlement <= coverage_limit
  - settlement >= 0

Output: SettlementOutput (decision, settlement_amount, calculation_breakdown)
```

### 6. LLM-as-Judge Evaluator (LangGraph Node)

**File:** `src/evaluation/evaluator.py`

```
Separate LLM call (not the same context as pipeline agents)

Evaluates:
  - Accuracy (25%): Is the math correct?
  - Completeness (20%): Were all policy clauses checked?
  - Fairness (20%): Consistent with similar claims?
  - Safety (20%): Fraud signals handled properly?
  - Transparency (15%): Reasoning traceable?

Overall score = weighted average
Pass threshold: 0.70
Failed -> route to HITL instead of auto-releasing

Output: EvaluationOutput (overall_score, per_dimension_scores, passed, feedback)
```

### 7. Communication Agent (LangGraph Node)

**File:** `src/agents/communication_agent.py`

```
LLM generates:
  1. Claimant-facing email (professional, empathetic, legally accurate)
  2. Internal adjuster notes (technical, complete, legal-review ready)
  3. Next steps for claimant
  4. Appeal instructions (if denied/partial)

IMPORTANT: Never mentions fraud in claimant communication.
           Fraud context is internal notes only.

Output: CommunicationOutput (subject, message, internal_notes, next_steps)
```

---

## Conditional Routing Logic

### After Intake
```python
def route_after_intake(state):
    intake = state["intake_output"]
    if not intake.is_valid:
        return "communication_agent"     # PATH D: Denial
    if amount < $500 and clean_history:
        return "settlement_calculator"   # PATH E: Fast mode
    return "fraud_crew"                  # Continue normal pipeline
```

### After Fraud
```python
def route_after_fraud(state):
    fraud = state["fraud_output"]
    if fraud.fraud_score >= 0.90:
        return "auto_reject"             # PATH C: Confirmed fraud
    if fraud.fraud_score >= 0.65:
        return "hitl_checkpoint"         # PATH B: HITL escalation
    return "damage_assessor"             # PATH A: Continue normal
```

### After Evaluation
```python
def route_after_evaluation(state):
    if not state["evaluation_passed"]:
        return "hitl_checkpoint"         # Quality gate failed -> HITL
    return "communication_agent"         # Release decision
```

---

## Security Architecture

### PII Masking Flow

```
Raw claim:
  { "claimant_name": "Jane Smith",
    "claimant_email": "jane@email.com",
    "incident_description": "Contact jane@email.com for details" }

After mask_claim():
  { "claimant_name": "[CLAIMANT_NAME]",
    "claimant_email": "[EMAIL]",
    "incident_description": "Contact [EMAIL] for details" }

After mask for DOB, SSN:
  { "claimant_dob": "[REDACTED]",      // Field-level (complete removal)
    "ssn": "[REDACTED]" }               // Never sent to LLM
```

### Audit Log Entry

```json
{
  "timestamp": "2024-11-15T14:30:22.123456+00:00",
  "claim_id": "CLM-2024-001234",
  "agent": "settlement_calculator",
  "action": "settlement_calculation",
  "input": { "claim_type": "auto_collision", "estimated": 8500 },
  "output": { "decision": "approved", "settlement_usd": 6780, "confidence": 0.88 },
  "tokens_used": 2100,
  "cost_usd": 0.0008,
  "duration_ms": 2340,
  "error": null,
  "hash": "a1b2c3d4e5f6..."
}
```

---

## HITL Queue Architecture

### Database Schema (SQLite)

```sql
CREATE TABLE hitl_queue (
    ticket_id TEXT PRIMARY KEY,        -- HITL-XXXXXXXX
    claim_id TEXT NOT NULL,            -- CLM-2024-XXXXXX
    priority TEXT NOT NULL,            -- critical | high | normal
    priority_score REAL NOT NULL,      -- 0-100
    triggers TEXT NOT NULL,            -- JSON array of trigger reasons
    review_brief TEXT NOT NULL,        -- Formatted brief for reviewer
    state_snapshot TEXT NOT NULL,      -- JSON (PII-safe claim state)
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    sla_deadline TEXT NOT NULL,
    resolved_at TEXT,
    reviewer_id TEXT,
    human_decision TEXT,               -- One of ClaimDecision values
    human_notes TEXT,
    override_ai INTEGER DEFAULT 0      -- 1 if human overrode AI
);
```

### REST API

```
POST /hitl/enqueue          -> Creates ticket, returns ticket_id
GET  /hitl/queue             -> List pending (ordered by priority_score DESC)
GET  /hitl/ticket/{id}       -> Full ticket with brief and state snapshot
POST /hitl/decide/{id}       -> Submit human decision (resolves ticket)
GET  /hitl/stats             -> Queue summary statistics
```

---

## Guardrails Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 1: PRE-EXECUTION (before each agent call)                     │
│   - Agent call count < 25?                                          │
│   - Total tokens < 50,000?                                          │
│   - Total cost < $0.50?                                             │
│   - Same agent not called > 10 times? (loop detection)              │
│   - Execution time < 300 seconds?                                   │
│                                                                     │
│   HARD STOP: call/token/cost limits -> raises GuardrailsViolation   │
│   SOFT STOP: timeout -> skips remaining agents, logs warning        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 2: POST-EXECUTION (after each agent returns)                  │
│   - Output confidence >= threshold? (per-agent minimums)            │
│   - Reasoning fields non-empty? (hallucination proxy)               │
│   - Updates: call count, token count, cost accumulator              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 3: PIPELINE-LEVEL (routing decisions)                         │
│   - Fraud score routing (proceed / HITL / auto-reject)              │
│   - Evaluation quality gate (passed? -> release : HITL)             │
│   - Settlement safety cap (never exceed 115% of assessed damage)    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Orchestration | LangGraph StateGraph | Conditional routing, state management, HITL checkpoints |
| Fraud Crew | CrewAI Process.sequential | Role-based expert agents with manager synthesis |
| LLM | Google Gemini 2.5 Flash | Free tier, structured output, fast inference |
| Structured Output | Pydantic v2 + .with_structured_output() | Zero parsing errors, typed agent communication |
| HITL Queue | SQLite | Zero-dependency, embedded, works everywhere |
| HITL API | FastAPI | REST endpoints for human reviewer interface |
| Dashboard | Streamlit + custom CSS | Interactive claim processing and HITL review |
| PII Masking | regex + field-level rules | No external service dependency |
| Audit Logging | NDJSON + SHA-256 | Tamper-detectable, streamable, grep-friendly |
| Configuration | YAML + python-dotenv | Hierarchical config with environment overrides |
| Testing | pytest | 47 tests covering all non-LLM components |

---

## HITL Lifecycle (Visual Flow)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    HITL LIFECYCLE (Detailed)                                │
└────────────────────────────────────────────────────────────────────────────┘

  LangGraph Pipeline                    HITL System                 Human Reviewer
  ─────────────────                    ───────────                 ──────────────

  [Agent detects trigger]
  e.g. amount >= $10K
           │
           ▼
  check_hitl_required()
  Returns: triggers, priority, score
           │
           ▼
  format_hitl_brief()
  Creates human-readable summary
           │
           ▼
  enqueue_claim() ─────────────────► SQLite INSERT
  Returns: ticket_id                 hitl_queue table
           │                              │
           ▼                              │
  Pipeline marks claim as                 │
  ESCALATED_HUMAN_REVIEW                  │
  Pipeline stops here.                    │
                                          ▼
                                    GET /hitl/queue ◄──────── Reviewer opens
                                    Returns sorted            Streamlit Tab 2
                                    tickets by priority       or calls REST API
                                          │
                                          ▼
                                    GET /hitl/ticket/{id} ◄── Reviewer clicks
                                    Returns: full brief,      a ticket
                                    state snapshot,
                                    AI recommendation
                                          │
                                          ▼
                                                          ──► Reviewer reads brief
                                                              Reviews fraud score
                                                              Reads AI recommendation
                                                              │
                                                              ▼
                                    POST /hitl/decide/{id} ◄── Reviewer submits:
                                    Body: decision,             - decision (approve/deny)
                                          reviewer_id,          - notes
                                          notes,                - override_ai (bool)
                                          override_ai
                                          │
                                          ▼
                                    SQLite UPDATE
                                    status = 'resolved'
                                    resolved_at = now()
                                          │
                                          ▼
                                    log_hitl_event()
                                    Audit log entry with
                                    human decision recorded
                                          │
                                          ▼
  Pipeline resumes ◄─────────────── Decision available
  (in production: LangGraph            via get_human_decision()
   interrupt/resume pattern)
           │
           ▼
  Communication Agent generates
  notification based on human decision
           │
           ▼
  log_final_decision()
  human_reviewed = True
           │
           ▼
  END
```

---

## Complete Claim Lifecycle (All Layers)

This shows what happens for EVERY claim, including all three layers (security, guardrails, pipeline):

```
[Claimant submits claim via Streamlit or CLI]
        │
        ▼
┌─ SECURITY LAYER ──────────────────────────────────────────────────────────┐
│  1. mask_claim() - regex + field-level PII removal                        │
│  2. Original stored in state.claim (never sent to LLM)                    │
│  3. Masked version stored in state.masked_claim                           │
└───────────────────────────────────────────────┬───────────────────────────┘
                                                │
┌─ GUARDRAILS LAYER (pre-check) ────────────────┤
│  Check: agent_calls < 25?                     │
│  Check: total_tokens < 50,000?                │
│  Check: total_cost < $0.50?                   │
│  Check: same agent not called > 10x? (loop)   │
│  Check: execution_time < 300s?                │
│  HARD STOP if budget exceeded                 │
│  SOFT STOP if timeout (skip remaining agents) │
└───────────────────────────────────────────────┤
                                                │
┌─ PIPELINE LAYER ──────────────────────────────▼───────────────────────────┐
│                                                                            │
│  INTAKE AGENT                                                              │
│  ├── Policy lookup (SQLite, no LLM)                                        │
│  ├── Active date check (no LLM)                                            │
│  ├── Document completeness check (no LLM)                                  │
│  ├── LLM: nuanced eligibility assessment                                   │
│  └── Output: IntakeValidationOutput {is_valid, confidence, flags}          │
│       │                                                                    │
│       ├── [invalid] ──► COMMUNICATION AGENT (denial) ──► END              │
│       └── [valid] ──►                                                      │
│                                                                            │
│  FRAUD CREW (CrewAI)                                                       │
│  ├── Rule-based pattern check (no LLM, grounding)                          │
│  ├── Agent 1: Pattern Analyst (LLM + check_fraud_patterns tool)            │
│  ├── Agent 2: Anomaly Detector (LLM + anomaly_detection + baseline tools)  │
│  ├── Agent 3: Social Validator (LLM only, reasons from text)               │
│  ├── Composite: pattern(40%) + anomaly(35%) + consistency(25%)             │
│  └── Output: FraudAssessmentOutput {fraud_score, risk_level, concerns}     │
│       │                                                                    │
│       ├── [score >= 0.90] ──► AUTO-REJECT ──► COMMS ──► END              │
│       ├── [score >= 0.65] ──► HITL CHECKPOINT ──► COMMS ──► END          │
│       └── [proceed] ──►                                                    │
│                                                                            │
│  DAMAGE ASSESSOR                                                           │
│  ├── calculate_vehicle_acv() (no LLM)                                      │
│  ├── should_total_loss() (no LLM)                                          │
│  ├── get_repair_estimate_range() (no LLM)                                  │
│  ├── LLM: independent damage assessment with tool data as grounding        │
│  └── Output: DamageAssessmentOutput {assessed_usd, repair_vs_replace}      │
│       │                                                                    │
│       ▼                                                                    │
│  POLICY CHECKER                                                            │
│  ├── get_coverage_for_claim_type() (no LLM)                                │
│  ├── LLM: coverage determination with policy data                          │
│  └── Output: PolicyCheckOutput {coverage_status, covered_amount}           │
│       │                                                                    │
│       ▼                                                                    │
│  SETTLEMENT CALCULATOR                                                     │
│  ├── apply_depreciation() (no LLM)                                         │
│  ├── LLM: step-by-step calculation with safety cap enforcement             │
│  ├── Post-LLM safety cap: min(settlement, 115% of assessed, coverage_limit)│
│  └── Output: SettlementOutput {decision, settlement_amount, breakdown}     │
│       │                                                                    │
│       ▼                                                                    │
│  LLM-AS-JUDGE EVALUATOR                                                   │
│  ├── Separate LLM call (fresh context, no pipeline bias)                   │
│  ├── Scores: accuracy, completeness, fairness, safety, transparency        │
│  ├── Pass threshold: 0.70                                                  │
│  └── Output: EvaluationOutput {overall_score, passed, feedback}            │
│       │                                                                    │
│       ├── [score >= 0.70] ──►                                              │
│       └── [score < 0.70] ──► HITL CHECKPOINT ──► COMMS ──► END           │
│                                                                            │
│  COMMUNICATION AGENT                                                       │
│  ├── LLM: generate claimant email + internal notes                         │
│  ├── Never mentions fraud to claimant (internal only)                      │
│  └── Output: CommunicationOutput {subject, message, next_steps}            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                                │
┌─ GUARDRAILS LAYER (post-check) ───────────────┤
│  Check: output.confidence >= threshold?       │
│  Check: reasoning fields non-empty?           │
│  Update: agent_call_count, tokens, cost       │
└───────────────────────────────────────────────┤
                                                │
┌─ AUDIT LAYER ─────────────────────────────────▼───────────────────────────┐
│  log_agent_action() - after EVERY agent                                    │
│  log_hitl_event() - after HITL enqueue/resolve                             │
│  log_final_decision() - at pipeline end                                    │
│  Format: NDJSON, one file per day, SHA-256 hash per entry                  │
│  Retention: 7 years (insurance compliance)                                 │
└────────────────────────────────────────────────────────────────────────────┘
```
