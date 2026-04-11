# Smart Claims Processor - Architecture Deep Dive

## Why Insurance Claims Processing?

Insurance claims are the perfect multi-agent use case because:

1. **High stakes** - Financial decisions with real consequences
2. **Multiple expert domains** - Fraud detection, damage assessment, policy law, actuarial math
3. **Natural HITL** - Regulations often require human sign-off above certain amounts
4. **Auditability** - Every decision must be traceable for legal/regulatory review
5. **Fraud risk** - ~10% of claims are fraudulent, requiring adversarial thinking

## Framework Design Decision: LangGraph + CrewAI

### Why LangGraph for the main pipeline?
- **State machine orchestration** - Claims have clear states and transitions
- **Conditional routing** - Different paths for fraud, HITL, auto-reject, fast-mode
- **Built-in checkpointing** - HITL pause/resume is native to LangGraph
- **Append-only state** - `Annotated[list, operator.add]` prevents race conditions

### Why CrewAI for fraud detection?
- **Role-based expertise** - Three agents each bring a distinct expert perspective
- **Manager delegation** - CrewAI's manager pattern naturally synthesizes consensus
- **Independent reasoning** - Each agent sees the same data but reasons differently
- **Simpler for "committees"** - CrewAI is more natural when you want N agents to debate and agree

### Combined value
Neither framework alone would be as good:
- LangGraph alone: You'd write awkward "internal loops" for multi-perspective fraud analysis
- CrewAI alone: You'd struggle with the complex conditional routing of the full claims pipeline

## Security Architecture

### PII Masking Strategy
```
Raw Claim (with PII)
        │
        ▼
PIIMasker.mask_claim()
        │
        ▼
Masked Claim (safe for LLMs)  ──► All LLM prompts use this
        │
        ▼
Stored in ClaimsState.masked_claim
        │
Original Claim.claimant_* fields ──► Only used for final communication merge
```

**What gets masked:**
- Email: `→ [EMAIL]`
- Phone: `→ [PHONE]`
- SSN: `→ [SSN]`
- Credit card: `→ [CREDIT_CARD]`
- DOB: `→ [REDACTED]`
- Names: `→ [CLAIMANT_NAME]` (field-level, not regex)

### Audit Log Architecture
- **Format**: NDJSON (newline-delimited JSON) - streamable and grep-able
- **Immutability**: SHA-256 hash of each entry for tamper detection
- **Retention**: 7 years (insurance compliance standard)
- **Granularity**: Every agent action, HITL event, and final decision logged
- **Failure handling**: Audit log errors are alerted but never crash the pipeline

## Guardrails Architecture

Three layers:

### Layer 1: Pre-execution (before each agent)
- Token budget not exceeded
- Cost budget not exceeded
- Agent call count not exceeded
- No loop detected (same agent called too many times)
- Execution time not exceeded

### Layer 2: Post-execution (after each agent returns)
- Output confidence above minimum threshold
- Reasoning fields not empty (hallucination proxy check)

### Layer 3: Pipeline-level (routing logic)
- Evaluation score gate (LLM-as-judge must pass before release)
- Fraud score thresholds (auto-reject vs HITL vs proceed)
- Policy coverage (deny fast-path if clearly not covered)

## HITL Architecture

### Priority Scoring
```
Priority Score (0-100) = 
  (amount_factor    × 0.30) +
  (fraud_score      × 0.35) +
  (confidence_gap   × 0.20) +
  (repeat_claimant  × 0.15)
  × 100

CRITICAL:  score >= 80  (4-hour SLA)
HIGH:      score 60-79  (24-hour SLA)
NORMAL:    score < 60   (72-hour SLA)
```

### HITL Workflow
```
Trigger detected
       │
       ▼
enqueue_claim() → SQLite queue
       │
       ▼
LangGraph node pauses (or marks ESCALATED_HITL)
       │
FastAPI reviewer polls GET /hitl/queue
       │
Human reviews brief + state snapshot
       │
Human submits POST /hitl/decide/{ticket_id}
       │
LangGraph resumes from checkpoint
       │
Decision merged into state → communication_agent
```

### LangGraph Interrupt Pattern (Production)
In production, use LangGraph's native `interrupt()`:
```python
from langgraph.types import interrupt

def hitl_checkpoint_node(state):
    ticket_id = enqueue_claim(...)
    # Pause execution until human decides
    human_result = interrupt({"ticket_id": ticket_id, "brief": review_brief})
    return {"human_decision": human_result["decision"]}
```
For learning, we show the queue pattern with polling.

## Evaluation Architecture

### LLM-as-Judge Pattern
```
Full pipeline outputs
        │
        ▼
Judge LLM (separate Gemini call)
        │
        ▼
EvaluationOutput {
    accuracy_score,
    completeness_score,
    fairness_score,
    safety_score,
    transparency_score,
    overall_score,
    passed,
    feedback,
    flags,
}
        │
overall_score >= 0.70?
        │
    YES │                    NO
        ▼                    ▼
communication_agent    hitl_checkpoint (route for human review)
```

### Why a separate judge model?
The same model that made the decision has cognitive bias toward its own outputs.
A separate judge call (even the same model, but without the pipeline context)
evaluates more objectively. In production, use a different model family as judge.

## Cost Optimization

### Fast Mode (< $500 claims, clean history)
Skip: fraud_crew, damage_assessor
Saves: ~60% of tokens
Use when: Amount tiny + no fraud flags + policy clearly active

### Sampling Evaluation
Only evaluate 10% of routine claims automatically.
Always evaluate: HITL claims, >$10K claims, human overrides.
Saves: ~90% of evaluation costs on bulk processing.

### CrewAI Cost Control
- `max_rpm=10` throttles crew requests
- `max_iter=3` caps each agent's reasoning loops
- Pattern checks run BEFORE LLM (rules first, LLM second)
