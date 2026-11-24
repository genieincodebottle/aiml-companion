# Interview-topic map

Each subsystem maps to the systems-design and AI-engineering questions a Staff /
Principal / AI Platform Engineer interview probes. Use this to rehearse: open the file,
explain the decision, name the trade-off.

## Distributed systems & reliability

| Topic | Where | The question it answers |
|-------|-------|-------------------------|
| Control-plane / data-plane split | `control_plane.py`, `gateway/`, `runtime/` | "How do you separate config from execution at scale?" |
| Event sourcing | `models/events.py` | "How is workflow state made durable and auditable?" |
| Durable execution / checkpoint-resume | `runtime/engine.py` | "How do you resume a hours-long agent without re-billing LLM calls?" |
| Saga / compensating transactions | `runtime/saga.py` | "There's no distributed transaction across an LLM, a DB, and a bus. Now what?" |
| Dead-letter queue | `runtime/dlq.py` | "What happens to a workflow you can't finish?" |
| Fallback / circuit-break | `gateway/core.py`, `gateway/providers.py` | "A model provider is down mid-request." |
| Idempotency / no re-run on resume | `runtime/engine.py` (`completed` skip, `_rehydrate`) | "How do you avoid double-executing a step after a restart?" |
| Maker-checker (four eyes) | `runtime/engine.py` (`resume_after_approval`) | "How do you stop one person approving their own regulatory filing?" |
| Approver audit trail | `audit` on the case, `approval_*` events | "Who approved this, when, and why?" |

## AI engineering / LLMOps / AgentOps

| Topic | Where | The question |
|-------|-------|--------------|
| AI gateway (routing, cache, budgets, audit) | `gateway/core.py` | "How does every model call get governed uniformly?" |
| Model routing (tier → model) | `gateway/core.py`, registry cards | "Cheap vs premium per step - who decides and how?" |
| Semantic caching | `gateway/cache.py` | "How do you avoid paying twice for the same question?" |
| Prompt registry + versioning + labels | `registry/prompts.py` | "How do you ship and roll back a prompt change safely?" |
| Shadow / canary / A-B | registry labels + eval gate | "How does a new prompt version earn production traffic?" |
| Offline eval + golden sets | `evaluation/run_eval.py` | "How do you know a change is better before shipping?" |
| LLM-as-judge | `evaluation/run_eval.py` | "How do you score open-ended generation at scale?" |
| Regression / promotion gate | `evaluation/run_eval.py` (`promote`) | "What blocks a regressed prompt from reaching users, and why not promote on an offline run?" |
| Structured outputs everywhere | `gateway/providers.py`, schemas | "How do you keep the runtime from parsing free text?" |
| Distributed tracing (OpenTelemetry) | `observability.py` | "How do you debug a 20-step multi-agent trajectory?" |
| Per-case cost accounting | `gateway/core.py` | "How do you attribute spend and cap it?" |

## Agentic architecture

| Topic | Where | The question |
|-------|-------|--------------|
| Hierarchical supervisor + workers | `tenants/fincrime/squad.py` | "Why not one big agent?" |
| Agent cards / A2A declaration | `registry/agents.py` | "How does an agent declare its capabilities and limits?" |
| Human-in-the-loop gate | `runtime/engine.py` | "Where and how does a human stay in control?" |
| Reflection / critic | `tenants/fincrime/agents.py` (critic) | "How do you catch a bad agent output before it acts?" |
| Evidence-grounded generation | narrative + critic | "How do you stop the model from fabricating claims?" |

## Security & governance

| Topic | Where | The question |
|-------|-------|--------------|
| Policy-as-code, fail-closed | `policy/engine.py` | "How is tool access authorized and defaulted?" |
| Least privilege | `policy/engine.py`, agent cards | "How do you bound what each agent can do?" |
| Zero-trust workload identity | `identity.py` | "How does one agent prove it may call a tool?" |
| Prompt-injection / indirect injection | `guardrails.py` | "Attacker-controlled text enters a tool result. Defense?" |
| PII masking / data protection | `gateway/redaction.py` | "How do you keep PII out of the provider and logs?" |
| Audit trail | `gateway/core.py` | "Prove to a regulator what every model call did." |

## Domain (financial crime)

| Topic | Where | The question |
|-------|-------|--------------|
| Entity-network / knowledge graph | `tenants/fincrime/data.py` | "How do you find structuring rings?" |
| Synthetic data with planted ground truth | `tenants/fincrime/data.py` | "How do you build/eval this without a bank's data?" |
| Regulated human-gated disposition | policy + runtime gate | "Why can't the agent file the SAR autonomously?" |
