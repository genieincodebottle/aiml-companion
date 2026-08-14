# Northwind Freight - Deployment and Operations Runbook

**Owner:** Forward Deployed Engineer, `<your name>`
**Customer contact:** `<name>`, Head of Dispatch Operations
**Status:** `<Draft | Reviewed by Northwind Security | Approved for go-live>`
**Last updated:** `<YYYY-MM-DD>`

This is the document you hand to Northwind's platform and security teams. They
will read it without you in the room, and they will decide whether the system is
allowed to run based on what is written here. Anything you cannot point at in
code or configuration does not belong in it.

Fill in every `<angle bracket>` before the security review. A runbook with
placeholders left in it tells the reader the system was never operated.

---

## 1. Network boundary

Everything inside the dashed line runs in Northwind's AWS account, in their VPC,
under their IAM. Nothing in this repository provisions infrastructure outside it.

```
                            PUBLIC INTERNET
                                  |
                                  | HTTPS 443 only
                                  v
              +-------------------------------------+
              |  Northwind ALB  (their WAF, their    |
              |  TLS cert, their access logs)        |
              +-------------------------------------+
                                  |
  - - - - - - - - - - - - - - - - | - - - - - - - - - - - - - - - - - - - - -
  ,  NORTHWIND VPC  <vpc-id>      |            private subnets only          ,
  '                               v                                          '
  ,   +----------------------------------------------------------------+     ,
  '   |  app container  (deploy/Dockerfile, non-root uid 10001)         |    '
  ,   |    - JWT verification: deploy/jwt_middleware.py                 |     ,
  '   |    - scope enforcement at the edge, fails closed                |    '
  ,   +----------------------------------------------------------------+     ,
  '        |                    |                       |                    '
  ,        v                    v                       v                     ,
  '   +----------+     +------------------+     +------------------+         '
  ,   | legacy   |     |  model endpoint  |     |  audit log sink  |          ,
  '   | dispatch |     |  <in-boundary>   |     |  <CloudWatch or  |         '
  ,   | API      |     |                  |     |   S3 bucket>     |          ,
  '   +----------+     +------------------+     +------------------+         '
  ,        ^                                                                  ,
  '        |  private route, no NAT                                          '
  - - - - -|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           |
    Northwind's existing dispatch database  <identifier>
```

The compose file in this directory mirrors that topology locally. The
`internal: true` network there is the local stand-in for private subnets with no
NAT gateway. If you change either one, change both, and say why in an ADR.

---

## 2. What leaves the VPC, and what does not

State this precisely. "The data stays inside" is a claim; the table is evidence.

### Does NOT leave

| Data | Where it lives | Control that keeps it there |
|---|---|---|
| Support ticket text | app memory + audit log | private subnets, no NAT route |
| Consignee names and addresses | app memory only, never persisted by us | no outbound egress on the model path |
| Dispatch records from the legacy API | app memory | private route to the legacy host |
| Retrieval index built from their corpus | `<EBS volume / EFS mount>` | encrypted at rest with `<KMS key>` |
| Model prompts and completions | audit log inside their account | log sink is a Northwind resource |

### DOES leave, and under what conditions

| Data | Destination | Why | Approved by |
|---|---|---|---|
| `<aggregate counts only, no ticket text>` | `<your monitoring>` | uptime | `<name, date>` |
| `<nothing else>` | | | |

If you cannot complete the second table with "nothing else", stop and get it in
writing before go-live. The most common way this goes wrong is quiet, not loud:
an SDK with default telemetry, an error tracker capturing request bodies, or a
model API called from a code path nobody drew on the diagram.

### The model question

The single decision that determines whether the residency claim survives.

- **In-boundary model:** `<name, and where it runs>`. Nothing leaves. Costs more
  and constrains the model choice.
- **Hosted model via approved egress:** requires an explicit allowlisted proxy,
  a signed data processing agreement, and confirmation that the provider does
  not train on submitted data.

Decision: `<which, and the ADR number that argues it>`.

---

## 3. SSO integration

Northwind uses `<Okta | Entra ID | other>`. The app never holds a password and
never issues its own tokens.

1. Northwind creates an OIDC application for the service and gives you the
   issuer URL and JWKS URI.
2. Register the API audience: `api://northwind-dispatch-ai`. The audience is
   what stops a valid token minted for a different internal app from working
   here. Do not skip it because signature verification already passes.
3. Create the scopes. Two at minimum, and they must be separate groups:
   - `dispatch:read`  - triage, retrieval, viewing tickets
   - `dispatch:write` - reroute, hold, anything that mutates their system
   - `audit:read`     - reading the audit log, granted to compliance not to ops
4. Map scopes to Northwind's existing AD groups. Do not invent a new group
   structure. Their joiner/leaver process already drives the existing groups;
   anything you create is a group nobody deprovisions.
5. Set `OIDC_ISSUER` and `OIDC_AUDIENCE` in the task definition. Both are
   non-secret. The service refuses to start if either is missing, which is
   deliberate: a default issuer is a default trust anchor.
6. Verify the split before go-live, in their environment, with their tokens:

   ```
   # read token against a read route  -> 200
   # read token against a write route -> 403, not 401
   # expired token                    -> 401
   # token for another audience       -> 401
   ```

   Record the four results with timestamps. `tests/test_mcp_auth.py` asserts the
   same behaviour locally, but a local pass is not evidence about their IdP.

**Token lifetime:** `<15 min access, refresh via their IdP>`. Long-lived tokens
turn a single leaked log line into standing access.

---

## 4. Audit logging

| Question | Answer |
|---|---|
| What is recorded | every tool call: timestamp, subject (`sub`), scopes used, route, tool name, arguments hash, outcome, latency |
| What is NOT recorded | raw consignee PII in argument values, model API keys, full ticket bodies unless `<agreed>` |
| Destination | `<CloudWatch log group / S3 bucket, named>` in Northwind's account |
| Retention | `<N>` days, set by Northwind policy `<reference>`, not by us |
| Immutability | `<S3 object lock / log group with no delete permission on the app role>` |
| Who can read it | `audit:read` scope, granted to `<compliance group>` |
| Alerting | `<alarm on write-scope calls outside business hours>` |

Two rules that are not negotiable:

1. **No tool call completes without an audit record.** Write the record before
   returning the result, not after. If the audit write fails, the call fails.
   An audit log with gaps is not usable as evidence, and the gaps will be
   exactly the calls that mattered.
2. **The application role cannot delete from the log sink.** If the app can
   erase its own trail, the trail proves nothing to an auditor.

---

## 5. Break-glass access

For when the system is failing and the normal path is not available.

**Preconditions.** Break-glass is for production incidents, not for debugging
convenience and not for a demo that is going badly.

**Procedure.**

1. Incident commander (`<name / rota>`) declares the incident in `<channel>`.
2. Assume the break-glass role `<role ARN>` via `<their PAM tool>`. It is
   time-boxed to `<60>` minutes and cannot be extended, only re-requested.
3. Post the reason in `<channel>` at the moment of assumption, not afterwards.
4. All actions taken under the role are logged to `<destination>` and reviewed
   by `<name>` within `<1 business day>`.
5. If credentials were exposed, rotate before closing: `<which secrets, where>`.
6. Write the incident note within 24 hours. `docs/escalation-memo.md` is the
   format.

**What break-glass does NOT grant:** direct write access to the production
dispatch database. If an incident appears to require that, it is a Northwind
decision made by `<name>`, not an FDE decision made at 2am.

**Last exercised:** `<date>`. An untested break-glass path is a plan, not a
capability. Exercise it before go-live, once, on purpose.

---

## 6. Incident contacts

| Role | Name | Channel | Hours |
|---|---|---|---|
| FDE (you) | `<name>` | `<slack / phone>` | `<hours + timezone>` |
| Northwind ops lead | `<name>` | `<channel>` | `<hours>` |
| Northwind platform on-call | `<rota>` | `<pagerduty>` | 24/7 |
| Northwind security | `<name>` | `<channel>` | `<hours>` |
| Data protection officer | `<name>` | `<email>` | business hours |
| Escalation after 30 min | `<exec sponsor>` | `<channel>` | |

**Severity definitions.** Agree these with Northwind, do not assume them.

- **Sev1** - system is producing wrong dispatch decisions, or data has left the
  boundary. Page immediately. Kill switch: `<how to disable, in one command>`.
- **Sev2** - system is down but not wrong. Ops falls back to the manual process.
- **Sev3** - degraded quality, no incorrect action taken.

A wrong answer is more severe than no answer. If the ops team cannot tell which
of those is happening, that is a Sev1 by default.

---

## 7. Pre-go-live checklist

Do not schedule go-live until every line is checked and initialled.

**Security**
- [ ] Image runs as non-root, confirmed in their cluster, not just locally
- [ ] No secrets in the image (`docker history` reviewed, layers included)
- [ ] Secrets injected from `<their secret manager>` at runtime
- [ ] Read/write scope split verified against their IdP with real tokens
- [ ] All four token tests in section 3 recorded with timestamps
- [ ] Unknown-route policy verified to fail closed, not open
- [ ] Container security review signed off by `<name, date>`

**Data**
- [ ] Egress table in section 2 completed and countersigned
- [ ] Model residency decision made and recorded in an ADR
- [ ] No third-party telemetry SDK is active (dependency list reviewed)
- [ ] Retrieval index encrypted at rest, key owned by Northwind
- [ ] PII redaction verified on a sample of 20 real audit records

**Operations**
- [ ] Healthcheck verified to fail when the app is genuinely unhealthy, not just
      when the process is dead
- [ ] Audit log writing to the agreed sink, retention policy applied
- [ ] Alarms firing to a channel someone actually reads
- [ ] Break-glass exercised once end to end
- [ ] Rollback tested: previous image redeployed in `<N>` minutes
- [ ] Kill switch tested by someone from Northwind, not by you

**Evaluation**
- [ ] `python eval/run_eval.py` passes, with >= 30 cases across 5 modes
- [ ] Thresholds in `eval/thresholds.yaml` defended in an ADR
- [ ] Northwind ops lead has seen the per-failure-mode table and agrees the
      numbers are acceptable for their process
- [ ] Known failure modes documented in `docs/handover.md`

**Handover**
- [ ] `docs/handover.md` complete, and someone at Northwind has run the system
      end to end using only that document, with you silent in the room
- [ ] Named owner at Northwind for each component
- [ ] Support arrangement agreed in writing, with an end date

---

## 8. Known limitations

State these before go-live, in writing. Every one of them will otherwise be
discovered by a user and reported as a defect.

- `<the two policy documents contradict each other on porter release; the system
  routes those cases to a human rather than picking a side>`
- `<the 2024/2025 schema change means records before <date> lack <field>>`
- `<the legacy API rate-limits without a header, so ingest backs off blind>`
- `<the evaluation set is <N> cases, drawn from <window>; performance on ticket
  types outside that window is unmeasured>`
