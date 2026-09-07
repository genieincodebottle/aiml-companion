---
name: submit-claim
description: Submit a sample claim to the running backend and show the pipeline result
argument-hint: "[sample file name from data/sample_claims/]"
disable-model-invocation: true
allowed-tools: Bash(curl *) Read
---

# Submit a Sample Claim

1. Read the sample claim file from `data/sample_claims/$ARGUMENTS`
2. If `$ARGUMENTS` is empty, list available sample files in `data/sample_claims/` and ask which one to use
3. Log in as `claimant` to get a JWT token:
   ```bash
   TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username":"claimant","password":"claim123"}' | jq -r '.access_token')
   ```
4. Pick one claim entry from the sample file (tell the user which path you're testing, e.g. "path_a_normal_approval")
5. Submit it:
   ```bash
   curl -s -X POST http://localhost:8000/api/claims/submit \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '<claim JSON>'
   ```
6. Report: claim_id, final status, decision, settlement amount, whether HITL was triggered, and any errors.