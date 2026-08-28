---
name: check-backend
description: Check if the FastAPI backend is running and healthy
disable-model-invocation: true
allowed-tools: Bash(curl *)
---

# Check Backend Health

1. Check if the backend is reachable:
   ```bash
   curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs
   ```
2. If reachable, check auth by logging in:
   ```bash
   curl -s -X POST http://localhost:8000/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username":"admin","password":"admin123"}'
   ```
3. Report: backend status, whether auth works, and the active LLM provider/country if visible.
4. If not reachable, tell the user to start it with `uvicorn api.main:app --port 8000`.