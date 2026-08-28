# Operational Runbook

## Deployment

```bash
# Build and run (using Makefile)
make docker-run

# Or manually:
docker build -t model-server -f docker/Dockerfile .
docker run -p 8000:8000 model-server

# Verify health
curl http://localhost:8000/health
```

## Rollback

```bash
# Tag current version before deploying new one
docker tag model-server:latest model-server:previous

# If new version fails, rollback
docker stop model-server-new
docker run -p 8000:8000 model-server:previous
```

## Incident Response

1. **High latency (P95 > 200ms)**
   - Check container resource limits
   - Check model size and inference complexity
   - Scale horizontally if needed

2. **Error rate spike**
   - Check logs: `docker logs model-server`
   - Verify model file integrity
   - Check input data format changes

3. **Prediction drift**
   - Compare current prediction distribution to baseline
   - Check for upstream data pipeline changes
   - Trigger retraining if PSI > 0.25
