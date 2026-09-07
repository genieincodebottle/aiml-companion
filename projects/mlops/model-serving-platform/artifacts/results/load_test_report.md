# Load Test Report

> Sample results from Locust load test against local Docker container.

## Test Configuration
- **Users**: 100 concurrent
- **Spawn rate**: 10 users/second
- **Duration**: 5 minutes
- **Traffic**: 10% health + 90% predict

## Results

| Metric | Value |
|---|---|
| Total requests | 48,523 |
| Requests/sec | 161.7 |
| P50 latency | 12ms |
| P95 latency | 45ms |
| P99 latency | 89ms |
| Error rate | 0.02% |
| Avg response size | 142 bytes |

## SLA Compliance

| SLA | Target | Actual | Status |
|---|---|---|---|
| P95 latency | < 200ms | 45ms | PASS |
| Error rate | < 1% | 0.02% | PASS |
| Throughput | > 100 RPS | 161.7 | PASS |
