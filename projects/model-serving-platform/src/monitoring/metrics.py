"""
Monitoring - Latency tracking, drift detection, alerting.
"""
import time
import numpy as np
from collections import deque


class MetricsCollector:
    """Collect and report prediction metrics.
    
    TODO: Track P50/P95/P99 latency, prediction distribution, error rate.
    """
    
    def __init__(self, window_size=1000):
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.errors = 0
        self.total = 0
    
    def record_prediction(self, latency_ms, prediction):
        self.latencies.append(latency_ms)
        self.predictions.append(prediction)
        self.total += 1
    
    def record_error(self):
        self.errors += 1
        self.total += 1
    
    def get_latency_percentiles(self):
        if not self.latencies:
            return {}
        arr = np.array(self.latencies)
        return {
            "p50": np.percentile(arr, 50),
            "p95": np.percentile(arr, 95),
            "p99": np.percentile(arr, 99),
        }
    
    def get_error_rate(self):
        return self.errors / max(self.total, 1)


if __name__ == "__main__":
    collector = MetricsCollector()
    # Smoke test
    for i in range(100):
        collector.record_prediction(latency_ms=np.random.exponential(20), prediction=np.random.randint(0, 2))
    print(f"Latency percentiles: {collector.get_latency_percentiles()}")
    print(f"Error rate: {collector.get_error_rate()}")
