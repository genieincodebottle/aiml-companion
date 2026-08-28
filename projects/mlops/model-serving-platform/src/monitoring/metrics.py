"""
Monitoring - Latency tracking, drift detection (PSI), alerting.
"""
from collections import deque

import numpy as np


def compute_psi(reference, current, n_bins=10):
    """Population Stability Index between reference and current distributions.

    PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))

    Industry thresholds:
    - PSI < 0.10: stable, no action
    - 0.10 <= PSI < 0.25: moderate shift, investigate
    - PSI >= 0.25: significant shift, retrain
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Bin edges from the reference distribution's quantiles
    breakpoints = np.unique(np.percentile(reference, np.linspace(0, 100, n_bins + 1)))
    if len(breakpoints) < 2:
        return 0.0

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Small epsilon floor avoids log(0) / division by zero on empty bins
    eps = 1e-4
    ref_pct = ref_counts / len(reference) + eps
    cur_pct = cur_counts / len(current) + eps
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify_psi(psi_value):
    """Map a PSI value to an action category."""
    if psi_value < 0.10:
        return "stable"
    if psi_value < 0.25:
        return "moderate_shift"
    return "significant_shift"


class MetricsCollector:
    """Collect and report prediction metrics: P50/P95/P99 latency,
    prediction distribution (for drift checks via PSI), and error rate.
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

    def check_prediction_drift(self, reference_predictions, n_bins=10):
        """PSI of recent predictions vs a reference window.

        Returns (psi, status) where status is stable / moderate_shift /
        significant_shift.
        """
        psi = compute_psi(reference_predictions, list(self.predictions), n_bins=n_bins)
        return psi, classify_psi(psi)


if __name__ == "__main__":
    collector = MetricsCollector()
    # Smoke test
    rng = np.random.default_rng(42)
    reference = rng.integers(0, 2, 500)
    for i in range(100):
        collector.record_prediction(latency_ms=rng.exponential(20), prediction=int(rng.integers(0, 2)))
    print(f"Latency percentiles: {collector.get_latency_percentiles()}")
    print(f"Error rate: {collector.get_error_rate()}")
    psi, status = collector.check_prediction_drift(reference)
    print(f"Prediction drift: PSI={psi:.4f} ({status})")
