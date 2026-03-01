"""
Prometheus Metrics Integration.

Latency histogram, prediction counter, error counter for monitoring.

Usage:
    from src.app.metrics import PREDICTION_LATENCY, PREDICTION_COUNT, track_prediction
"""
import time
from prometheus_client import Histogram, Counter, start_http_server

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time to serve a prediction',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions served',
    ['model_version', 'prediction_class']
)

ERROR_COUNT = Counter(
    'prediction_errors_total',
    'Total prediction errors',
    ['error_type']
)


def track_prediction(model_version, prediction, latency):
    """Record prediction metrics."""
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_COUNT.labels(
        model_version=model_version,
        prediction_class=str(prediction)
    ).inc()


def track_error(error_type):
    """Record error metrics."""
    ERROR_COUNT.labels(error_type=error_type).inc()


def start_metrics_server(port=9090):
    """Start Prometheus metrics endpoint."""
    start_http_server(port)
    print(f"Metrics server started on port {port}")
