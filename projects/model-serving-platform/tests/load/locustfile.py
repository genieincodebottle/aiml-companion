"""
Locust Load Testing for ML Model Server.

Usage:
    locust -f tests/load/locustfile.py --host http://localhost:8000
    Target: 100 users, spawn rate 10/s, run for 5 minutes
"""
from locust import HttpUser, task, between
import random


class ModelUser(HttpUser):
    """Simulates realistic user traffic to the model server."""
    wait_time = between(0.1, 0.5)

    @task(1)
    def health_check(self):
        """10% of requests are health checks."""
        self.client.get("/health")

    @task(9)
    def predict(self):
        """90% of requests are predictions."""
        features = [random.gauss(0, 1) for _ in range(4)]
        self.client.post("/predict", json={"features": features})
