"""Tests for financial and analytical calculators."""

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.calculators import (
    calculate_financial_ratios,
    calculate_risk_score,
    calculate_sentiment_score,
    format_currency,
)


class TestFinancialRatios:
    def test_net_margin_healthy(self):
        ratios = calculate_financial_ratios({"revenue": 1000, "net_income": 150})
        assert ratios["net_margin"]["value"] == 15.0
        assert ratios["net_margin"]["assessment"] == "healthy"

    def test_net_margin_negative(self):
        ratios = calculate_financial_ratios({"revenue": 1000, "net_income": -50})
        assert ratios["net_margin"]["value"] == -5.0
        assert ratios["net_margin"]["assessment"] == "negative"

    def test_current_ratio_strong(self):
        ratios = calculate_financial_ratios({"current_assets": 500, "current_liabilities": 200})
        assert ratios["current_ratio"]["value"] == 2.5
        assert ratios["current_ratio"]["assessment"] == "strong"

    def test_current_ratio_concerning(self):
        ratios = calculate_financial_ratios({"current_assets": 80, "current_liabilities": 200})
        assert ratios["current_ratio"]["assessment"] == "concerning"

    def test_roe_excellent(self):
        ratios = calculate_financial_ratios({
            "revenue": 1000, "net_income": 100,
            "total_assets": 500, "total_liabilities": 100,
        })
        assert ratios["return_on_equity"]["value"] == 25.0
        assert ratios["return_on_equity"]["assessment"] == "excellent"

    def test_empty_data(self):
        ratios = calculate_financial_ratios({})
        assert ratios == {}

    def test_zero_revenue(self):
        ratios = calculate_financial_ratios({"revenue": 0, "net_income": 100})
        assert "net_margin" not in ratios


class TestRiskScore:
    def test_no_risks(self):
        result = calculate_risk_score([])
        assert result["rating"] == "insufficient_data"

    def test_low_risk(self):
        risks = [{"severity": "low", "likelihood": "unlikely"}]
        result = calculate_risk_score(risks)
        assert result["rating"] == "low"

    def test_critical_risk(self):
        risks = [
            {"severity": "critical", "likelihood": "very_likely"},
            {"severity": "high", "likelihood": "likely"},
            {"severity": "critical", "likelihood": "likely"},
        ]
        result = calculate_risk_score(risks)
        assert result["rating"] in ("critical", "high")
        assert result["total_risks"] == 3

    def test_breakdown_counts(self):
        risks = [
            {"severity": "critical", "likelihood": "likely"},
            {"severity": "medium", "likelihood": "possible"},
            {"severity": "medium", "likelihood": "unlikely"},
        ]
        result = calculate_risk_score(risks)
        assert result["breakdown"]["critical"] == 1
        assert result["breakdown"]["medium"] == 2


class TestSentimentScore:
    def test_empty_events(self):
        result = calculate_sentiment_score([])
        assert result["label"] == "insufficient_data"

    def test_positive_sentiment(self):
        events = [
            {"sentiment": "positive", "impact": "high"},
            {"sentiment": "positive", "impact": "medium"},
        ]
        result = calculate_sentiment_score(events)
        assert result["label"] in ("positive", "very_positive")
        assert result["positive_count"] == 2

    def test_negative_sentiment(self):
        events = [
            {"sentiment": "negative", "impact": "high"},
            {"sentiment": "negative", "impact": "high"},
        ]
        result = calculate_sentiment_score(events)
        assert result["label"] in ("negative", "very_negative")
        assert result["negative_count"] == 2

    def test_mixed_sentiment(self):
        events = [
            {"sentiment": "positive", "impact": "high"},
            {"sentiment": "negative", "impact": "high"},
        ]
        result = calculate_sentiment_score(events)
        assert result["count"] == 2


class TestFormatCurrency:
    def test_trillions(self):
        assert format_currency(2.5e12) == "$2.5T USD"

    def test_billions(self):
        assert format_currency(1.8e9) == "$1.8B USD"

    def test_millions(self):
        assert format_currency(42e6) == "$42.0M USD"

    def test_thousands(self):
        assert format_currency(5500) == "$5.5K USD"

    def test_small_amount(self):
        assert format_currency(42.50) == "$42.50 USD"
