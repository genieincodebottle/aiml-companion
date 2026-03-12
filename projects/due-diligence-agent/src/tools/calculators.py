"""
Financial and analytical calculators for agent tool use.

Pure functions with no LLM calls - deterministic, testable.
"""

from __future__ import annotations

import math


def calculate_financial_ratios(data: dict) -> dict:
    """Calculate standard financial health ratios from raw data.

    Args:
        data: Dict with keys like 'revenue', 'net_income', 'total_assets',
              'total_liabilities', 'current_assets', 'current_liabilities',
              'operating_cash_flow', 'total_debt', 'ebitda', 'shares_outstanding'.

    Returns:
        Dict of calculated ratios with interpretations.
    """
    ratios = {}

    # Profitability
    revenue = data.get("revenue", 0)
    net_income = data.get("net_income", 0)
    if revenue and revenue != 0:
        margin = net_income / revenue * 100
        ratios["net_margin"] = {
            "value": round(margin, 2),
            "unit": "%",
            "assessment": "healthy" if margin > 10 else "moderate" if margin > 5 else "weak" if margin > 0 else "negative",
        }

    gross_profit = data.get("gross_profit", 0)
    if revenue and revenue != 0 and gross_profit:
        gm = gross_profit / revenue * 100
        ratios["gross_margin"] = {
            "value": round(gm, 2),
            "unit": "%",
            "assessment": "strong" if gm > 40 else "moderate" if gm > 20 else "thin",
        }

    # Liquidity
    ca = data.get("current_assets", 0)
    cl = data.get("current_liabilities", 0)
    if cl and cl != 0:
        cr = ca / cl
        ratios["current_ratio"] = {
            "value": round(cr, 2),
            "unit": "x",
            "assessment": "strong" if cr > 2 else "adequate" if cr > 1 else "concerning",
        }

    # Leverage
    total_assets = data.get("total_assets", 0)
    total_liabilities = data.get("total_liabilities", 0)
    if total_assets and total_assets != 0:
        dte = total_liabilities / total_assets
        ratios["debt_to_assets"] = {
            "value": round(dte, 2),
            "unit": "x",
            "assessment": "low_leverage" if dte < 0.4 else "moderate" if dte < 0.6 else "high_leverage",
        }

    total_debt = data.get("total_debt", 0)
    ebitda = data.get("ebitda", 0)
    if ebitda and ebitda > 0:
        de = total_debt / ebitda
        ratios["debt_to_ebitda"] = {
            "value": round(de, 2),
            "unit": "x",
            "assessment": "healthy" if de < 2 else "manageable" if de < 4 else "high" if de < 6 else "critical",
        }

    # Return on equity
    equity = total_assets - total_liabilities
    if equity and equity > 0:
        roe = net_income / equity * 100
        ratios["return_on_equity"] = {
            "value": round(roe, 2),
            "unit": "%",
            "assessment": "excellent" if roe > 20 else "good" if roe > 15 else "moderate" if roe > 10 else "low",
        }

    return ratios


def calculate_risk_score(risks: list[dict]) -> dict:
    """Calculate aggregate risk score from a list of risk items.

    Args:
        risks: List of dicts with 'severity' and 'likelihood' keys.

    Returns:
        Aggregate risk score and rating.
    """
    if not risks:
        return {"score": 0.0, "rating": "insufficient_data", "breakdown": {}}

    severity_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    likelihood_weights = {"very_likely": 4, "likely": 3, "possible": 2, "unlikely": 1, "rare": 0.5}

    total_score = 0
    max_possible = 0
    breakdown = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for risk in risks:
        sev = severity_weights.get(risk.get("severity", "medium"), 2)
        lik = likelihood_weights.get(risk.get("likelihood", "possible"), 2)
        total_score += sev * lik
        max_possible += 16  # 4 * 4
        sev_key = risk.get("severity", "medium")
        if sev_key in breakdown:
            breakdown[sev_key] += 1

    normalized = total_score / max_possible if max_possible > 0 else 0
    rating = (
        "critical" if normalized > 0.75
        else "high" if normalized > 0.5
        else "moderate" if normalized > 0.25
        else "low"
    )

    return {
        "score": round(normalized, 3),
        "rating": rating,
        "total_risks": len(risks),
        "breakdown": breakdown,
    }


def calculate_sentiment_score(events: list[dict]) -> dict:
    """Calculate aggregate sentiment from a list of news events.

    Args:
        events: List of dicts with 'sentiment' and 'impact' keys.

    Returns:
        Aggregate sentiment score and trend assessment.
    """
    if not events:
        return {"score": 0.0, "label": "insufficient_data", "count": 0}

    sentiment_values = {"positive": 1.0, "neutral": 0.0, "negative": -1.0, "mixed": -0.2}
    impact_weights = {"high": 3.0, "medium": 2.0, "low": 1.0}

    weighted_sum = 0
    total_weight = 0

    for event in events:
        sv = sentiment_values.get(event.get("sentiment", "neutral"), 0)
        iw = impact_weights.get(event.get("impact", "medium"), 2)
        weighted_sum += sv * iw
        total_weight += iw

    score = weighted_sum / total_weight if total_weight > 0 else 0
    label = (
        "very_positive" if score > 0.5
        else "positive" if score > 0.15
        else "neutral" if score > -0.15
        else "negative" if score > -0.5
        else "very_negative"
    )

    return {
        "score": round(score, 3),
        "label": label,
        "count": len(events),
        "positive_count": sum(1 for e in events if e.get("sentiment") == "positive"),
        "negative_count": sum(1 for e in events if e.get("sentiment") == "negative"),
    }


def format_currency(value: float, currency: str = "USD") -> str:
    """Format a number as currency string."""
    if abs(value) >= 1e12:
        return f"${value / 1e12:.1f}T {currency}"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.1f}B {currency}"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M {currency}"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.1f}K {currency}"
    return f"${value:.2f} {currency}"
