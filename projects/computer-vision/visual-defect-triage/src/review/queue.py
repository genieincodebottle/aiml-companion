"""Order the review queue by expected value rather than arrival time."""


def priority(confidence: float, batch_value: float) -> float:
    """Uncertainty times value.

    Uncertainty alone over-samples genuinely ambiguous images that no human can
    resolve either. Multiplying by the value of the batch keeps attention on
    decisions that are both informative and worth making.
    """
    uncertainty = 1.0 - abs(confidence - 0.5) * 2.0      # peaks at 0.5
    return uncertainty * batch_value


def order(items: list[dict]) -> list[dict]:
    """items: [{"confidence": float, "batch_value": float, ...}]"""
    return sorted(items, key=lambda i: -priority(i["confidence"], i["batch_value"]))
