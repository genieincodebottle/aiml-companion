"""Calculator tool for data analysis.

Provides safe math evaluation for the analyst agent.
"""

import math
import logging

logger = logging.getLogger(__name__)

# Safe math functions available to the calculator
SAFE_MATH = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
}


def calculate(expression: str) -> str:
    """Safely evaluate a math expression and return the result as a string.

    Supports basic arithmetic (+, -, *, /, **) and common math functions
    (sqrt, log, round, abs, min, max, sum).

    Examples:
        calculate("2 + 3 * 4")          -> "14"
        calculate("sqrt(144)")           -> "12.0"
        calculate("round(3.14159, 2)")   -> "3.14"
    """
    try:
        # Only allow safe operations - no builtins, no imports
        result = eval(expression, {"__builtins__": {}}, SAFE_MATH)  # noqa: S307
        logger.info("Calculated: %s = %s", expression, result)
        return str(result)
    except Exception as e:
        logger.error("Calculation failed for '%s': %s", expression, e)
        return f"Error: could not evaluate '{expression}' - {e}"


def percentage_change(old: float, new: float) -> str:
    """Calculate percentage change between two values."""
    if old == 0:
        return "Error: cannot calculate percentage change from zero"
    change = ((new - old) / abs(old)) * 100
    direction = "increase" if change > 0 else "decrease"
    return f"{abs(change):.1f}% {direction}"


def compare_values(items: dict[str, float]) -> str:
    """Compare numeric values and return a ranked summary.

    Args:
        items: {"Label": value, ...}

    Returns:
        Formatted comparison string.
    """
    if not items:
        return "No items to compare."

    sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
    lines = []
    for rank, (label, value) in enumerate(sorted_items, 1):
        lines.append(f"  {rank}. {label}: {value}")

    return "Ranking (highest to lowest):\n" + "\n".join(lines)
