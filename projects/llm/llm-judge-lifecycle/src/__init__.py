"""LLM Judge Lifecycle: an LLM judge as a maintained system, not a static score.

Four phases, one per module group:

    benchmark.py  I   Birth       the labelled ground truth, and honest splits
    rart.py       II  Training    rubric text as the parameter, an LLM as the optimiser
    serving.py    III Deployment  the judge as gate AND critic, with a retry budget
    monitoring.py IV  Monitoring  a drift band pegged to human disagreement

Built after Kong et al., "The Lifecycle of LLM-as-a-Judge for Large-Scale
Recommendation Explanations" (Netflix, COLM 2026 workshops, arXiv:2608.18300).
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
