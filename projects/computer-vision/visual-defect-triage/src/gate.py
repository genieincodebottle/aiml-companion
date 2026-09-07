"""Where the model stops and policy starts.

Routing is a business rule, so it lives in its own module with its own tests,
separate from anything that touches a tensor.
"""
from src.config import settings
from src.schemas import DefectClass, Route

# Some defects never auto-accept whatever the confidence, because a structural
# failure reaching a customer is not comparable to a scratch reaching one.
NEVER_AUTO = frozenset({DefectClass.HAIRLINE_CRACK, DefectClass.WELD_VOID})


def route(predicted: DefectClass, confidence: float) -> Route:
    if predicted in NEVER_AUTO:
        return Route.REVIEW
    if confidence >= settings.accept_above:
        return Route.AUTO_ACCEPT
    if confidence <= settings.reject_below:
        return Route.AUTO_REJECT
    return Route.REVIEW
