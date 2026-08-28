"""muse-glimmer-lab, a hands-on companion to the Muse Glimmer post.

Named exports only, so that importing one thing does not drag in httpx.
"""

from .config import ARCH, QUANTS, SAMPLING, REASONING_STRENGTHS

__all__ = ["ARCH", "QUANTS", "SAMPLING", "REASONING_STRENGTHS"]
__version__ = "0.1.0"
