"""Cost per million tokens, from MEASURED throughput. YOU implement it.

There is deliberately no price table in this repo. GPU prices change weekly and differ by
cloud, region and commitment. You supply `gpu_hourly_usd` from the provider page you
rented from, and your report names that page and the date you read it.
"""
from __future__ import annotations


def cost_per_million_tokens(tokens_per_s: float, gpu_hourly_usd: float, num_gpus: int = 1,
                            utilisation: float = 1.0) -> float:
    """YOU. Dollars to produce one million tokens.

        cost = (gpu_hourly_usd * num_gpus) / (tokens_per_s * 3600 * utilisation) * 1e6

    `utilisation` (0 < u <= 1) is the fraction of the hour the GPUs are doing useful work at
    the measured rate; a fleet sized for peak is idle most of the day, and that idle time is
    part of the real cost. Raise ValueError when tokens_per_s <= 0, utilisation is outside
    (0, 1], or gpu_hourly_usd < 0.

    Example: $2.00/h, 1000 tok/s, 1 GPU, full utilisation -> about $0.556 per million tokens.
    """
    raise NotImplementedError("analysis/cost.py: cost_per_million_tokens")


# ======================================================================================
# GIVEN (not stubs): the input-token and self-host-versus-API arithmetic. analysis.decide
# calls these. Read them: every number in the report's cost section comes from here.
# ======================================================================================
def cost_per_request(req_per_s: float, gpu_hourly_usd: float, num_gpus: int = 1, utilisation: float = 1.0) -> float:
    """GIVEN. GPU dollars per request: (gpu_hourly_usd * num_gpus / 3600) / req_per_s / utilisation.

    This charges every GPU second to the requests it served, input and output alike: prefill and
    decode share the same engine steps, so engine_sim and a real server cannot bill them apart."""
    if req_per_s <= 0 or not (0 < utilisation <= 1) or gpu_hourly_usd < 0:
        raise ValueError("req_per_s > 0, 0 < utilisation <= 1 and gpu_hourly_usd >= 0 are required")
    return gpu_hourly_usd * num_gpus / 3600.0 / req_per_s / utilisation


def blended_cost_per_million_tokens(cost_per_request_usd: float, mean_prompt_tokens: float,
                                    mean_output_tokens: float) -> float:
    """GIVEN. Dollars per million tokens when input and output tokens are counted together.

    Compare with cost_per_million_tokens (output tokens only). The blended figure is lower because
    the same GPU time is spread over more tokens. It prices a prompt token like an output token,
    which a real API does not, so use it for orders of magnitude, not for a price list."""
    total = mean_prompt_tokens + mean_output_tokens
    if total <= 0:
        raise ValueError("mean_prompt_tokens + mean_output_tokens must be positive")
    return cost_per_request_usd / total * 1e6


def api_cost_per_request(mean_prompt_tokens: float, mean_output_tokens: float,
                         api_input_usd_per_m: float, api_output_usd_per_m: float) -> float:
    """GIVEN. What the same request costs through a hosted API: you supply both prices (input and
    output are priced separately) and the page and date you read them from. No price table here."""
    if min(api_input_usd_per_m, api_output_usd_per_m, mean_prompt_tokens, mean_output_tokens) < 0:
        raise ValueError("prices and token counts must be non-negative")
    return (mean_prompt_tokens * api_input_usd_per_m + mean_output_tokens * api_output_usd_per_m) / 1e6


def breakeven_utilisation(cost_per_request_full_usd: float, api_cost_per_request_usd: float) -> float:
    """GIVEN. The fraction of the time your GPUs must be busy for self-hosting to cost the same as
    the API: cost_per_request_at_full_utilisation / api_cost_per_request. Above this fraction
    self-hosting is cheaper. Above 1.0 it cannot win at this operating point, however busy the
    fleet is. It ignores engineering time, the quality gap between models and the API's rate limits."""
    if api_cost_per_request_usd <= 0:
        raise ValueError("api_cost_per_request_usd must be positive")
    return cost_per_request_full_usd / api_cost_per_request_usd
