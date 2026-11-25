"""GPU-hours per million tokens, from MEASURED throughput. YOU implement it.

This repo works in GPU time, not money. A GPU-hour is one GPU busy for one hour, the same unit on
every cloud and in every year. It tells you how much hardware a workload uses, which is the part
you control. Any price you attach later is a multiplication outside this repo.
"""
from __future__ import annotations


def gpu_hours_per_million_tokens(tokens_per_s: float, num_gpus: int = 1, utilisation: float = 1.0) -> float:
    """YOU. GPU-hours needed to produce one million tokens.

        gpu_hours = num_gpus * 1e6 / (tokens_per_s * 3600 * utilisation)

    `utilisation` (0 < u <= 1) is the fraction of the hour the GPUs are doing useful work at
    the measured rate; a fleet sized for peak is idle most of the day, and that idle time is
    part of the real usage. With utilisation below 1 the result is the EFFECTIVE GPU-hours.
    Raise ValueError when tokens_per_s <= 0, utilisation is outside (0, 1], or num_gpus < 1.

    Worked example (arithmetic only, not a measurement): 1000 tok/s, 1 GPU, full utilisation
    -> 1e6 / (1000 * 3600) = 0.2778 GPU-hours per million tokens. At utilisation 0.5 the same
    fleet needs 0.5556 effective GPU-hours.
    """
    raise NotImplementedError("analysis/cost.py: gpu_hours_per_million_tokens")


# ======================================================================================
# GIVEN (not stubs): the input-token and self-host-versus-API arithmetic. analysis.decide
# calls these. Read them: every number in the report's usage section comes from here.
#
# The self-host-versus-API question is answered with one quantity, GPU-seconds per request.
#   self-host: the GPU-seconds one request uses on your fleet at your operating point.
#   API:       the GPU-seconds of YOUR GPU that the API's rate for the same request is worth.
#              You express the API's rate for 1M tokens as "GPU-hours of your GPU", i.e. the
#              API's rate per 1M tokens divided by your GPU's hourly rate. That ratio is the
#              only API input; the repo never holds a price.
# breakeven_utilisation() then says how busy your GPUs must be for the two to tie.
# ======================================================================================
def gpu_seconds_per_request(req_per_s: float, num_gpus: int = 1, utilisation: float = 1.0) -> float:
    """GIVEN. GPU-seconds one request uses: num_gpus / req_per_s / utilisation.

    Example: one GPU serving 0.5 req/s at full utilisation uses 1 / 0.5 = 2 GPU-seconds per request.

    This charges every GPU second to the requests it served, input and output alike: prefill and
    decode share the same engine steps, so engine_sim and a real server cannot bill them apart."""
    if req_per_s <= 0 or not (0 < utilisation <= 1) or num_gpus < 1:
        raise ValueError("req_per_s > 0, 0 < utilisation <= 1 and num_gpus >= 1 are required")
    return num_gpus / req_per_s / utilisation


def blended_gpu_hours_per_million_tokens(gpu_seconds_per_req: float, mean_prompt_tokens: float,
                                         mean_output_tokens: float) -> float:
    """GIVEN. GPU-hours per million tokens when input and output tokens are counted together.

    Compare with gpu_hours_per_million_tokens (output tokens only). The blended figure is lower because
    the same GPU time is spread over more tokens. It weighs a prompt token like an output token,
    which a real API does not, so use it for orders of magnitude, not for a rate card."""
    total = mean_prompt_tokens + mean_output_tokens
    if total <= 0 or gpu_seconds_per_req < 0:
        raise ValueError("mean_prompt_tokens + mean_output_tokens must be positive and gpu seconds non-negative")
    return gpu_seconds_per_req / 3600.0 / total * 1e6


def api_gpu_seconds_per_request(mean_prompt_tokens: float, mean_output_tokens: float,
                                api_input_gpu_h_per_m: float, api_output_gpu_h_per_m: float) -> float:
    """GIVEN. The GPU-seconds of your GPU that the same request is worth through a hosted API.

    `api_*_gpu_h_per_m` is the API's rate for 1M input (or output) tokens divided by the hourly
    rate of the GPU you ran, so it reads "GPU-hours of my GPU that 1M API tokens are worth". Input
    and output are separate because APIs rate them separately. Example: 400 input and 100 output
    tokens at 1.0 and 3.0 GPU-hours per 1M -> (400 * 1.0 + 100 * 3.0) / 1e6 = 7e-4 GPU-hours = 2.52 GPU-seconds."""
    if min(api_input_gpu_h_per_m, api_output_gpu_h_per_m, mean_prompt_tokens, mean_output_tokens) < 0:
        raise ValueError("rates and token counts must be non-negative")
    return (mean_prompt_tokens * api_input_gpu_h_per_m + mean_output_tokens * api_output_gpu_h_per_m) / 1e6 * 3600.0


def breakeven_utilisation(gpu_seconds_per_request_full: float, api_gpu_seconds_per_request_value: float) -> float:
    """GIVEN. The fraction of the time your GPUs must be busy for self-hosting to use the same GPU
    time as the API is worth: gpu_seconds_per_request_at_full_utilisation / api_gpu_seconds_per_request.
    Above this fraction self-hosting wins. Above 1.0 it cannot win at this operating point, however
    busy the fleet is. It ignores engineering time, the quality gap between models and the API's rate limits."""
    if api_gpu_seconds_per_request_value <= 0:
        raise ValueError("api_gpu_seconds_per_request_value must be positive")
    return gpu_seconds_per_request_full / api_gpu_seconds_per_request_value
