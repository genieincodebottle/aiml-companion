"""Shared configuration for all agents."""

import os
import logging
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file (searches current dir, then parent dirs)
# ---------------------------------------------------------------------------

# Find .env relative to this file's location (project root)
_project_root = Path(__file__).resolve().parent.parent
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    load_dotenv()  # fallback: search from cwd upwards

# ---------------------------------------------------------------------------
# Environment config (override via .env or shell exports)
# ---------------------------------------------------------------------------

# LLM Provider: "bedrock" (default), "groq", or "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Provider-specific settings
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.6-flash")

# Resolve the active model ID based on provider
if LLM_PROVIDER == "groq":
    DEFAULT_MODEL_ID = GROQ_MODEL_ID
elif LLM_PROVIDER == "gemini":
    DEFAULT_MODEL_ID = GEMINI_MODEL_ID
else:
    DEFAULT_MODEL_ID = BEDROCK_MODEL_ID

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
MAX_RESEARCH_RESULTS = int(os.getenv("MAX_RESEARCH_RESULTS", "5"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("multi-agent")

# Log active provider on import
logger.info("LLM Provider: %s | Model: %s", LLM_PROVIDER, DEFAULT_MODEL_ID)


# ---------------------------------------------------------------------------
# AWS clients (cached - one per process, only created for bedrock provider)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_bedrock_client():
    """Return a Bedrock Runtime client for the converse() API."""
    import boto3
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


@lru_cache(maxsize=1)
def get_bedrock_agent_client():
    """Return a Bedrock Agent Runtime client (for AgentCore invocations)."""
    import boto3
    return boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_groq_client():
    """Return a Groq client."""
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError(
            "\n" + "=" * 60 + "\n"
            "  GROQ PACKAGE NOT INSTALLED\n"
            + "=" * 60 + "\n"
            "  Run: uv pip install groq\n"
            + "=" * 60 + "\n"
        )
    if not GROQ_API_KEY:
        raise RuntimeError(
            "\n" + "=" * 60 + "\n"
            "  GROQ_API_KEY NOT SET\n"
            + "=" * 60 + "\n"
            "  1. Get a free API key at https://console.groq.com/keys\n"
            "  2. Add to .env: GROQ_API_KEY=gsk_your_key_here\n"
            + "=" * 60 + "\n"
        )
    return Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_gemini_client():
    """Return a configured Google GenAI client."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError(
            "\n" + "=" * 60 + "\n"
            "  GOOGLE-GENAI PACKAGE NOT INSTALLED\n"
            + "=" * 60 + "\n"
            "  Run: uv pip install google-genai\n"
            + "=" * 60 + "\n"
        )
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "\n" + "=" * 60 + "\n"
            "  GEMINI_API_KEY NOT SET\n"
            + "=" * 60 + "\n"
            "  1. Get a free API key at https://aistudio.google.com/apikey\n"
            "  2. Add to .env: GEMINI_API_KEY=your_key_here\n"
            + "=" * 60 + "\n"
        )
    return genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Unified LLM call - routes to the active provider
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_message: str,
    model_id: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    retries: int = 2,
) -> str:
    """Send a single-turn message to the configured LLM provider.

    Routes to Bedrock, Groq, or Gemini based on the LLM_PROVIDER env var.
    Each agent wraps this with its own system prompt.
    """
    if LLM_PROVIDER == "groq":
        return _call_groq(system_prompt, user_message, model_id, max_tokens, temperature, retries)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini(system_prompt, user_message, model_id, max_tokens, temperature, retries)
    else:
        return _call_bedrock(system_prompt, user_message, model_id, max_tokens, temperature, retries)


def call_llm_with_history(
    system_prompt: str,
    messages: list[dict],
    model_id: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """Send a multi-turn conversation to the configured LLM provider.

    For Bedrock: `messages` should be [{\"role\": \"user\"|\"assistant\", \"content\": [{\"text\": \"...\"}]}]
    For Groq/Gemini: automatically converts from Bedrock format.
    """
    if LLM_PROVIDER == "groq":
        return _call_groq_with_history(system_prompt, messages, model_id, max_tokens, temperature)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini_with_history(system_prompt, messages, model_id, max_tokens, temperature)
    else:
        return _call_bedrock_with_history(system_prompt, messages, model_id, max_tokens, temperature)


# ---------------------------------------------------------------------------
# Bedrock implementation
# ---------------------------------------------------------------------------

def _call_bedrock(system_prompt, user_message, model_id, max_tokens, temperature, retries):
    import time
    from botocore.exceptions import ClientError

    client = get_bedrock_client()
    model = model_id or DEFAULT_MODEL_ID

    for attempt in range(retries + 1):
        try:
            response = client.converse(
                modelId=model,
                system=[{"text": system_prompt}],
                messages=[
                    {"role": "user", "content": [{"text": user_message}]},
                ],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                },
            )
            return response["output"]["message"]["content"][0]["text"]

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_msg = e.response["Error"]["Message"]

            if "AccessDeniedException" in error_code:
                if "INVALID_PAYMENT_INSTRUMENT" in error_msg:
                    raise RuntimeError(
                        f"\n{'=' * 60}\n"
                        f"  AWS BILLING ISSUE\n"
                        f"{'=' * 60}\n"
                        f"  Your AWS account needs a valid payment method.\n"
                        f"  Go to: https://console.aws.amazon.com/billing/\n"
                        f"  -> Payment Methods -> Add a credit/debit card\n"
                        f"  Then wait 2 minutes and try again.\n"
                        f"{'=' * 60}\n"
                    ) from e
                raise RuntimeError(
                    f"\n{'=' * 60}\n"
                    f"  MODEL ACCESS DENIED\n"
                    f"{'=' * 60}\n"
                    f"  Model: {model}\n"
                    f"  Fix: Go to https://console.aws.amazon.com/bedrock/home#/modelaccess\n"
                    f"  and enable 'Anthropic Claude Sonnet 4.6'.\n"
                    f"\n"
                    f"  Or use a cheaper model in .env:\n"
                    f"  BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0\n"
                    f"\n"
                    f"  Or switch to a free provider:\n"
                    f"  LLM_PROVIDER=groq\n"
                    f"  LLM_PROVIDER=gemini\n"
                    f"{'=' * 60}\n"
                ) from e

            if "ValidationException" in error_code:
                raise RuntimeError(
                    f"\n{'=' * 60}\n"
                    f"  INVALID MODEL ID\n"
                    f"{'=' * 60}\n"
                    f"  Model '{model}' is not valid in region '{AWS_REGION}'.\n"
                    f"  Check available models at:\n"
                    f"  https://console.aws.amazon.com/bedrock/home#/modelaccess\n"
                    f"\n"
                    f"  Try: BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0\n"
                    f"{'=' * 60}\n"
                ) from e

            if "ThrottlingException" in error_code and attempt < retries:
                wait = 2 ** attempt
                logger.warning("Rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, retries)
                time.sleep(wait)
                continue

            raise


def _call_bedrock_with_history(system_prompt, messages, model_id, max_tokens, temperature):
    client = get_bedrock_client()
    model = model_id or DEFAULT_MODEL_ID

    response = client.converse(
        modelId=model,
        system=[{"text": system_prompt}],
        messages=messages,
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )

    return response["output"]["message"]["content"][0]["text"]


# ---------------------------------------------------------------------------
# Groq implementation
# ---------------------------------------------------------------------------

def _call_groq(system_prompt, user_message, model_id, max_tokens, temperature, retries):
    import time

    client = get_groq_client()
    model = model_id or DEFAULT_MODEL_ID

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str and attempt < retries:
                wait = 2 ** attempt
                logger.warning("Groq rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, retries)
                time.sleep(wait)
                continue
            raise


def _call_groq_with_history(system_prompt, messages, model_id, max_tokens, temperature):
    client = get_groq_client()
    model = model_id or DEFAULT_MODEL_ID

    # Convert Bedrock message format to OpenAI/Groq format
    groq_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        role = msg["role"]
        text = msg["content"][0]["text"] if isinstance(msg["content"], list) else msg["content"]
        groq_messages.append({"role": role, "content": text})

    response = client.chat.completions.create(
        model=model,
        messages=groq_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------

def _call_gemini(system_prompt, user_message, model_id, max_tokens, temperature, retries):
    import time
    from google.genai import types

    client = get_gemini_client()
    model = model_id or DEFAULT_MODEL_ID

    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            return response.text

        except Exception as e:
            error_str = str(e).lower()
            if ("rate" in error_str or "quota" in error_str) and attempt < retries:
                wait = 2 ** attempt
                logger.warning("Gemini rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, retries)
                time.sleep(wait)
                continue
            raise


def _call_gemini_with_history(system_prompt, messages, model_id, max_tokens, temperature):
    from google.genai import types

    client = get_gemini_client()
    model = model_id or DEFAULT_MODEL_ID

    # Convert Bedrock message format to Gemini format
    gemini_contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        text = msg["content"][0]["text"] if isinstance(msg["content"], list) else msg["content"]
        gemini_contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

    response = client.models.generate_content(
        model=model,
        contents=gemini_contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.text
