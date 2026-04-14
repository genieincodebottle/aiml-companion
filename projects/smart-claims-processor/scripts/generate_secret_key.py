"""
Generate a cryptographically strong API_SECRET_KEY for JWT signing.

Why this exists
---------------
api/security.py signs JWT access tokens with `API_SECRET_KEY` from the env.
If you leave it at the default `dev-secret-change-me`, anyone who reads the
code can forge admin tokens. Use this script to produce a random key and
paste it into `.env`.

Usage
-----
    python scripts/generate_secret_key.py              # print + write to .env if missing
    python scripts/generate_secret_key.py --write      # force overwrite .env entry
    python scripts/generate_secret_key.py --print-only # just print, don't touch .env
    python scripts/generate_secret_key.py --bytes 48   # customize strength (default 48 bytes)

The key is base64-url encoded, ~64 characters for the default 48 bytes of
entropy - well above the 32-byte minimum recommended for HS256.
"""
from __future__ import annotations

import argparse
import re
import secrets
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
ENV_VAR = "API_SECRET_KEY"


def generate_key(num_bytes: int = 48) -> str:
    """Return a base64-url safe random string. 48 bytes -> ~64 chars."""
    return secrets.token_urlsafe(num_bytes)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _has_real_key(env_text: str) -> bool:
    """True if API_SECRET_KEY is set to something other than a placeholder."""
    match = re.search(rf"^{ENV_VAR}=(.*)$", env_text, flags=re.MULTILINE)
    if not match:
        return False
    value = match.group(1).strip().strip('"').strip("'")
    placeholders = {
        "",
        "change-me-to-a-long-random-string",
        "dev-secret-change-me",
        "your-secret-key-here",
    }
    return value not in placeholders


def _upsert(env_text: str, key: str) -> str:
    """Insert or replace the API_SECRET_KEY line, preserving the rest of .env."""
    line = f"{ENV_VAR}={key}"
    if re.search(rf"^{ENV_VAR}=.*$", env_text, flags=re.MULTILINE):
        return re.sub(rf"^{ENV_VAR}=.*$", line, env_text, flags=re.MULTILINE)
    if env_text and not env_text.endswith("\n"):
        env_text += "\n"
    return env_text + line + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate API_SECRET_KEY for JWT signing.")
    parser.add_argument("--bytes", type=int, default=48, help="entropy bytes (default 48)")
    parser.add_argument("--write", action="store_true", help="force overwrite existing .env value")
    parser.add_argument("--print-only", action="store_true", help="just print, don't touch .env")
    args = parser.parse_args()

    key = generate_key(args.bytes)
    print(f"\n  {ENV_VAR}={key}\n")

    if args.print_only:
        return 0

    # Make sure .env exists - bootstrap from .env.example if needed.
    if not ENV_PATH.exists():
        if ENV_EXAMPLE_PATH.exists():
            ENV_PATH.write_text(ENV_EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  Created {ENV_PATH.name} from .env.example")
        else:
            ENV_PATH.write_text("", encoding="utf-8")
            print(f"  Created empty {ENV_PATH.name}")

    env_text = _read(ENV_PATH)

    if _has_real_key(env_text) and not args.write:
        print(f"  {ENV_VAR} already set in {ENV_PATH.name}. Use --write to overwrite.")
        return 0

    ENV_PATH.write_text(_upsert(env_text, key), encoding="utf-8")
    print(f"  Wrote {ENV_VAR} to {ENV_PATH.name}. Restart the backend for it to take effect.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
