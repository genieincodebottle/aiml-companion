#!/usr/bin/env bash
# Check that the gateway denies an account that is not on the allowlist.
#
# Why this is a script and not a note: from your own account a working
# allowlist and a broken one look identical. The only way to know is to try
# from outside, and the only way to keep knowing is to make it repeatable.
#
# Usage:
#   deploy/verify-allowlist.sh <bot-url-or-handle>
#
# This does not automate the messaging platform. It walks you through the
# check and records the result, so a green deploy has evidence behind it.
set -euo pipefail

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "usage: $0 <bot-handle-or-url>" >&2
  exit 2
fi

echo "Allowlist check for: $TARGET"
echo
echo "1. Confirm GATEWAY_ALLOW_ALL_USERS is not set anywhere:"
if grep -rIn "GATEWAY_ALLOW_ALL_USERS" "${HERMES_HOME:-$HOME/.hermes}" 2>/dev/null; then
  echo "   FOUND. If any of those are 'true', anyone who finds the bot has a shell." >&2
  exit 1
else
  echo "   not set  (good)"
fi

echo
echo "2. Confirm an explicit allowlist exists in config.yaml:"
CFG="${HERMES_HOME:-$HOME/.hermes}/config.yaml"
if [[ -f "$CFG" ]] && grep -qiE "allow(ed)?_users|allowlist" "$CFG"; then
  echo "   present  (good)"
else
  echo "   NOT FOUND in $CFG" >&2
  echo "   Without one you are relying on the platform to keep strangers out." >&2
  exit 1
fi

echo
echo "3. Confirm secrets are not world-readable:"
ENVF="${HERMES_HOME:-$HOME/.hermes}/.env"
if [[ -f "$ENVF" ]]; then
  MODE=$(stat -c '%a' "$ENVF" 2>/dev/null || stat -f '%A' "$ENVF")
  if [[ "$MODE" == "600" ]]; then
    echo "   $ENVF is $MODE  (good)"
  else
    echo "   $ENVF is $MODE, expected 600" >&2
    exit 1
  fi
fi

echo
echo "4. MANUAL, and the one that actually proves it:"
echo "   Message the bot from an account that is NOT on the allowlist."
echo "   Silence is the correct result. Any reply means layer 1 is off."
echo
read -r -p "   Did the unauthorised account get a reply? [y/N] " ans
case "$ans" in
  [yY]*) echo "   FAIL: the gateway answered a stranger." >&2; exit 1 ;;
  *)     echo "   PASS" ;;
esac
