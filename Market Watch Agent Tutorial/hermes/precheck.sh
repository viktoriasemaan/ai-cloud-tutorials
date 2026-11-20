#!/usr/bin/env bash
# Hermes cron pre-check: runs before every scheduled agent run (copy to ~/.hermes/profiles/<profile>/scripts/).
# Its stdout goes into the agent's prompt; a last line of {"wakeAgent": false} skips the model for that tick.
# Set MARKET_AGENT_REPO to this repo's folder. Set REPLAY_DATE=YYYY-MM-DD to replay a past day instead of live.
REPO="${MARKET_AGENT_REPO:?set MARKET_AGENT_REPO to the tutorial folder}"
if [ -n "${REPLAY_DATE:-}" ]; then
  exec python3 "$REPO/scripts/precheck.py" --date "$REPLAY_DATE"
fi
exec python3 "$REPO/scripts/precheck.py"
