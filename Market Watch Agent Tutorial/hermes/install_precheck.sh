#!/usr/bin/env bash
# Copy the pre-check into a Hermes profile's scripts folder, pointing it at this repo.
#   bash hermes/install_precheck.sh marketwatch                 live: precheck.sh
#   bash hermes/install_precheck.sh marketwatch 2026-10-29      replay a past day: precheck_2026-10-29.sh
set -euo pipefail
PROFILE="${1:-marketwatch}"; DAY="${2:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/.hermes/profiles/$PROFILE/scripts"
mkdir -p "$DEST"
NAME="precheck${DAY:+_$DAY}.sh"
sed "s|\${MARKET_AGENT_REPO:?set MARKET_AGENT_REPO to the tutorial folder}|$REPO|" "$REPO/hermes/precheck.sh" > "$DEST/$NAME"
[ -n "$DAY" ] && sed -i "s|^if \[ -n \"\${REPLAY_DATE:-}\" \]|REPLAY_DATE=$DAY\nif [ -n \"\${REPLAY_DATE:-}\" ]|" "$DEST/$NAME"
chmod +x "$DEST/$NAME"
echo "installed $DEST/$NAME (repo: $REPO)"
