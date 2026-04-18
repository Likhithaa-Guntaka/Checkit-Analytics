#!/usr/bin/env bash
# Checkit Analytics RAG — incremental update runner
#
# Schedule with cron (every Monday at 9am):
#   crontab -e
#   Add: 0 9 * * 1 /path/to/run_update.sh
#
# Run manually:
#   bash run_update.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

python3 scheduler.py
