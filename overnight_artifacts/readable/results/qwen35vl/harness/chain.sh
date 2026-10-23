#!/bin/bash
# chain.sh <wait-log> <campaign-out> <specs...>: wait for CAMPAIGN DONE in wait-log (if given), then run.
W=$1; shift; OUT=$1; shift
if [ -n "$W" ] && [ "$W" != "-" ]; then until grep -q "CAMPAIGN DONE" "$W" 2>/dev/null; do sleep 10; done; fi
exec /root/work/mstar/.venv/bin/python /root/work/vlh/campaign.py "$OUT" "$@"
