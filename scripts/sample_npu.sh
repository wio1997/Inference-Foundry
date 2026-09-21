#!/usr/bin/env bash
set -eo pipefail
if [ "$#" -ne 2 ]; then echo "usage: sample_npu.sh <seconds> <output>" >&2; exit 2; fi
DURATION="$1"
OUTPUT="$2"
END=$(( $(date +%s) + DURATION ))
while [ "$(date +%s)" -lt "$END" ]; do
  date -u '+%Y-%m-%dT%H:%M:%SZ' >> "$OUTPUT"
  npu-smi info >> "$OUTPUT" 2>&1
  sleep 5
done
