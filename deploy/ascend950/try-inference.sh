#!/usr/bin/env bash
set -euo pipefail
curl --fail-with-body -sS http://127.0.0.1:18110/readyz
curl --fail-with-body -sS http://127.0.0.1:18110/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"只回答数字：1+1等于几？"}],"thinking":{"type":"disabled"},"max_tokens":32,"temperature":0,"stream":false}'
