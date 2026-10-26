#!/usr/bin/env bash
# Full-payload sha256 verification of local weights vs node1 source manifest (193 files).
# Chunks run with cwd = weights dir (manifest paths are ./relative to it). Any failure -> nonzero.
set -u
T=/data/wio/sglang-glm53-33-41-20261013
W=/data/wio/glm53-w8a8-tuning-local
SD=$T/logs/shaverify
mkdir -p "$SD"
rm -f "$SD"/chunk-* "$SD"/summary.txt
split -n l/8 -d -a 1 "$T/evidence/src-weights.sha256" "$SD/chunk-"
cd "$W" || exit 9
export T SD
ls "$SD"/chunk-[0-9] | xargs -P 8 -I{} sh -c 'sha256sum -c "{}" > "{}.out" 2>&1; echo "chunk {} rc=$?" >> "$SD/summary.txt"'
fails=$(grep -c 'rc=[1-9]' "$SD/summary.txt" 2>/dev/null || true)
okc=$(cat "$SD"/chunk-*.out | grep -c ': OK$' || true)
other=$(cat "$SD"/chunk-*.out | grep -v ': OK$' | head -20)
echo "chunks_failed=$fails ok_lines=$okc"
[ -n "$other" ] && echo "$other"
if [ "$okc" = "193" ] && [ "$fails" = "0" ]; then echo SHA256_RESULT=PASS_ALL_193; exit 0; else echo SHA256_RESULT=FAIL; exit 1; fi
