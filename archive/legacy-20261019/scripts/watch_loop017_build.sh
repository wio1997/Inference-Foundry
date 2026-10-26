#!/usr/bin/env bash
set -uo pipefail
root=/data/wio/Inference_Foundry
evidence=$root/evidence/20260920_loop017_compressor
build_pid=2298389
while kill -0 "$build_pid" 2>/dev/null; do
  sleep 30
done
prefix=$(cat "$evidence/candidate_prefix.txt")
vendor=$prefix/vendors/custom_transformer
if [[ ! -s "$vendor/op_api/lib/libcust_opapi.so" ]]; then
  echo "package missing after build PID ended" > "$evidence/candidate_run.log"
  echo 90 > "$evidence/candidate_exit_code.txt"
  exit 90
fi
cd "$root"
bash scripts/run_loop017_isolated_bench.sh > "$evidence/candidate_run.log" 2>&1
status=$?
echo "$status" > "$evidence/candidate_exit_code.txt"
exit "$status"
