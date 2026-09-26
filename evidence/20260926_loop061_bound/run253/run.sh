#!/usr/bin/env bash
set -u
cd /data/wio/Inference_Foundry
out=evidence/20260926_loop061_bound/run253
while read -r name exe bytes dtype; do
  test -n "$name" || continue
  docker exec dsv4ab bash -lc "cd /tmp/extreme_hccl_test_cann910 && timeout -s INT -k 5s 90s mpirun --allow-run-as-root -n 8 ./bin/$exe -b $bytes -e $bytes -i 1024 -p 8 -d $dtype -n 50 -w 20" > "$out/$name.log" 2>&1
  rc=$?
  printf '%s\n' "$rc" > "$out/$name.exit"
  printf '%s %s\n' "$name" "$rc"
  if test "$rc" -ne 0; then
    tail -10 "$out/$name.log"
  fi
done <<'CASES'
ag_12k_fp32 all_gather_test 12288 fp32
ag_96k_bf16 all_gather_test 98304 bfp16
ag_384k_bf16 all_gather_test 393216 bfp16
ag_3102720_bf16 all_gather_test 3102720 bfp16
rs_96k_bf16 reduce_scatter_test 98304 bfp16
a2a_96k_bf16 alltoall_test 98304 bfp16
CASES
npu-smi info | grep -c 'No running processes found' > "$out/idle_count.txt"
