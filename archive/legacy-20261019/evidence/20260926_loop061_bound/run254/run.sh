#!/usr/bin/env bash
set -u
cd /data/wio/Inference_Foundry
out=evidence/20260926_loop061_bound/run254
for mode in 0 1; do
  while read -r name exe; do
    docker exec dsv4ab bash -lc "cd /tmp/extreme_hccl_test_cann910 && HCCL_BUFFSIZE=256 timeout -s INT -k 5s 90s mpirun --allow-run-as-root -x HCCL_BUFFSIZE -n 8 ./bin/$exe -b 98304 -e 98304 -i 1024 -p 8 -d bfp16 -n 50 -w 20 -t $mode" > "$out/$name.t$mode.log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$out/$name.t$mode.exit"
    printf '%s_t%s %s\n' "$name" "$mode" "$rc"
    if test "$rc" -ne 0; then tail -8 "$out/$name.t$mode.log"; fi
  done <<'CASES'
ag all_gather_test
rs reduce_scatter_test
a2a alltoall_test
CASES
done
npu-smi info | grep -c 'No running processes found' > "$out/idle_count.txt"
