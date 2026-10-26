#!/usr/bin/env bash
set -euo pipefail
cd /data/wio/Inference_Foundry
out=evidence/20260928_loop081_bound/run654
exec 9>/tmp/loop080_frontier.lock
flock -n 9
if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo 'service active' >&2; exit 75
fi
pids=()
for rank in 0 1 2 3 4 5 6 7; do
    docker exec -w /data/wio/Inference_Foundry vllm-ascend26-dsv4f-w4a8 \
        python3 scripts/loop081_light_event_budget_run654.py --device "$rank" --cycles 1025 --marks 5 \
        >"$out/event_budget_rank${rank}.json" 2>"$out/event_budget_rank${rank}.stderr" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ $status != 0 ]]; then exit "$status"; fi
python3 - <<'PY'
from pathlib import Path
import json
root=Path('evidence/20260928_loop081_bound/run654')
rows=[json.loads((root/f'event_budget_rank{r}.json').read_text()) for r in range(8)]
assert [x['device'] for x in rows]==list(range(8))
assert all(x['event_count']==5126 for x in rows)
(root/'all8_event_budget.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'all8':True,'max_create_ms':max(x['create_ms'] for x in rows),'max_first_record_ms':max(x['first_record_ms'] for x in rows),'max_query_and_elapsed_ms':max(x['query_and_elapsed_ms'] for x in rows)}))
PY
