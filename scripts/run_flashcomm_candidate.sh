#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT="${CANDIDATE_OUT:-$ROOT/evidence/20260920_sp_dsa_off}"
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
mkdir -p "$OUT"
for i in $(seq 1 180); do
    if [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models || true)" = 200 ]; then
        date -u '+ready=%Y-%m-%dT%H:%M:%SZ' | tee "$OUT/ready.txt"
        break
    fi
    if [ "$i" = 180 ]; then echo 'service readiness timeout' >&2; exit 1; fi
    sleep 10
done
python3 - "$OUT" <<'PY'
import json, pathlib, subprocess, sys
out=pathlib.Path(sys.argv[1])
ps=subprocess.check_output(['docker','exec','dsv4ab','ps','-eo','pid,args'],text=True)
lines=[x for x in ps.splitlines() if 'vllm serve ' in x and 'grep' not in x]
(out/'serve_command.txt').write_text('\n'.join(lines)+'\n')
assert len(lines)==1 and '"enable_flashcomm1":false' in lines[0] and '"enable_dsa_cp":false' in lines[0], lines
PY
docker exec dsv4ab python3 "$ROOT/scripts/golden.py" --dataset "$DATASET" --out "$OUT/golden4.json" > "$OUT/golden.log" 2>&1
python3 - "$ROOT" "$OUT" <<'PY'
import json,os,pathlib,sys
root,out=map(pathlib.Path,sys.argv[1:])
ref=json.loads((root/'evidence/20260920_baseline/golden4.json').read_text())
cur=json.loads((out/'golden4.json').read_text())
checks=[{'index':r['index'],'prompt_equal':r['prompt_sha256']==c['prompt_sha256'],
         'output_equal':r['output_sha256']==c['output_sha256'],
         'tokens_equal':r['completion_tokens']==c['completion_tokens']}
        for r,c in zip(ref,cur)]
result={'pass':len(ref)==len(cur)==4 and all(all(v for k,v in x.items() if k!='index') for x in checks),'checks':checks}
(out/'golden_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if not result['pass'] and os.environ.get('ALLOW_NONDETERMINISTIC_GOLDEN') != '1': raise SystemExit(1)
PY
docker exec dsv4ab python3 "$ROOT/scripts/check_functional.py" --golden "$OUT/golden4.json" --out "$OUT/functional_check.json" > "$OUT/functional.log" 2>&1
curl -sS --fail http://127.0.0.1:8080/metrics > "$OUT/metrics_before.txt"
date -u '+warmup_start=%Y-%m-%dT%H:%M:%SZ' > "$OUT/times.txt"
docker exec dsv4ab python3 "$ROOT/scripts/bench.py" --dataset "$DATASET" --out "$OUT/warmup.json" --limit 48 --concurrency 12 --max-tokens 1024 > "$OUT/warmup.log" 2>&1
date -u '+warmup_end=%Y-%m-%dT%H:%M:%SZ' >> "$OUT/times.txt"
for n in 1 2 3; do
    date -u "+bench${n}_start=%Y-%m-%dT%H:%M:%SZ" >> "$OUT/times.txt"
    docker exec dsv4ab python3 "$ROOT/scripts/bench.py" --dataset "$DATASET" --out "$OUT/bench48_${n}.json" --limit 48 --concurrency 12 --max-tokens 1024 > "$OUT/bench48_${n}.log" 2>&1
    date -u "+bench${n}_end=%Y-%m-%dT%H:%M:%SZ" >> "$OUT/times.txt"
done
curl -sS --fail http://127.0.0.1:8080/metrics > "$OUT/metrics_after.txt"
python3 - "$OUT" <<'PY'
import json,pathlib,statistics,sys
out=pathlib.Path(sys.argv[1]); rows=[]
for i in (1,2,3):
    x=json.loads((out/f'bench48_{i}.json').read_text())['summary']
    assert x['success']==48 and x['fail']==0 and x['n']==48, x
    rows.append(x)
result={'runs':rows,'median_output_tps':statistics.median(x['output_tps'] for x in rows),
        'median_tpot_ms':statistics.median(x['tpot_ms_mean'] for x in rows),
        'median_ttft_ms':statistics.median(x['ttft_ms_mean'] for x in rows)}
(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
date -u '+done=%Y-%m-%dT%H:%M:%SZ' >> "$OUT/times.txt"
