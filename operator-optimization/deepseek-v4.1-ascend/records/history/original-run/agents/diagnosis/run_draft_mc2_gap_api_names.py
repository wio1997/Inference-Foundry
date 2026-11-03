from pathlib import Path
import base64,hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/query_draft_mc2_gap_api_names.py';P=T/'agents/diagnosis/draft-mc2-gap-host-intervals.json';SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
D='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
b64=base64.b64encode(P.read_bytes()).decode('ascii')
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',S.read_text(),'--trace-root',D,'--prior-b64',b64])],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('read-only query failed: '+r.stderr)
x=json.loads(r.stdout);x['script_sha256']=hashlib.sha256(S.read_bytes()).hexdigest();x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();x['no_device_work']=True
out=T/'agents/diagnosis/draft-mc2-gap-api-names.json'
if out.exists():raise RuntimeError('output exists')
out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'rows':len(x['rows']),'whitelist_names_found':sum(not a['name_withheld'] for a in x['rows'])}))
