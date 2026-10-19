from pathlib import Path
import json,socket,subprocess,hashlib
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
cid='3caaa2a7856b78e27a692a909505d9d4a34969b68217c64869e5f4397bc25b5b'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running']
p='/vllm-workspace/vllm-ascend/vllm_ascend/ops/mla.py'
b=subprocess.check_output(['docker','exec',cid,'cat',p],timeout=20)
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False);(out/'mla.py').write_bytes(b)
probe=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live1-20261019/node33/indexer-k-graph-overlap-probe');rows=[]
for f in sorted(probe.glob('rank-*.jsonl')):
 frames=[json.loads(s) for s in f.read_text().splitlines()];calls=[z for v in frames for z in v['calls']];rows.append({'rank':frames[0]['rank'],'calls':len(calls),'selected':sum(z['selected'] for z in calls),'eligible':sum(z['eligible'] for z in calls),'max_entries':max(v['graph_entries'] for v in frames)})
assert len(rows)==16 and all(z['selected']==z['eligible']==z['max_entries']==0 for z in rows)
assert not(probe/'enable').exists()
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'completed':True,'source':p,'source_bytes':len(b),'source_SHA':hashlib.sha256(b).hexdigest(),'all16':rows,'verdict':'INVALID_INACTIVE_WRAPPER_BINDING','functional_numeric_only_native_fallback':True,'quality_formal_not_run':True})+'\nGLM53_RESULT_JSON_END')
