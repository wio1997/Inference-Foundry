#!/usr/bin/env python3
"""Extract bounded frozen-source role mapping; no runtime or semantic completeness claim."""
from pathlib import Path
import hashlib,json
T=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent
files=['evidence/model-config.json','records.json','evidence/FUSION-SWEEP.md','sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py','sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py']
sha=lambda f:hashlib.sha256((T/f).read_bytes()).hexdigest()
c=json.loads((T/files[0]).read_text())['text_config'];r=json.loads((T/files[1]).read_text())['fusion_candidates'];
latest=lambda i, seq: next((v for v in reversed(seq) if v<=i),None)
rows=[]
for i in range(c['num_hidden_layers']):
 ratio=c['compress_ratios'][i]; kv=latest(i,c['kv_source_layer_ids']) if ratio else None; ix=latest(i,c['index_source_layer_ids']) if ratio else None
 f=['F01','F02','F03','F04','F05','F06','F07','F08','F13','F14','F15','F16','F20','F21','F22','F23']
 if i in c['index_source_layer_ids']: f += ['F09','F10','F12']
 if i in c['kv_source_layer_ids']: f += ['F11','F19']
 if i in c['kv_source_layer_ids'] and ratio==2: f += ['F17','F18']
 if i in c['engram_layer_ids']: f += ['F24','F25']
 rows.append({'layer':i,'compress_ratio':ratio,'kv_source':kv,'index_source':ix,'is_kv_source':i in c['kv_source_layer_ids'],'is_index_source':i in c['index_source_layer_ids'],'is_candidate_source':i==c['candidate_source_layer_id'],'uses_candidate_filter':ix is not None and ix>c['candidate_source_layer_id'],'engram':i in c['engram_layer_ids'],'applicable_review_ids':f,'limitations':'source role applicability only; not proof every branch executed or all primitive nodes covered'})
by_kv={str(s):[x['layer'] for x in rows if x['kv_source']==s] for s in c['kv_source_layer_ids']};by_idx={str(s):[x['layer'] for x in rows if x['index_source']==s] for s in c['index_source_layer_ids']}
registry=[]
for x in r:
 registry.append({'id':x['id'],'boundary':x.get('producer_consumer'),'imported_status':x['status'],'semantic_conditions':x.get('semantic_conditions'),'recorded_known_consumers':x.get('other_consumers'),'candidate_graph_path':x.get('candidate_graph_path'),'warning':'Prior imported candidate; status not independently re-proven. RULED_OUT_IN_SCOPE on F01/F04/F13 means already represented as fused in reviewed source, not physical impossibility.'})
out={'source_sha256':{x:sha(x) for x in files},'extractor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'method':'Frozen model config + documented _latest_source and build_layer_plan; F IDs are applicability index to existing boundary records, not completed semantic proof','layers':rows,'kv_source_to_layer_roles':by_kv,'index_source_to_layer_roles':by_idx,'candidate_source':c['candidate_source_layer_id'],'candidate_filter_index_sources':[x['layer'] for x in rows if x['is_index_source'] and x['uses_candidate_filter']],'previous_registry':registry,'unknown':['actual branch selection and per-forward token/slot lengths','alias, lowerings and complete consumers','numeric equivalence and ability to preserve overlap','trace runtime evidence per F candidate and all kernel ownership'], 'no_device_work':True}
(O/'frozen-layer-coverage.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('layers',len(rows),'kv users',by_kv,'index users',by_idx,'candidate filter index sources',out['candidate_filter_index_sources'])
