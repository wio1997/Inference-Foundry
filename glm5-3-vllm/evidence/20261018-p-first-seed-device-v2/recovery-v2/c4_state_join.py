from pathlib import Path
import collections,json,subprocess
root=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=root/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2';trace=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2/D-c4-diag/artifacts/adopt-trace');events=[json.loads(line) for p in trace.glob('*.jsonl') for line in p.read_text().splitlines() if line.strip()]
reports={}
for label in ('c4-control-repeat','c4-matched-pairs'):
 data=json.loads(next((e/label).rglob('metrics.json')).read_text());flat={**data,'rows':[r for b in data['bursts'] for r in b['rows']]};(e/(label+'-flattened-derived.json')).write_text(json.dumps(flat,indent=2)+'\n')
 p=subprocess.run(['python3',str(root/'glm5-3-vllm/runtime-patches/p-first-seed-v2/compare_work.py'),'--trace-dir',str(trace),'--client',str(e/(label+'-flattened-derived.json'))],capture_output=True,text=True,cwd=root);assert p.returncode==0,p.stderr;(e/(label+'-work.json')).write_text(p.stdout)
 rows=[]
 for bidx,burst in enumerate(data['bursts']):
  for client in burst['rows']:
   rid=client['D_chunks'][0]['id'];selected=sorted([(ev,row) for ev in events if ev['tp_rank']==0 for row in ev['requests'] if row['req_id'].startswith(rid+'-0-')],key=lambda x:x[0]['monotonic_ns']);assert selected
   params=client['P_response']['kv_transfer_params'];assert params['last_token_id']==client['token_ids'][0] and params['remote_num_prompt_tokens']==client['input_tokens']
   assert params['remote_request_id'].startswith(client['P_response']['id']+'-0-') and client['P_response']['id']==rid
   steps=[{'step':i,'base_host_computed':q['computed_before'],'host_query':q['host_query_token'],'host_history_count':q['output_history_count'],'host_history_head':q['initial_history_head'],'padded':ev['num_tokens_padded'],'graph_mode':ev['graph_mode'],'batch_req_ids':[z['req_id'] for z in ev['requests']]} for i,(ev,q) in enumerate(selected)]
   rows.append({'burst':bidx,'mode':client['mode'],'N':client['input_tokens'],'header_id':client['header_request_id'],'public_id':rid,'P_internal_id':params['remote_request_id'],'D_internal_id':selected[0][1]['req_id'],'P_confirmed_first_id':params['last_token_id'],'prompt_sha256':client['prompt_sha256'],'C1_first_difference':client['first_C1_difference'],'first_3_ids':client['token_ids'][:3],'steps':steps})
 report={'rows':rows,'initial_all16_position_query_history_checks':'pass (see adjacent -work.json)','P_to_D_identity_and_confirmed_first_matches_all':True,'scope':'Initial host Request/DTO mapping and full-call dispatch metadata, not full cache equality, actual GPU draft/input readback or a proof of numeric noise.'};(e/(label+'-state-join.json')).write_text(json.dumps(report,indent=2)+'\n');reports[label]=report
print(json.dumps({'all_initial_checks_pass':True,'joined_requests':sum(len(v['rows']) for v in reports.values())}))
for label,report in reports.items():
 examples=[v for v in report['rows'] if v['C1_first_difference'] is not None]
 for row in examples[:2]:print(json.dumps({'label':label,'mode':row['mode'],'N':row['N'],'first_C1_diff':row['C1_first_difference'],'first_steps':row['steps'][:4]}))
