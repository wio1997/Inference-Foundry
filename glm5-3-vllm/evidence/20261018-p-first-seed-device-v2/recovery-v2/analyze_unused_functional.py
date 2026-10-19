from pathlib import Path
import json
r=Path('/Users/wio/work/Inference-Foundry-execution-plan/glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2');base=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2/D-unused-functional/artifacts');records=[json.loads(x) for f in (base/'divergence-probe').glob('*.jsonl') for x in f.read_text().splitlines() if x.strip()];decisions=[x for x in records if x['kind']=='unused_draft_decision'];skips=[x for x in records if x['kind']=='native_skip_drafting_executed'];policies=[x for x in records if x['kind']=='shared_overlap_actual_calls'];traces=[json.loads(x) for f in (base/'adopt-trace').glob('*.jsonl') for x in f.read_text().splitlines() if x.strip()];quality=json.loads(next((r/'unused-single-answer-quality').rglob('metrics.json')).read_text());assert quality['completed'] and len(quality['rows'])==36;rows=[];long=[]
for client in quality['rows']:
 public=client['response']['id'];rank_steps={}
 for rank in range(16):
  ds=sorted([x for x in decisions if x['tp_rank']==rank and any(rid.startswith(public+'-0-') for rid in x['request_ids'])],key=lambda x:x['monotonic_ns']);ss=[x for x in skips if x['tp_rank']==rank and any(rid.startswith(public+'-0-') for rid in x['request_ids'])];assert ds
  for d in ds:
   assert d['eligible_all_max1_D_TP16_DP1_PP1'] and d['max_tokens'] and set(d['max_tokens'])=={1};assert d['selected_skip']==(client['mode']=='draft-skip' and d['native_fits'])
  assert len(ss)==sum(not d['native_fits'] or d['selected_skip'] for d in ds)
  steps=sorted([(event,q) for event in traces if event['tp_rank']==rank for q in event['requests'] if q['req_id'].startswith(public+'-0-')],key=lambda x:x[0]['monotonic_ns']);assert len(steps)==len(ds)
  rank_steps[rank]=[{'computed':q['computed_before'],'scheduled':q['scheduled_tokens'],'spec_ids':q['initial_spec_ids']} for _,q in steps]
 if client['N']>1024:
  assert all(len(steps)==2 and steps[0]['computed']==0 and steps[0]['scheduled']==1024 and steps[1]['computed']==1024 for steps in rank_steps.values())
  long.append({'N':client['N'],'mode':client['mode'],'all16_two_target_chunks':True,'rank0_steps':rank_steps[0]})
 rows.append({'public_id':public,'N':client['N'],'mode':client['mode'],'cohort':client['cohort'],'whole_answer':client['response']['choices'][0]['text'],'whole_answer_pass':client['whole_answer_pass'],'all16_native_skip_decisions_match_mode':True,'rank0_steps':rank_steps[0]})
guards=[]
for label in ['unused-guard-max2','unused-guard-max64']:
 data=json.loads(next((r/label).rglob('metrics.json')).read_text());assert data['completed'] and len(data['rows'])==6
 for client in data['rows']:
  public=client['chunks'][0]['id'];count=0
  for rank in range(16):
   ds=[x for x in decisions if x['tp_rank']==rank and any(rid.startswith(public+'-0-') for rid in x['request_ids'])];assert ds
   assert all(d['requested_skip'] and not d['eligible_all_max1_D_TP16_DP1_PP1'] and not d['selected_skip'] and all(value in (2,64) for value in d['max_tokens']) for d in ds);count+=len(ds)
  assert client['exact_pass'] and client['text_matches_control'];guards.append({'label':label,'public_id':public,'mode':client['mode'],'N':client['input_tokens'],'all16_declined_skip':True,'decision_records':count,'exact_full_ids_text_usage':True})
report={'quality_requests':len(rows),'all_whole_numeric_answers_pass':True,'quality_rows':rows,'long_input_chunk_proof':long,'protected_requests':len(guards),'all16_protected_max2_max64_decline_skip':True,'guard_rows':guards,'scope':'Whole one-token task outputs and actual all16 branches; long native D chunking, not long PD or complete general model quality. Max2/64 original path preserved with full frozen fixture outputs. No timing conclusion.'};(r/'unused-functional-summary-v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'quality_requests':len(rows),'whole_answers':True,'long_chunks':long,'protected_requests':len(guards),'all16_guard_declines':True}))
