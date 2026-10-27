"""Bind saved SG output IDs by RID/step; retain zero-output tail and unknown edges."""
from pathlib import Path
import json,hashlib,collections,csv,time
D=Path(__file__).resolve().parent;R=D.parent
SNAP=R/'sg-replay-fences-gen67-root/remote-receipts/snapshot'
LABEL='observer-lineage/sglang/lineage-sg-replay-fences-g67'
WAVE=R/'executor-run-gen27/received/waves/full-lineage-replay-fences-g67-sglang-in512-c1-o32'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):return [dict(json.loads(l),_line=i+1)for i,l in enumerate(p.read_text().splitlines())]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def request_rid(front,cid):
 links=[x for x in front if x.get('event')=='sg_request_id_link'and x.get('external_request_id')==cid]
 assert len(links)==1 and isinstance(links[0].get('rid'),str),'missing_or_ambiguous_request_link'
 return links[0]['rid'],links[0]
def commit_tokens(rows,rid,expected=32):
 commits=[];zeros=[]
 for row in rows:
  if row.get('event')!='scheduler_commit':continue
  entries=[x for x in row.get('token_commit',[])if x['rid']==rid]
  if not entries:continue
  assert len(entries)==1 and [q['rid']for q in row['requests']]==[rid],'mixed_or_duplicate_request'
  t=entries[0];entry={'step':row['forward_iter'],'mode':row['forward_mode'],'line':row['_line'],'pid':row['pid'],'rank':row['rank_id'],**t}
  delta=t['model_sample_token_delta'];total=t['committed_output_ids_total']
  if delta==1:
   assert total==len(commits)+1 and isinstance(t['last_token_id'],int),'noncontiguous_output_total'
   commits.append(entry)
  elif delta==0:
   assert total==len(commits)and commits and t['last_token_id']==commits[-1]['last_token_id'],'invalid_zero_delta'
   zeros.append(entry)
  else:raise AssertionError('cannot_reconstruct_multi_or_unknown_delta')
 assert len(commits)==expected and len(zeros)==1 and zeros[0]['step']>commits[-1]['step'],'incomplete_or_unexpected_tail'
 allsteps=[x['step']for x in commits+zeros]
 assert len(set(allsteps))==len(allsteps),'duplicate_commit_step'
 return commits,zeros

def join_graph_tokens(rows,rid,commits,zeros,fences):
 pairs=collections.defaultdict(list)
 for row in rows:
  if row.get('event')!='sg_replay_outer_event':continue
  work=json.loads(row['work_key'])
  if not any(x.get('rid')==rid for x in work.get('requests',[])):continue
  assert [x['rid']for x in work['requests']]==[rid],'mixed_graph_work'
  pairs[row['invocation']].append((row,work))
 bystep={};native={f['invocation']:f for f in fences};assert len(native)==len(fences)
 for inv,values in pairs.items():
  assert len(values)==2 and sorted(x['side']for x,w in values)==['end','start'],'bad_outer_pair'
  a,work=values[0];assert all(w==work and r['pid']==a['pid']and r['rank_id']==a['rank_id']for r,w in values),'conflicting_work_key'
  step=work['step'];assert step not in bystep and inv in native,'nonunique_or_unresolved_native_invocation'
  f=native[inv];assert len(f['layer_rows'])==64 and {x['layer_idx']for x in f['layer_rows']}==set(range(64)),'missing_native_layers'
  bystep[step]={'invocation':inv,'outer_event_lines':sorted(r['_line']for r,w in values),'native_fence':f,'pid':a['pid'],'rank':a['rank_id']}
 outputs=[]
 for t in commits+zeros:
  step=t['step'];g=bystep.get(step)
  if t['mode']=='1':assert g is None,'prefill_unexpected_graph'
  else:
   assert t['mode']=='2'and g,'decode_missing_explicit_graph_work'
   assert g['pid']==t['pid']and g['rank']==t['rank'],'graph_commit_process_mismatch'
  outputs.append({**t,'graph_invocation':g['invocation']if g else None,'outer_event_lines':g['outer_event_lines']if g else [],'native_layer_rows':len(g['native_fence']['layer_rows'])if g else 0})
 assert set(bystep)=={t['step']for t in outputs if t['mode']=='2'},'uncommitted_graph_work'
 return outputs

def decode_selected(ids,evidence):
 tok=evidence['files']['tokenizer.json'];assert tok['decoder']['type']=='ByteLevel'
 # GPT byte-level alphabet inverse; no merges/retokenization are involved.
 bs=list(range(ord('!'),ord('~')+1))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
 for b in range(256):
  if b not in bs:bs.append(b);cs.append(256+n);n+=1
 inverse={chr(c):b for b,c in zip(bs,cs)}
 added={a['id']:a for a in tok['added_tokens_relevant']};parts=[]
 for i in ids:
  piece=tok['id_to_token'][str(i)]
  if i in added:
   if added[i]['special']:continue
   parts.append(piece.encode())
  else:parts.append(bytes(inverse[c]for c in piece))
 return b''.join(parts).decode('utf-8')

def main():
 inputs=[]
 def pin(p,expected=None):
  p=Path(p);h=sha(p)
  if expected is not None:assert h==expected,('SHA_mismatch',str(p))
  inputs.append({'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':h});return p
 quality_pin=read(pin(D/'quality-comparison-pin.json'))['quality_sha256']
 source_contract=read(pin(D/'source-contract.json'));pin(D/'installed-source/download.json',source_contract['source_receipt_sha256'])
 for row in source_contract['files']:pin(R/row['path'],row['sha256'])
 gen27=read(R/'executor-run-gen27/received/download-with-logs.json');gen27refs={x['relative_path']:x for x in gen27['files']};pin(R/'executor-run-gen27/received/download-with-logs.json')
 for rel in ['waves/full-lineage-replay-fences-g67-sglang-in512-c1-o32/client/manifest.json','waves/full-lineage-replay-fences-g67-sglang-in512-c1-o32/client/request-0000/response.raw','waves/full-lineage-fx-model-g81-exec-g3-vllm-in512-c1-o32/client/request-0000/response.raw']:
  pin(R/'executor-run-gen27/received'/rel,gen27refs[rel]['sha256'])
 receipt=read(SNAP/'snapshot-download.json');refs={x['relative_path']:x for x in receipt['files']};pin(SNAP/'snapshot-download.json')
 def observed(name):
  rel=LABEL+'/'+name;p=pin(SNAP/rel,refs[rel]['sha256']);return lines(p)
 front=observed('frontend-stage.jsonl');manifest=read(pin(WAVE/'client/manifest.json'));cid=manifest['requests'][0]['request_id'];rid,link=request_rid(front,cid)
 acceptance=read(pin(R/'sg-replay-fences-gen67-root/root-native-acceptance.json'));assert acceptance['fence_count']==128 and acceptance['layer_row_count']==8192
 state=read(pin(SNAP/'sg-replay-fences-gen67-root/state.json',refs['sg-replay-fences-gen67-root/state.json']['sha256']))
 hook=R/'executor-run-gen27/received/runtime-sg-replay-fences-gen67-root/q27_sg_probe.py';pin(hook,state['source_SHA256']['runtime-sg-replay-fences-gen67-root/q27_sg_probe.py'])
 for name in ['sg_replay_outer_events.py','frontend_stage_probe_patch.py']:
  pin(R/'executor-run-gen27/received/runtime-sg-replay-fences-gen67-root'/name,state['source_SHA256']['runtime-sg-replay-fences-gen67-root/'+name])
 perrank=[];ledger=[];ids_by_rank=[]
 for rank,pid in [(0,61),(1,65),(2,69),(3,73)]:
  rows=observed('sg-'+str(pid)+'.jsonl');commits,zeros=commit_tokens(rows,rid);assert all(x['pid']==pid and x['rank']==rank for x in commits+zeros)
  compact=R/f'native-sg-recovery/gen74/rank{rank}-replay-native-compact.json';a=next(x for x in acceptance['ranks']if x['rank_id']==rank);native=read(pin(compact,a['compact_sha256']))
  assert native['rank_id']==rank and native['hardware_process']['device']==f'NPU {rank+4}'
  result=join_graph_tokens(rows,rid,commits,zeros,native['fences']);ledger.extend(result);ids=[t['last_token_id']for t in commits];ids_by_rank.append(ids)
  perrank.append({'rank':rank,'pid':pid,'output_commits':len(commits),'prefill_output_commits':sum(t['mode']=='1'for t in commits),'decode_output_commits':sum(t['mode']=='2'for t in commits),'zero_output_graph_tail':zeros[0]['step'],'explicit_graph_to_output_commit_pairs':sum(x['graph_invocation']is not None and x['model_sample_token_delta']==1 for x in result),'native_layer_rows_attached_to_output_commits':sum(x['native_layer_rows']for x in result if x['model_sample_token_delta']==1),'native_layer_rows_zero_tail':sum(x['native_layer_rows']for x in result if x['model_sample_token_delta']==0)})
 assert all(x==ids_by_rank[0]for x in ids_by_rank),'TP_rank_output_ID_disagreement'
 ids=ids_by_rank[0]
 frontend=observed('sg-13.jsonl');batches=[r for r in frontend if r.get('event')=='tokenizer_detokenizer_batch_output'and r.get('rids')==[rid]]
 assert len(batches)==32 and all(x['type']=='BatchStrOutput'and len(x['output_ids'])==1 and len(x['output_ids'][0])==1 for x in batches)
 frontend_ids=[r['output_ids'][0][0]for r in batches];assert frontend_ids==ids,'frontend_output_ID_sequence_differs'
 frames=[];wire=pin(WAVE/'client/request-0000/response.raw')
 for line in wire.read_text().splitlines():
  if line.startswith('data: ')and line[6:]!='[DONE]':frames.append(json.loads(line[6:]))
 assert all(x['id']==rid for x in frames);textframes=[(i,x['choices'][0]['text'])for i,x in enumerate(frames)if x['choices']and x['choices'][0].get('text')]
 text=''.join(t for i,t in textframes);assert len(textframes)==32
 te=read(pin(D/'tokenizer-evidence.json'));decoded=decode_selected(ids,te);assert decoded==text,'tokenizer_decode_differs_from_saved_wire'
 # Each local decoded prefix must equal the cumulative text prefix. This is
 # content agreement, not an independently observed per-frame token-ID edge.
 prefix='';prefix_checks=[]
 for n,(frame_i,chunk)in enumerate(textframes,1):
  prefix+=chunk;local=decode_selected(ids[:n],te);assert local==prefix
  prefix_checks.append({'output_index':n,'JSON_frame_index':frame_i,'token_id':ids[n-1],'decoded_prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest()})
 native=read(pin(D/'native-tokenizer-decode-v2.json'))
 if native['status']=='CPU_Rust_tokenizer_decode_complete':
  out=native['output'];assert out['tokenizer_sha256']==te['files']['tokenizer.json']['sha256']and out['ids']==ids and out['decoded_prefixes']==[decode_selected(ids[:i],te)for i in range(33)]
  assert not out['torch_loaded']and not out['torch_npu_loaded']
 vp=R/'executor-run-gen27/received/waves/full-lineage-fx-model-g81-exec-g3-vllm-in512-c1-o32/client/request-0000/response.raw';pin(vp)
 vtext=''.join(c.get('text','')for l in vp.read_text().splitlines()if l.startswith('data: ')and l[6:]!='[DONE]'for c in json.loads(l[6:]).get('choices',[]));assert vtext!=text
 comparison=read(pin(R/'executor-run-gen27/output-text-comparison.json',quality_pin));assert hashlib.sha256(text.encode()).hexdigest()==comparison['requests'][1]['text_sha256']
 with(D/'sg-output-token-ledger.csv').open('w')as f:
  keys=['rank','pid','rid','step','mode','committed_output_ids_total','model_sample_token_delta','last_token_id','line','graph_invocation','native_layer_rows'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(ledger)
 write(D/'sg-output-token-ledger.json',ledger)
 output={'task_id':'Q27-FULL-MAP-20261025','generation':35,'status':'executor_verified_SG_explicit_step_output_token_lineage_pending_root_acceptance','verified_ns':time.time_ns(),'review_level':'standalone executor; root acceptance pending','scope':'Saved cleanSG gen67 TP4 512/C1/out32 only; no new NPU/service/SDK/performance measurement','source_environment':{'Host Architecture':'aarch64','card':'Ascend910B1','CANN':'9.0.0','torch_npu':'2.10.0','physical_cards':[4,5,6,7]},'CPU_audit_architecture':'x86_64; not an NPU performance result','request':{'client_request_id':cid,'manager_rid':rid,'link_line':link['_line']},'per_rank':perrank,'output_ids':ids,'TP_rank_ID_sequences_equal':True,'frontend_batch_ID_sequence_equal':True,'frontend_batch_event_lines':[r['_line']for r in batches],'decoded_text':text,'decoded_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'tokenizer':{'sha256':te['files']['tokenizer.json']['sha256'],'decoder':te['files']['tokenizer.json']['decoder'],'current_read_ns':te['observed_ns'],'historical_loaded_tokenizer_SHA_available':False,'native_decode_status':native['status'],'think_token':next(a for a in te['files']['tokenizer.json']['added_tokens_relevant']if a['id']==248068)},'source_contract':source_contract,'prefix_content_checks':prefix_checks,'edge_classes':{'client_to_RID':'explicit same-record sg_request_id_link; exact full RID equals SSE id','graph_to_commit':'same rank/PID/explicit work_key.step and RID == scheduler_commit.forward_iter and token_commit.rid; not ordinal or nearest-time','commit_to_token_ID':'delta=1 and continuous committed_output_ids_total makes last_token_id the exact newly committed ID; zero delta retained','native_endpoint_to_graph':'existing root-accepted compact SHA and exact invocation join; no new raw trace audit this generation','scheduler_to_frontend_IDs':'independent same-RID 32-element output ID sequences are identical; per-message transport identity not observed','output_IDs_to_SSE_text':'current exact tokenizer decode and all32cumulative content prefixes equal saved wire; does not add missing per-frame token IDs'},'quality_inference':'SG wire text including <think> is consistent with committed scheduler output IDs and independently logged frontend output IDs; this is not merely an invented HTTP text string. Cross-framework root cause and vLLM generated token IDs remain unknown.','vllm_token_IDs':'not logged in saved output-processing or SSE records; do not substitute re-encoding as sampled token evidence','limitations':['Current tokenizer full SHA is observed now, not an immutable historical loaded-tokenizer receipt; exact decode agreement corroborates only this sequence.','<think> ID248068 currently has special=false; skip_special_tokens=true therefore does not remove it in native tokenizers decode. This explains retention, not why SG generated reasoning.','No logits/first-divergent-sampled-token/weight equivalence proof; no whole original matrix quality conclusion or causal framework attribution.','Graph-to-commit binds observed scheduler batch step; does not establish per-kernel ownership, per-layer causal dependence, exclusive time, all-stream critical path, or recoverable E2E benefit.','SSE-frame per-message identity and detokenizer transit IDs remain unobserved; content-prefix agreement is kept separate from identity edges.','HCCL waiting and alias dedup retained; rank totals never summed into wall-clock.','No actual204 completion, R allocation, publication, optimization adoption or overall acceptance.'],'inputs':inputs,'overall_complete':False,'optimization_adopted':False,'root_acceptance':'pending'}
 write(D/'sg-output-lineage-audit.json',output)
 print(json.dumps({'status':output['status'],'TP4_output_tokens':len(ids),'explicit_graph_output_pairs':sum(x['explicit_graph_to_output_commit_pairs']for x in perrank),'zero_tail_pairs':len(perrank),'frontend_IDs_match':True,'text_SHA':output['decoded_text_sha256'],'native_decode':native['status']}))
if __name__=='__main__':main()
