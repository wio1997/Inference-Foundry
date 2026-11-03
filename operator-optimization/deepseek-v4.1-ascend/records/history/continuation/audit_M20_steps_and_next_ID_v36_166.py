from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,re,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18/M20-allrank-merged-draft-confidence-step-and-next-ID-scope-v36.json');b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='6ab7fb525ff4fb57eaeba777005b7bd077621f5b3f247ef7533682bd9e9cac2e';d=json.loads(b);OUT=p.parent/'M20-source-ordered-five-step-and-next-ID-row-audit-v36.json';assert not OUT.exists()
byapi=defaultdict(list)
for t in d['native_TASKs']:
 for a in t['source_APIs']:byapi[(t['database_index'],a['api_rowid'])].append(t)
def opcounts(tasks):return dict(Counter(t['definition']['opType_text']if t['definition']else'NO_COMPUTE'for t in tasks))
def uniq(apis,di):return{(di,t['TASK']['task_record_rowid']):t for a in apis for t in byapi[(di,a['API']['api_rowid'])]}
def rows(c):
 m=re.search(r'-?\d+',c.get('inputShapes_text',''));return int(m.group())if m else None
streams=defaultdict(list)
for a in d['native_APIs']:streams[(a['database_index'],a['API']['globalTid'])].append(a)
cycles=[];steps=[];issues=[];covered=set();stepids=set()
expected={'Slice':4,'GatherV2':1,'MatMulV2':1,'Add':1,'ViewCopy':2,'ArgMaxV2':1,'Cast':1}
for (di,tid),aa in streams.items():
 cur=None;step=[];parts=[]
 for a in sorted(aa,key=lambda a:(a['API']['startNs'],a['API']['api_rowid'])):
  fs=a['API']['project_frames'];start=any('/llm_base_proposer.py(1584)'in f for f in fs);end=any('/llm_base_proposer.py(1602)'in f for f in fs)
  if start:
   if cur is not None:issues.append(dict(kind='UNFINISHED_AT_NEXT_SEED',database_index=di,seed=cur[0]['API']['api_rowid']))
   cur=[a];step=[];parts=[];continue
  if cur is None:continue
  cur.append(a);step.append(a)
  if end:
   ts=uniq(step,di);counts=opcounts(ts.values());gemms=[t['definition']for t in ts.values()if t['definition']and t['definition']['opType_text']=='MatMulV2'];B=rows(gemms[0])if len(gemms)==1 else None
   z=dict(database_index=di,globalTid=tid,ordinal=len(parts),seed_API=cur[0]['API']['api_rowid'],publication_API=a['API']['api_rowid'],contexts=sorted({x['context']for x in step}),rows=B,TASKs=len(ts),types=counts,pattern_matches=counts==expected,TASK_keys=[list(k)for k in ts]);steps.append(z);parts.append(z);stepids.update(ts)
   if counts!=expected:issues.append(dict(kind='STEP_PATTERN_MISMATCH',step=z))
   step=[]
   if len(parts)==5:
    ts=uniq(cur,di);cycles.append(dict(database_index=di,globalTid=tid,seed_API=cur[0]['API']['api_rowid'],last_publication_API=a['API']['api_rowid'],contexts=sorted({x['context']for x in cur}),rows=sorted({x['rows']for x in parts},key=str),steps=parts,TASKs=len(ts),TASK_keys=[list(k)for k in ts]));covered.update(ts);cur=None;parts=[]
 if cur is not None:issues.append(dict(kind='UNFINISHED_END',database_index=di,seed=cur[0]['API']['api_rowid']))
allstep={ (t['database_index'],t['TASK']['task_record_rowid'])for t in d['native_TASKs']if any(any('/llm_base_proposer.py('+str(line)+')'in f for f in a['project_frames'])for a in t['source_APIs']for line in[1587,1588,1589,1599,1602])}
outside=[t for t in d['native_TASKs']if(t['database_index'],t['TASK']['task_record_rowid'])not in covered]
nextid=[t for t in d['native_TASKs']if any(': prepare_next_token_ids_padded'in f for a in t['source_APIs']for f in a['project_frames'])]
regimes=[]
for h in d['native_canonical_fullchains']:
 pass
for t in nextid:
 if t['definition']and t['definition']['opType_text']=='LogicalAnd':regimes.append(t['definition']['inputShapes_text'])
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),complete_five_step_patterns=len(cycles),steps=len(steps),pattern_issues=len(issues),step_type_pattern=expected,step_pattern_match_count=sum(s['pattern_matches']for s in steps),cycle_regimes=[dict(context=k[0],rows=k[1],patterns=v)for k,v in Counter(('|'.join(c['contexts']),str(c['rows']))for c in cycles).items()],step_regimes=[dict(context=k[0],rows=k[1],steps=v)for k,v in Counter(('|'.join(s['contexts']),s['rows'])for s in steps).items()],covered_seed_and_step_TASKs=len(covered),unassigned_step_TASKs=len(allstep-stepids),outside_step_pattern_TASKs=len(outside),outside_step_pattern_types=opcounts(outside),next_ID_TASKs=len(nextid),next_ID_types=opcounts(nextid),next_ID_mask_shape_counts=dict(Counter(regimes)),whole_module_closed=False,overall_complete=False)
result=dict(summary=summary,cycles=cycles,steps=steps,issues=issues,outside_TASKs=outside,next_ID_TASKs=nextid,limits=['Five-step source-order patterns require seed and every publication and reject unfinished/extra step tasks; not formal Python invocation/request or pointer proof.','Never associate next-ID helpers to a target execute root or Markov cycle by equal counts/nearest time.','Mask shapes/counts do not reveal actual sampled IDs, holes, valid counts, repeated Markov IDs or confidence flags.','No source/model/torch/NPU/service/health work.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
