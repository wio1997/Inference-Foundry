from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18';OUT=M/'M18-context-bank-copy-and-input-metadata-publication-surface-v34.json';assert not OUT.exists()
def pin(p,s):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==s;return json.loads(b)
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0');ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');selected={};apirows=[];bychain=defaultdict(list);keys=set()
for db in fs['databases']:
 di=db['database_index'];selected[di]={}
 for a in db['API_records']:
  text='\n'.join(a['project_frames']);role='CONTEXT_BANK_COPY'if'/dflash_proposer.py(122)'in text else'CONTEXT_QUERY_SLOT_METADATA'if'/dflash_proposer.py(132)'in text else None
  if role:
   z=dict(database_index=di,API=a,role=role,phase='DUMMY_ANCESTRY'if'_dummy_run'in text or': dummy_run'in text else'PROPOSAL_OR_OTHER_ANCESTRY');selected[di][a['api_rowid']]=z;apirows.append(z);keys.add(a['chain_key'])
tasks=[]
for ref in ix['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);cc={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  hits=sorted(set(t.get('native_linked_API_rowids',[]))&selected[di].keys())
  if not hits:continue
  rr=[selected[di][key]for key in hits];z=dict(database_index=di,TASK=t,definition=cc.get(t['globalTaskId']),roles=sorted({r['role']for r in rr}),phases=sorted({r['phase']for r in rr}),source_APIs=[r['API']for r in rr]);tasks.append(z)
  for key in{r['API']['chain_key']for r in rr}:bychain[key].append(z)
def sh(c):return dict(op=c['opType_text'],input=c.get('inputShapes_text'),output=c.get('outputShapes_text'),input_types=c.get('inputDataTypes_text'),output_types=c.get('outputDataTypes_text'))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),native_APIs=len(apirows),native_TASKs=len(tasks),native_chains=len(keys),roles=[dict(role=k[0],phase=k[1],op=k[2],TASKs=v)for k,v in Counter(('|'.join(r['roles']),'|'.join(r['phases']),r['definition']['opType_text']if r['definition']else'NO_COMPUTE_DEFINITION')for r in tasks).items()],typed_shapes=[dict(role=k[0],signature=json.loads(k[1]),TASKs=v)for k,v in Counter(('|'.join(r['roles']),json.dumps(sh(r['definition']),sort_keys=True))for r in tasks if r['definition']).items()],whole_module_closed=False,overall_complete=False)
headers=[dict(chain_key=key,APIs=sum(r['API']['chain_key']==key for r in apirows),TASKs=len(bychain[key]),project_frames=[f['frame']for f in fs['canonical_fullchains'][key]if'/vllm'in f['frame']])for key in sorted(keys)]
result=dict(summary=summary,native_APIs=apirows,native_TASKs=tasks,fullchains={key:fs['canonical_fullchains'][key]for key in keys},limits=['Specificnewsource-boundpublicationcopy/metadataonly;notwholeinput-preparationtiming.','Memcpy/outputbasecapacityshapedoesnotproveallcapacitybytescopied;sourceprefix/capturebanksemanticsseparate.','Copyeliminationdoesnotdeletequery/slot/position/rejected-tokenmetadataeffects.','Noexactproposerinvocation/root/alias/generationfromequalcountsorordinalpairing.','Nonewsource/model/torch/NPU/service/healthwork.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,chain_headers=headers,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
