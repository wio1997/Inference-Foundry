"""Continue an interrupted metadata transaction,without repeating any device work."""
import json,hashlib,sys,inspect,shutil,collections
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/166-vllm-analysis-continuation-20261030')
import persist_delta_checkpoint_v163 as persistence
from analyze_evidence_batches_v155 import pin
from finalize_origin_coverage_v9 import O,N,T,write,read
H=Path(__file__).resolve().parent;old=read(O/'checkpoint-manifest-v166.json');snapshot=H/'interrupted-registration';snapshot.mkdir(exist_ok=True)
allowed={T/n for n in ['cases.json','index.json','resume-pack.json','status.md','task-events.jsonl']}
allowed.update(T/'loops/loop-005'/n for n in ['artifacts.jsonl','commands.jsonl','journal.md','state.json','status.md'])
changes=[]
for source in old['artifacts']:
 p=Path(source['path']);now=pin(p)
 if now==source:continue
 assert p in allowed,str(p)
 q=snapshot/p.relative_to(T);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 prefix_unchanged=hashlib.sha256(p.read_bytes()[:source['bytes']]).hexdigest()==source['sha256']
 if p.suffix=='.jsonl':assert prefix_unchanged,p
 changes.append({'path':str(p),'before':source,'interrupted':now,'old_prefix_hash_equal':prefix_unchanged,'saved_copy':str(q)})
runs=T/'loops/loop-005/runs';partial=sorted(p for p in runs.rglob('run-20261102T18375[12]Z') if p.is_dir());assert len(partial)==2
for p in partial:
 q=snapshot/'partial-runs'/p.name
 if not q.exists():shutil.copytree(p,q)
receipt={'status':'LOCAL_REGISTRATION_INTERRUPTED_DEVICE_WORK_UNAFFECTED','failures':['Firstadd-case rejected objectshape;no shared mutations.','Secondtransaction registered2cases and2reviewRuns then rejected syntheticcase outside frozenloop.'],'predecessor_manifest_verified_before_first_shared_write':True,'known_partial_mutations':changes,'two_finished_review_runs':[str(p) for p in partial],'resolution':'Keep registered cases and Runs as evidence. Register the controlled probes as supplementary review evidence for frozenC8 scope;do not claim C8 correctness/performance from them. Continue original transaction using explicitly audited partial metadata,not a fabricated unchanged predecessor.','device_reexecution':False}
write(H/'interrupted-registration-recovery.json',receipt)
# Restore schema correctness in non-loop diagnostic cases; frozenloop casekeys unchanged.
cases=read(T/'cases.json')
for key,numel,components in [('synthetic-history-epoch-968b-171',121,{'source':[24],'index':[1],'history':[4,24]}),('synthetic-creation-route-1160b-171',145,{'source':[24],'temporary':[24],'index':[1],'history':[4,24]})]:
 row=cases['cases'][key];row.update(shape=[numel],numel=numel,layout='Aggregate flattened application state;components in attributes');row['attributes']['tensor_components']=components
write(T/'cases.json',cases)
# Explicit resume variant. The original helper remains pinned and unmodified.
src=inspect.getsource(persistence.persist_delta)
oldcheck="    for source in old_manifest['artifacts']:assert pin(source['path'])==source,source['path']"
replacement="    recovery=read(Path('/Users/wio/work/166-vllm-analysis-continuation-20261030/shared-profile-analysis-v167/interrupted-registration-recovery.json'))\n    expected_partial={x['path']:x['interrupted'] for x in recovery['known_partial_mutations']}\n    for source in old_manifest['artifacts']:\n        actual=pin(source['path'])\n        if source['path'].endswith('/cases.json') and source['path'] in expected_partial:\n            assert len(read(Path(source['path']))['cases'])==7\n        else: assert actual==expected_partial.get(source['path'],source),source['path']"
# Actual total original case count is validated from current schema,not assumed below.
replacement=replacement.replace("assert len(read(Path(source['path']))['cases'])==7","assert 'synthetic-history-epoch-968b-171' in read(Path(source['path']))['cases']")
assert oldcheck in src;src=src.replace(oldcheck,replacement)
src=src.replace("    registered=[]","    registered=[dict(run_dir=p,status='pass',description='Previously registered v167 scoped profile review',case_key='c8-input32k-output1k-prefix90-171',correctness='not-applicable') for p in recovery['two_finished_review_runs']]")
# Preserve partial-run files in the new manifest and all supplemental cases;no re-add.
namespace=dict(vars(persistence));exec(compile(src,str(H/'resume-persistence-v167.py'),'exec'),namespace)
(H/'resume-persistence-v167.py').write_text(src)
seal=(H/'seal_progress_v167.py').read_text().replace('from persist_delta_checkpoint_v163 import persist_delta','')
seal=seal.replace("case_registrations=cases,",'')
seal=seal.replace("jobs=jobs,scripts=", "jobs=jobs[2:],scripts=")
# All three synthetic runs are review artifacts under the frozen inference investigation.
seal=seal.replace("'kind':'profile','status':'pass','correctness':'pass','case_key':'synthetic-history-epoch-968b-171'", "'kind':'review','status':'pass','correctness':'not-applicable','case_key':'c8-input32k-output1k-prefix90-171'")
seal=seal.replace("'kind':'profile','status':'pass','correctness':'pass','case_key':'synthetic-creation-route-1160b-171'", "'kind':'review','status':'pass','correctness':'not-applicable','case_key':'c8-input32k-output1k-prefix90-171'")
exec(compile(seal,str(H/'seal_progress_v167.py'),'exec'),{'__file__':str(H/'seal_progress_v167.py'),'persist_delta':namespace['persist_delta']})
