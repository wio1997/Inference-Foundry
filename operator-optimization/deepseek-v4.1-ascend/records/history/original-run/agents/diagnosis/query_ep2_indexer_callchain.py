#!/usr/bin/env python3
"""Read only own EP2 profile, resolve existing launch associations and stack IDs."""
import hashlib,json,shlex,subprocess
from pathlib import Path
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';stage='wio-vllm-v41-repro169-stage-20261028'
identity='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602|sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
code=r'''import pathlib,sqlite3,hashlib,json,re,collections
root=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
ps=list(root.glob('*ep2_rank*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_2.db'));assert len(ps)==1;p=ps[0]
h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
assert h.hexdigest()=='2f51a3a608eaf21fbadbb98d54e79032bc23638da76cd80344e9b5c363add7c9'
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.execute('pragma query_only=on');c.row_factory=sqlite3.Row
# An explicit join, preserving cardinality; no assumption that time containment links objects.
links=[]
for cann in (126630,126633,126640):
 ids=[r[0] for r in c.execute('select id from CONNECTION_IDS where connectionId=?',(cann,))];assert len(ids)==1
 rows=[dict(r) for r in c.execute('select a.startNs,a.endNs,a.globalTid,a.connectionId,a.callchainId,s.value name from PYTORCH_API a left join STRING_IDS s on s.id=a.name where a.connectionId=?',(ids[0],))]
 assert len(rows)==(0 if cann==126630 else 1)
 links.append({'cann_connection_id':cann,'framework_connection_id':ids[0],'pytorch_api':rows})
assert [x['framework_connection_id'] for x in links]==[117246,244391,244392]
assert links[1]['pytorch_api'][0]['callchainId']==547589 and links[2]['pytorch_api'][0]['callchainId']==547593
# Dequeue matching uses the exact main-thread span and name; NULL means no id to join.
tid=4780298607293
rows=[dict(r) for r in c.execute('select a.startNs,a.endNs,a.globalTid,a.connectionId,a.callchainId,s.value name from PYTORCH_API a left join STRING_IDS s on s.id=a.name where a.globalTid=? and a.startNs<? and a.endNs>? order by a.startNs',(tid,'1793241334239383160','1793241334237248320'))]
dq=[x for x in rows if x['name'] in ('Dequeue@aclnnVllmAscendQuantLightningIndexerV2','Dequeue@_prepare_indexer_indices_kernel')]
assert len(dq)==2 and all(x['callchainId'] is None for x in dq)
chains=collections.defaultdict(list)
for r in c.execute('select a.id,a.stackDepth,s.value from PYTORCH_CALLCHAINS a left join STRING_IDS s on s.id=a.stack where a.id in (547589,547593) order by a.id,a.stackDepth'):
 chains[str(r[0])].append({'depth':r[1],'frame':r[2]})
assert len(chains['547589'])==38 and len(chains['547593'])==41
assert 'indexer.py(275): select_projected' in chains['547589'][1]['frame']
assert 'prepare_indexer_indices.py(123): prepare_indexer_indices' in chains['547593'][3]['frame']
assert 'indexer.py(291): select_projected' in chains['547593'][4]['frame']
# Filename inventory is a bounded hint, not proof of absence for opaque binary formats.
prof=p.parent.parent;files=[q for q in prof.rglob('*') if q.is_file()]
patterns=re.compile(r'(sched|ftrace|perf.data|syscall|osrt|thread.state)',re.I)
matches=[str(q.relative_to(prof)) for q in files if patterns.search(q.name)]
print(json.dumps({'db_sha256':h.hexdigest(),'links':links,'dequeue_records':dq,
 'callchain_depth_count':{k:len(v) for k,v in chains.items()},'top_frames':{k:v[:7] for k,v in chains.items()},
 'inventory_file_count':len(files),'inventory_name_pattern':patterns.pattern,'inventory_matching_paths':matches,
 'limitations':['Explicit CONNECTION_IDS links CANN launch to framework submit API, not CANN outer/Dequeue callchain; table semantics require exporter identity.','Captured Python submission stack is neither a sampled stack of the Dequeue worker nor a trace of inside the ACLNN call.','Filename search covers the selected profile folder and chosen terms only; opaque binary content not decoded.','No host/device timestamp subtraction; no proof of runnable versus sleeping or particular wait reason.']},indent=2))'''
def call(a):
 r=subprocess.run([ssh,shlex.join(a)],capture_output=True,text=True,timeout=90)
 if r.returncode:raise RuntimeError(r.stderr[-1000:])
 return r.stdout
assert call(['docker','inspect','--format','{{.Id}}|{{.Image}}',stage]).strip()==identity
raw=call(['docker','exec',stage,'nice','-n','15','ionice','-c','3','python3','-c',code]);data=json.loads(raw)
assert call(['docker','inspect','--format','{{.Id}}|{{.Image}}',stage]).strip()==identity
p=Path(__file__).with_name('ep2-indexer-callchain.json');p.write_text(json.dumps(data,indent=2)+'\n')
print('json_sha256',hashlib.sha256(p.read_bytes()).hexdigest(),'files',data['inventory_file_count'],'candidates',data['inventory_matching_paths']);print('stack_lengths',data['callchain_depth_count'])
