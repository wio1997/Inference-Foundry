from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18'
OUT=M/'M21-original-entrypoint-and-selector-gap-native-census-v41.json';assert not OUT.exists()
REQUEST=__REQUEST__
started=time.monotonic()
def pin(path,sha):
    b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,path;return json.loads(b)
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
covered=set(REQUEST['already_selected_chain_keys']);entries=REQUEST['entries']
patterns=[('/vllm_ascend/'+x['rel']+'('+str(x['line'])+'): '+x['function'])for x in entries]
selected=defaultdict(dict);apis=[];entry_counts=[Counter()for _ in entries];chain_hits=defaultdict(set)
for db in fs['databases']:
    di=db['database_index']
    for a in db['API_records']:
        hits=[i for i,s in enumerate(patterns)if any(s in f for f in a['project_frames'])]
        if not hits:continue
        already=a['chain_key']in covered
        x=dict(database_index=di,API=a,entry_indices=hits,in_v37_selector=already)
        selected[di][a['api_rowid']]=x;apis.append(x);chain_hits[a['chain_key']].update(hits)
        for i in hits:entry_counts[i]['inside_existing_selector'if already else'outside_existing_selector']+=1
tasks=[];bychain=defaultdict(list)
for ref in ix['databases']:
    assert time.monotonic()-started<140,'finite CPU audit deadline'
    di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);defs={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
    for t in d['TASK_records']:
        hits=sorted(set(t.get('native_linked_API_rowids',[]))&selected[di].keys())
        if not hits:continue
        chains=sorted({selected[di][k]['API']['chain_key']for k in hits})
        x=dict(database_index=di,TASK=t,definition=defs.get(t['globalTaskId']),source_API_rowids=hits,source_chains=chains,has_outside_chain=any(k not in covered for k in chains))
        tasks.append(x)
        for k in chains:bychain[k].append(x)
headers=[]
for k in sorted(chain_hits):
    rr=bychain[k];shape_counts=Counter()
    for r in rr:
        c=r['definition']
        if c is not None:
            s=dict(op=c['opType_text'],input=c.get('inputShapes_text'),output=c.get('outputShapes_text'),input_types=c.get('inputDataTypes_text'),output_types=c.get('outputDataTypes_text'))
            shape_counts[json.dumps(s,sort_keys=True)]+=1
    headers.append(dict(chain_key=k,in_v37_selector=k in covered,APIs=sum(a['API']['chain_key']==k for a in apis),TASKs=len(rr),entry_indices=sorted(chain_hits[k]),typed_shapes=[dict(signature=json.loads(s),TASKs=n)for s,n in shape_counts.items()],project_frames=[f['frame']for f in fs['canonical_fullchains'][k]if'/vllm'in f['frame']]))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),original_entrypoints=len(entries),entry_API_counts=[dict(entry=e,**dict(c))for e,c in zip(entries,entry_counts)],native_APIs=len(apis),native_TASKs=len(tasks),native_chains=len(headers),outside_selector_APIs=sum(not a['in_v37_selector']for a in apis),outside_selector_TASKs=sum(t['has_outside_chain']for t in tasks),outside_selector_chains=sum(not h['in_v37_selector']for h in headers),elapsed_seconds=time.monotonic()-started,whole_module_closed=False,overall_complete=False)
result=dict(summary=summary,API_records=apis,TASK_records=tasks,chain_headers=headers,request=REQUEST,limits=['Exactcurrent166nativefile/line/functionancestrymatchesagainstoldEP3entryhints. Oldcounts/phaseidentitiesnotpromoted.',
    'Ancestryoverlapandalready-reviewedneighbor-moduleworkremainexplicit. MissingnativeAPI/TASKdoesnotprovenoCPUcopy/disabledbranch.',
    'All32savedidentityfiles/SHAverified;no newcapture/model/torch/NPU/service/healthwork.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,chain_headers=headers,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
