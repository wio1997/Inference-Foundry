"""Preserve stopped all16 raw, CPU-export and correlate only tagged new copies."""
from pathlib import Path
from collections import Counter,defaultdict
from decimal import Decimal
import hashlib
import json
import re
import shutil
import socket
import subprocess
import tarfile
import time

assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live2-20261019/node33')
gate=json.loads((root/'indexer-k-graph-copy-profile-all16-v2/result.json').read_text())
assert gate['completed'] and gate['candidate_restored_native'] and gate['observer_off']
assert len(gate['profiles'])==2 and all(z['profile_stopped'] and z['ranks']==list(range(16)) for z in gate['profiles'])
assert shutil.disk_usage(root).free>16*1024**3
out=Path(__file__).resolve().parent/'artifacts'
out.mkdir(exist_ok=False)
snapshot=root/'indexer-k-graph-copy-raw-v2'
snapshot.mkdir(exist_ok=False)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


manifest=[]
folders=[]
for phase in gate['profiles']:
    for name in phase['dirs']:
        folder=Path(name)
        assert folder.parent==root/'profile' and not folder.is_symlink()
        rank=int(re.search(r'_rank(\d+)_',folder.name).group(1))
        for p in sorted(folder.rglob('*')):
            if not p.is_file():
                continue
            assert not p.is_symlink()
            digest=sha(p)
            q=snapshot/folder.name/p.relative_to(folder)
            q.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,q)
            assert sha(q)==digest==sha(p)
            manifest.append(dict(condition=phase['condition'],rank=rank,path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest))
        folders.append((phase['condition'],rank,folder.name))
assert len(folders)==32 and sum(z['bytes'] for z in manifest)<4*1024**3
(out/'raw-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(out/'all16-raw.tar.gz','w:gz',compresslevel=3) as archive:
    for _,_,name in folders:
        archive.add(snapshot/name,arcname=name,recursive=True)
image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
argv=['docker','run','--rm','--pull','never','--network','none','--name','glm53-indexer-k-graph-copy-all16-export-v2',
      '--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','16','--memory','64g','--pids-limit','4096',
      '--tmpfs','/tmp:rw,size=2g','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1',
      '--device','/dev/davinci_manager','-v',str(snapshot)+':/profiles',
      '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/timeout',image,
      '--signal=TERM','--kill-after=15','900','python3','-c',
      "from torch_npu.profiler.profiler import analyse; analyse('/profiles',max_process_number=2,export_type='text')"]
start=time.monotonic()
with (out/'export-stdout.txt').open('xb') as a,(out/'export-stderr.txt').open('xb') as b:
    result=subprocess.run(argv,stdout=a,stderr=b,timeout=930)
assert result.returncode==0,result.returncode
rows=[]
for condition,rank,name in sorted(folders):
    traces=list((snapshot/name).rglob('trace_view.json'))
    assert len(traces)==1,(condition,rank,traces)
    p=traces[0]
    target=out/condition/('rank'+str(rank))
    target.mkdir(parents=True)
    shutil.copyfile(p,target/'trace_view.json')
    for csv in (snapshot/name).rglob('kernel_details.csv'):
        shutil.copyfile(csv,target/csv.name)
    obj=json.loads(p.read_text())
    events=obj['traceEvents'] if isinstance(obj,dict) else obj
    xs=[z for z in events if z.get('ph')=='X']
    labels=[z for z in xs if z.get('name','').startswith('glm53.indexer.Kgraph.input.')]
    dequeues={(z.get('name'),z.get('args',{}).get('correlation_id')):z for z in xs if z.get('cat')=='dequeue'}
    finishes={z['id']:z for z in events if z.get('cat')=='HostToDevice' and z.get('ph')=='f'}
    starts=[z for z in events if z.get('cat')=='HostToDevice' and z.get('ph')=='s']
    tasks=defaultdict(list)
    for z in xs:
        if 'Task Type' in z.get('args',{}):
            tasks[(z['pid'],z['tid'],str(z['ts']))].append(z)
    copies=[]
    for label in labels:
        begin=Decimal(str(label['ts']))
        end=begin+Decimal(str(label['dur']))
        children=[z for z in xs if z.get('pid')==label['pid'] and z.get('tid')==label['tid'] and begin<=Decimal(str(z['ts']))<=end]
        cpu=[z for z in children if z.get('name')=='aten::copy_']
        enq=[z for z in children if z.get('name')=='Enqueue@aclnnInplaceCopy']
        deq=dequeues.get(('Dequeue@aclnnInplaceCopy',enq[0]['args']['correlation_id'])) if len(enq)==1 else None
        flows=[z for z in starts if deq and z['tid']==deq['tid'] and Decimal(str(deq['ts']))<=Decimal(str(z['ts']))<=Decimal(str(deq['ts']))+Decimal(str(deq['dur']))]
        data=[]
        ids=[]
        for flow in flows:
            endpoint=finishes.get(flow['id'])
            if endpoint:
                matched=tasks[(endpoint['pid'],endpoint['tid'],str(endpoint['ts']))]
                data.extend(matched)
                ids.append(flow['id'])
        types=[z['args']['Task Type'] for z in data]
        copies.append(dict(label=label['name'],cpu_copies=len(cpu),enqueues=len(enq),
                           correlation_id=enq[0]['args']['correlation_id'] if enq else None,
                           flow_ids=ids,physical_types=types,
                           exact_SDMA=len(cpu)==len(enq)==1 and bool(data) and all(t=='SDMA_SQE' for t in types)))
    counts=Counter(z['label'].split('.')[4] for z in copies)
    # Full labels are glm53.indexer.Kgraph.input.<hidden|cos|sin>.<phase>.N<N>.
    balanced=bool(copies) and counts['hidden']==counts['cos']==counts['sin']
    row=dict(condition=condition,rank=rank,trace_bytes=p.stat().st_size,trace_sha256=sha(p),
             added_copies=len(copies),copy_kinds=dict(counts),balanced_three_inputs=balanced,
             N=sorted({int(re.search(r'\.N(\d+)$',z['label']).group(1)) for z in copies}),
             physical_types=dict(Counter(t for z in copies for t in z['physical_types'])),
             all_added_copy_SDMA=balanced and all(z['exact_SDMA'] for z in copies))
    (target/'copy-chains.json').write_text(json.dumps(copies,indent=2)+'\n')
    rows.append(row)
    del obj,events,xs,labels,dequeues,finishes,starts,tasks,copies
report=dict(completed=True,CPU_only_export=True,export_rc=result.returncode,export_elapsed_s=time.monotonic()-start,
            raw_files=len(manifest),raw_bytes=sum(z['bytes'] for z in manifest),
            raw_archive_bytes=(out/'all16-raw.tar.gz').stat().st_size,raw_archive_sha256=sha(out/'all16-raw.tar.gz'),
            rows=rows,all16_C1_C4_new_copies_SDMA=len(rows)==32 and all(z['all_added_copy_SDMA'] for z in rows),
            CID=gate['CID'],scope='Actual deployed C1/C4 diagnostics, every labeled new input copy follows enqueue/dequeue then exact HostToDevice flow to physical task. Strict SDMA_SQE only; untagged native host copies excluded. No formal service timing or claim about every future byte.')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
assert report['all16_C1_C4_new_copies_SDMA']
