"""Preserve stopped all16 raw, CPU-export and correlate only tagged new copies."""
from pathlib import Path
from collections import Counter,defaultdict
from bisect import bisect_left,bisect_right
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
assert snapshot.is_dir()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


old=Path('/data/wio/glm5-3-vllm-execution/20261017-inventory/node33-indexer_k_graph_copy_export_all16_v2-6e59a8b10665/artifacts')
manifest=json.loads((old/'raw-manifest.json').read_text())
# The original pre-export raw archive remains immutable and is included in this receipt.
shutil.copyfile(old/'raw-manifest.json',out/'raw-manifest.json')
shutil.copyfile(old/'all16-raw.tar.gz',out/'all16-raw.tar.gz')
original_snapshot=snapshot
snapshot=root/'indexer-k-graph-copy-export-missing-v4'
snapshot.mkdir(exist_ok=False)
folders=[];trace_roots={};missing=[]
for phase in gate['profiles']:
 for name in phase['dirs']:
  folder=Path(name);rank=int(re.search(r'_rank(\d+)_',folder.name).group(1));key=(phase['condition'],rank,folder.name)
  folders.append(key);source=original_snapshot/folder.name
  traces=list(source.rglob('trace_view.json'));csvs=list(source.rglob('kernel_details.csv'))
  complete=len(traces)==len(csvs)==1
  if complete:
   with traces[0].open('rb') as f:
    f.seek(max(0,traces[0].stat().st_size-128));complete=f.read().rstrip().endswith(b'}')
  if complete:
   trace_roots[folder.name]=source
  else:
   missing.append(folder.name);dest=snapshot/folder.name
   for row in manifest:
    rel=Path(row['path']);parts=rel.parts
    if parts[1]!=folder.name:continue
    original=root/rel;target=dest/Path(*parts[2:]);target.parent.mkdir(parents=True,exist_ok=True)
    assert original.stat().st_size==row['bytes'] and sha(original)==row['sha256']
    shutil.copyfile(original,target);assert sha(target)==row['sha256']
   assert dest.exists();trace_roots[folder.name]=dest
assert len(folders)==32
(out/'recovery-inputs.json').write_text(json.dumps(dict(missing=missing,reused_exported=len(folders)-len(missing),old_failed_export=str(old),CPU_only=True),indent=2)+'\n')
image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
argv=['docker','run','--rm','--pull','never','--network','none','--name','glm53-indexer-k-graph-copy-missing-export-v4',
      '--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','16','--memory','64g','--pids-limit','4096',
      '--tmpfs','/tmp:rw,size=2g','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1',
      '--device','/dev/davinci_manager','-v',str(snapshot)+':/profiles',
      '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/timeout',image,
      '--signal=TERM','--kill-after=15','600','python3','-c',
      "from torch_npu.profiler.profiler import analyse; analyse('/profiles',max_process_number=4,export_type='text')"]
start=time.monotonic()
with (out/'export-stdout.txt').open('xb') as a,(out/'export-stderr.txt').open('xb') as b:
    result=subprocess.run(argv,stdout=a,stderr=b,timeout=630) if missing else subprocess.CompletedProcess(argv,0)
assert result.returncode==0,result.returncode
rows=[]
for condition,rank,name in sorted(folders):
    traces=list(trace_roots[name].rglob('trace_view.json'))
    assert len(traces)==1,(condition,rank,traces)
    p=traces[0]
    target=out/condition/('rank'+str(rank))
    target.mkdir(parents=True)
    shutil.copyfile(p,target/'trace_view.json')
    for csv in trace_roots[name].rglob('kernel_details.csv'):
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
    cpu_by_thread=defaultdict(list);flows_by_thread=defaultdict(list)
    for z in xs:
        if z.get('name') in ['aten::copy_','Enqueue@aclnnInplaceCopy']:
            cpu_by_thread[(z['pid'],z['tid'])].append(z)
    for z in starts:flows_by_thread[z['tid']].append(z)
    for rows_ in list(cpu_by_thread.values())+list(flows_by_thread.values()):rows_.sort(key=lambda z:Decimal(str(z['ts'])))
    cpu_times={k:[Decimal(str(z['ts'])) for z in v] for k,v in cpu_by_thread.items()}
    flow_times={k:[Decimal(str(z['ts'])) for z in v] for k,v in flows_by_thread.items()}
    copies=[]
    for label in labels:
        begin=Decimal(str(label['ts']))
        end=begin+Decimal(str(label['dur']))
        key=(label['pid'],label['tid']);times=cpu_times.get(key,[])
        children=cpu_by_thread.get(key,[])[bisect_left(times,begin):bisect_right(times,end)]
        cpu=[z for z in children if z.get('name')=='aten::copy_']
        enq=[z for z in children if z.get('name')=='Enqueue@aclnnInplaceCopy']
        deq=dequeues.get(('Dequeue@aclnnInplaceCopy',enq[0]['args']['correlation_id'])) if len(enq)==1 else None
        flows=[]
        if deq:
            times=flow_times.get(deq['tid'],[]);start_=Decimal(str(deq['ts']));end_=start_+Decimal(str(deq['dur']))
            flows=flows_by_thread.get(deq['tid'],[])[bisect_left(times,start_):bisect_right(times,end_)]
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
    print(json.dumps(dict(progress_condition=condition,rank=rank,copies=len(copies),all_SDMA=row['all_added_copy_SDMA'])),flush=True)
    (target/'copy-chains.json').write_text(json.dumps(copies,indent=2)+'\n')
    rows.append(row)
    del obj,events,xs,labels,dequeues,finishes,starts,tasks,copies,cpu_by_thread,flows_by_thread,cpu_times,flow_times
report=dict(completed=True,CPU_only_export=True,export_rc=result.returncode,export_elapsed_s=time.monotonic()-start,
            raw_files=len(manifest),raw_bytes=sum(z['bytes'] for z in manifest),
            raw_archive_bytes=(out/'all16-raw.tar.gz').stat().st_size,raw_archive_sha256=sha(out/'all16-raw.tar.gz'),
            rows=rows,CPU_missing_reexport=missing,reused_exported=32-len(missing),all16_C1_C4_new_copies_SDMA=len(rows)==32 and all(z['all_added_copy_SDMA'] for z in rows),
            CID=gate['CID'],scope='Actual deployed C1/C4 diagnostics, every labeled new input copy follows enqueue/dequeue then exact HostToDevice flow to physical task. Strict SDMA_SQE only; untagged native host copies excluded. No formal service timing or claim about every future byte.')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
assert report['all16_C1_C4_new_copies_SDMA']
