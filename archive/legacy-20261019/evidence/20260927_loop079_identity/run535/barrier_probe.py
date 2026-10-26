import json, tempfile, subprocess, sys
from pathlib import Path
from scripts.loop079_formal_ledger_server_validate_selftest import make_fixture
from scripts.loop079_formal_ledger_server_validate import sha

with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    k=make_fixture(root)
    data=k['ledger_dir']/'pid111.jsonl'
    foot=k['ledger_dir']/'pid111.flush.jsonl'
    allrows=[json.loads(l) for p in k['ledger_dir'].glob('pid*.jsonl') if '.flush.' not in p.name for l in p.read_text().splitlines()]
    rows=[r for r in allrows if r['event']=='clock_origin' or r['phase']=='warmup']
    for r in rows:
        if r['event']=='runner_done':r['target_graph_mode']='CUDAGraphMode.FULL'
        if r['event']=='scheduler_append':r.update(stopped=r['bulk'],stale=False,resumable=False)
    report=json.loads(k['client_report'].read_text())
    warm={'status':'warmup48_client_admitted','request_count':48,'unique_response_ids':48,'request_index':[r for r in report['request_index'] if r['phase']=='warmup']}
    k['warmup_report'].write_text(json.dumps(warm))
    transition=json.loads(k['transitions_path'].read_text().splitlines()[0])
    marker=root/'phase.json'
    marker.write_text(json.dumps(transition,sort_keys=True,separators=(',',':'))+'\n')
    def write(rs):
        for pid in {r['pid'] for r in rs}:
            selected=[r for r in rs if r['pid']==pid]
            for i,r in enumerate(selected):r['seq']=i
            blob=b''.join((json.dumps(r,separators=(',',':'))+'\n').encode() for r in selected)
            (k['ledger_dir']/f'pid{pid}.jsonl').write_bytes(blob)
            (k['ledger_dir']/f'pid{pid}.flush.jsonl').write_text(json.dumps(dict(run_id=k['run_id'],pid=pid,flush_index=0,reason='fixture',committed_records=len(selected),first_seq=0,last_seq=len(selected)-1,byte_offset_begin=0,byte_offset_end=len(blob),blob_sha256=sha(blob)))+'\n')
    def call():
        return subprocess.run([sys.executable,'scripts/loop079_formal_ledger_phase_barrier.py','--ledger-dir',str(k['ledger_dir']),'--client-report',str(k['warmup_report']),'--phase-marker',str(marker),'--run-id',k['run_id'],'--output',str(root/'barrier.json')],capture_output=True,text=True)
    write(rows)
    good=call();assert good.returncode==0,good.stderr
    count=0
    for event,field,value in [('api_consume','api_request_id','wrong'),('runner_done','target_graph_mode','NONE'),('scheduler_append','stale',True),('output_queue','output_request_id','wrong')]:
        target=next(r for r in rows if r['event']==event and (event!='scheduler_append' or r['bulk']))
        old=target[field];target[field]=value;write(rows)
        bad=call();assert bad.returncode!=0,(event,bad.stdout)
        target[field]=old;count+=1
    write(rows)
    fb=json.loads(foot.read_text());fb['run_id']='wrong';foot.write_text(json.dumps(fb)+'\n')
    assert call().returncode!=0;count+=1
    print(json.dumps({'status':'pass','positives':1,'negatives':count,'scope':'synthetic warmup48 barrier, no service'}))
