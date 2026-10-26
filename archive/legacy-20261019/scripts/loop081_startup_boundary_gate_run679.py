"""CPU-only serializer and lossless concurrent buffered observer gate; offline."""
import sys,os,json,pickle,tempfile,threading,socket,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
with socket.socket() as s:
    s.settimeout(1)
    if s.connect_ex(('127.0.0.1',8080))==0:raise RuntimeError('refuse CPU gate while serving ready')
assert time.get_clock_info('perf_counter').implementation==time.get_clock_info('monotonic').implementation
assert not time.get_clock_info('perf_counter').adjustable
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from diagnostics.startup_boundary import observer as ob
s=SchedulerOutput.make_empty();s._ob_submit_id=719
class Sink:
    def send_multipart(self,buffers,**kwargs):self.buffers=[bytes(b) for b in buffers]
sink=Sink();fake=SimpleNamespace(_is_writer=True,n_local_reader=0,n_remote_reader=1,remote_socket=sink)
MessageQueue.enqueue(fake,('execute_model',(s,),{}))
obj=pickle.loads(sink.buffers[0],buffers=sink.buffers[1:]);assert obj[1][0]._ob_submit_id==719
assert obj[1][0].num_scheduled_tokens==s.num_scheduled_tokens
with tempfile.TemporaryDirectory() as t:
    os.environ['EXTREME_STARTUP_BOUNDARY_DIR']=t
    assert not ob.enabled();ob.core_event('disabled');ob.core_flush()
    assert not (Path(t)/'core.jsonl').exists()
    (Path(t)/'arm').touch();assert ob.enabled()
    ob.core_event('buffer_probe')
    assert not (Path(t)/'core.jsonl').exists(),'core_event must not write files'
    errors=[]
    def writer(n):
        try:
            for i in range(100):
                ob.core_event('concurrent',writer=n,index=i)
                if i%11==0:ob.core_flush()
        except BaseException as e:errors.append(repr(e))
    threads=[threading.Thread(target=writer,args=(n,)) for n in range(4)]
    for x in threads:x.start()
    for x in threads:x.join()
    assert not errors,errors
    ob.core_flush()
    rows=[json.loads(x) for x in (Path(t)/'core.jsonl').read_text().splitlines()]
    assert len(rows)==401 and len({(x['writer'],x['index']) for x in rows if x['kind']=='concurrent'})==400
    assert [r['observer_seq'] for r in rows]==list(range(1,402))
    timing=[json.loads(x) for x in (Path(t)/'core_flush.jsonl').read_text().splitlines()]
    assert sum(r['count'] for r in timing)==401
    assert all(r['end_ns']>=r['begin_ns'] and r['last_seq']-r['first_seq']+1==r['count'] for r in timing)
    before=(Path(t)/'core.jsonl').read_bytes();ob.core_flush()
    assert (Path(t)/'core.jsonl').read_bytes()==before
    owner=SimpleNamespace();ob.worker_event(owner,'seed_begin',submit_id=719);ob.worker_event(owner,'seed_end',submit_id=719)
    ob.flush(owner,0,5);assert owner._ordinary_boundary_rows==[]
    row=json.loads((Path(t)/'rank0_cohort5.json').read_text());assert len(row['events'])==2
    timing=json.loads((Path(t)/'rank0_flush.jsonl').read_text());assert timing['end_ns']>=timing['begin_ns']
    del os.environ['EXTREME_STARTUP_BOUNDARY_DIR']
print(json.dumps({'pass':True,'serializer':'actual MessageQueue.enqueue with captured transport; pickle decode','cpu_only':True,'buffered_core_records':401,'concurrent_writers_and_flushers':4,'no_per_request_file_write':True,'worker_flush_timed':True}))
