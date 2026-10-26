"""CPU gate for actual RPC serializer nonce and observer records; run offline."""
import sys,os,json,pickle,tempfile,threading
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from diagnostics.ordinary_boundary import observer as ob
s=SchedulerOutput.make_empty();s._ob_submit_id=719
class Sink:
 def send_multipart(self,buffers,**kwargs):self.buffers=[bytes(b) for b in buffers]
sink=Sink();fake=SimpleNamespace(_is_writer=True,n_local_reader=0,n_remote_reader=1,remote_socket=sink)
MessageQueue.enqueue(fake,('execute_model',(s,),{}))
obj=pickle.loads(sink.buffers[0],buffers=sink.buffers[1:]);assert obj[1][0]._ob_submit_id==719
assert obj[1][0].num_scheduled_tokens==s.num_scheduled_tokens
with tempfile.TemporaryDirectory() as t:
 os.environ['EXTREME_ORDINARY_BOUNDARY_DIR']=t
 assert not ob.enabled();ob.core_event('disabled');assert not (Path(t)/'core.jsonl').exists()
 (Path(t)/'arm').touch();assert ob.enabled()
 def writer(n):
  for i in range(25):ob.core_event('concurrent',writer=n,index=i)
 threads=[threading.Thread(target=writer,args=(n,)) for n in range(2)]
 for x in threads:x.start()
 for x in threads:x.join()
 rows=[json.loads(x) for x in (Path(t)/'core.jsonl').read_text().splitlines()]
 assert len(rows)==50 and len({(x['writer'],x['index']) for x in rows})==50
 owner=SimpleNamespace();ob.worker_event(owner,'seed_begin',submit_id=719);ob.worker_event(owner,'seed_end',submit_id=719)
 ob.flush(owner,0,5);assert owner._ordinary_boundary_rows==[]
 row=json.loads((Path(t)/'rank0_cohort5.json').read_text());assert len(row['events'])==2
 timing=json.loads((Path(t)/'rank0_flush.jsonl').read_text());assert timing['end_ns']>=timing['begin_ns']
 del os.environ['EXTREME_ORDINARY_BOUNDARY_DIR']
print(json.dumps({'pass':True,'serializer':'actual MessageQueue.enqueue with captured transport buffers; pickle decode','cpu_only':True,'concurrent_core_records':50}))
