"""CPU fake tests: import identity, window ACKs, loss, caps, and forwarding."""
import sys
if sys.flags.optimize:raise RuntimeError('tests require assertions enabled')
import hashlib,importlib.util,json,os,sys,tempfile,time,types
from pathlib import Path
import engram_bootstrap as boot
tempfile.tempdir=os.path.realpath(tempfile.gettempdir())
import audited_observer as obs
class Storage:pass
class Device:type='cpu';index=None
class Tensor:
 def __init__(self):self.shape=(13,24);self.dtype='torch.int32';self.device=Device();self.st=Storage();self.private='NEVER_LOG_THIS_VALUE'
 def stride(self):return (48,1)
 def storage_offset(self):return 24
 def is_contiguous(self):return False
 def untyped_storage(self):return self.st
sys.modules['torch']=types.SimpleNamespace(Tensor=Tensor)

def wait(path,cond):
 deadline=time.monotonic()+5
 while time.monotonic()<deadline:
  try:
   v=json.loads(path.read_text())
   if cond(v):return v
  except (OSError,ValueError):pass
  time.sleep(.05)
 raise AssertionError('status timeout: '+str(path))

with tempfile.TemporaryDirectory() as td:
 p=Path(td);src=p/'model.py';src.write_text('class AscendParallelEngramEmbedding:\n    def embed_gathered(self, gathered, num_tokens):\n        if num_tokens == 0: raise ValueError("original")\n        return gathered\n')
 boot.TARGET='wio_test_embedding';boot.EXPECTED_FILE=str(src);boot.EXPECTED_SHA=hashlib.sha256(src.read_bytes()).hexdigest()
 spec=importlib.util.spec_from_file_location(boot.TARGET,str(src));mod=importlib.util.module_from_spec(spec);sys.modules[boot.TARGET]=mod;spec.loader.exec_module(mod)
 old=mod.AscendParallelEngramEmbedding.__dict__['embed_gathered']
 goodsha=boot.EXPECTED_SHA;boot.EXPECTED_SHA='0'*64
 assert boot._patch(mod) is False and mod.AscendParallelEngramEmbedding.__dict__['embed_gathered'] is old
 boot.EXPECTED_SHA=goodsha
 os.environ.pop('WIO_ENGRAM_META_ENABLE',None);assert obs.install(mod.AscendParallelEngramEmbedding,{}) is False
 os.environ.update(WIO_ENGRAM_META_ENABLE='1',WIO_ENGRAM_RUN_ID='a'*32,WIO_DIAG_DIR=td,WIO_ENGRAM_META_PATH=str(p/'engram-metadata.jsonl'))
 assert boot._patch(mod) and mod.AscendParallelEngramEmbedding.__dict__['embed_gathered'].__wrapped__ is old
 inst=obs._STATE;assert inst is not None
 status=p/('observer-'+inst.process+'.json');wait(status,lambda r:r['status']=='INSTALLED')
 gate=p/'PROFILE_ACTIVE';gate.write_text(json.dumps({'run_id':'a'*32,'window_id':'b'*32}))
 wait(status,lambda r:r['status']=='WINDOW_OPEN')
 t=Tensor();obj=mod.AscendParallelEngramEmbedding();assert obj.embed_gathered(t,13) is t
 try:obj.embed_gathered(t,0)
 except ValueError as e:assert str(e)=='original'
 else:raise AssertionError('real exception swallowed')
 gate.unlink();closed=wait(status,lambda r:r['status']=='WINDOW_CLOSED')
 assert closed['window_id']=='b'*32 and closed['inflight']==0 and closed['calls_seen']==2 and closed['begin_written']==2 and closed['end_written']==2
 assert closed['graph_coverage']=='UNKNOWN' and closed['closure_snapshot']=='CONDITIONAL_NOT_PROOF'
 events=[json.loads(x) for x in (p/'engram-metadata.jsonl').read_bytes().splitlines()];assert len(events)==4 and all(x['window_id']=='b'*32 for x in events)
 assert b'NEVER_LOG_THIS_VALUE' not in (p/'engram-metadata.jsonl').read_bytes()
 # Force lock contention: original call proceeds, checkpoint explicitly tainted.
 gate.write_text(json.dumps({'run_id':'a'*32,'window_id':'c'*32}))
 wait(status,lambda r:r['status']=='WINDOW_OPEN' and r['window_id']=='c'*32)
 assert inst.mu.acquire(False)
 try:assert obj.embed_gathered(t,13) is t
 finally:inst.mu.release()
 gate.unlink();closed=wait(status,lambda r:r['status']=='WINDOW_CLOSED' and r['window_id']=='c'*32)
 assert closed['state_incomplete'] and closed['stats']['state_lock_drop']>=1
 inst.finish()
 # Existing status cap must refuse a 65th process rather than silently drop.
 capdir=p/'cap';capdir.mkdir()
 for i in range(obs.MAX_WORKERS):(capdir/('observer-'+format(i,'024x')+'.json')).write_text('{}')
 bounded=obs._Observer(str(capdir/'engram-metadata.jsonl'),str(capdir))
 assert not bounded.checkpoint('INSTALLED') and bounded.loss and bounded.stats['status_drop']>=1
 print(json.dumps({'status':'PASS','bootstrap_source_and_code_verification':'PASS','window_ack':'PASS','state_lock_loss_tainted':'PASS','original_forwarding':'PASS','graph':'UNKNOWN'}))
