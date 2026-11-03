"""Stage CPU-only smoke: actual torch tensors; no NPU module or hardware."""
import sys
if sys.flags.optimize:raise RuntimeError('tests require assertions enabled')
import hashlib, importlib.util, json, os, sys, tempfile, time
from pathlib import Path
import torch
import engram_bootstrap as boot
tempfile.tempdir=os.path.realpath(tempfile.gettempdir())
import audited_observer as obs
assert 'torch_npu' not in sys.modules

def wait(path, pred):
    deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        try:
            value=json.loads(path.read_text())
            if pred(value):return value
        except (OSError,ValueError):pass
        time.sleep(.05)
    raise AssertionError('status timeout')

with tempfile.TemporaryDirectory() as td:
    p=Path(td); src=p/'model.py'
    src.write_text('class AscendParallelEngramEmbedding:\n    def embed_gathered(self, gathered, num_tokens):\n        return gathered\n')
    boot.TARGET='wio_real_cpu_embedding';boot.EXPECTED_FILE=str(src);boot.EXPECTED_SHA=hashlib.sha256(src.read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location(boot.TARGET,str(src));module=importlib.util.module_from_spec(spec);sys.modules[boot.TARGET]=module;spec.loader.exec_module(module)
    os.environ.update(WIO_ENGRAM_META_ENABLE='1',WIO_ENGRAM_RUN_ID='a'*32,WIO_DIAG_DIR=td,WIO_ENGRAM_META_PATH=str(p/'engram-metadata.jsonl'))
    assert boot._patch(module)
    ob=obs._STATE;status=p/('observer-'+ob.process+'.json');wait(status,lambda x:x['status']=='INSTALLED')
    gate=p/'PROFILE_ACTIVE';gate.write_text(json.dumps({'run_id':'a'*32,'window_id':'b'*32}));wait(status,lambda x:x['status']=='WINDOW_OPEN')
    obj=module.AscendParallelEngramEmbedding();base=torch.zeros((13,2,24),dtype=torch.int32,device='cpu')
    for slot in (0,1,0,1):
        t=base[:,slot,:];assert obj.embed_gathered(t,13) is t
    gate.unlink();s=wait(status,lambda x:x['status']=='WINDOW_CLOSED')
    assert s['calls_seen']==4 and s['begin_written']==4 and s['end_written']==4 and not s['state_incomplete']
    ev=[json.loads(x) for x in (p/'engram-metadata.jsonl').read_text().splitlines()];begin=[x['tensor'] for x in ev if x['event']=='embed_begin']
    assert len(begin)==4 and [x['storage_offset'] for x in begin]==[0,24,0,24]
    assert all(x['shape']==[13,24] and x['stride']==[48,1] and x['is_contiguous'] is False and x['device_type']=='cpu' for x in begin)
    assert 'torch_npu' not in sys.modules
    ob.finish()
    print(json.dumps({'status':'PASS','real_cpu_tensor':True,'offsets':[0,24,0,24],'stride':[48,1],'torch_npu_imported':False,'graph_coverage':'UNKNOWN'}))
