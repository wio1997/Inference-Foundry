"""CPU-only exercise of natural import and default-off behavior."""
import sys
if sys.flags.optimize:raise RuntimeError('tests require assertions enabled')
from pathlib import Path
import hashlib,importlib,os,sys,tempfile,json
import engram_bootstrap as boot
tempfile.tempdir=os.path.realpath(tempfile.gettempdir())
with tempfile.TemporaryDirectory() as d:
 p=Path(d);source=p/'wio_test_natural.py';source.write_text('class AscendParallelEngramEmbedding:\n    def embed_gathered(self, gathered, num_tokens):\n        return gathered\n')
 boot.TARGET='wio_test_natural';boot.EXPECTED_FILE=str(source);boot.EXPECTED_SHA=hashlib.sha256(source.read_bytes()).hexdigest();sys.path.insert(0,d)
 os.environ.pop('WIO_ENGRAM_META_ENABLE',None);assert not boot.activate() and boot.TARGET not in sys.modules
 os.environ.update(WIO_ENGRAM_META_ENABLE='1',WIO_ENGRAM_RUN_ID='a'*32,WIO_DIAG_DIR=d,WIO_ENGRAM_META_PATH=str(p/'engram-metadata.jsonl'))
 assert boot.activate() and boot.TARGET not in sys.modules
 module=importlib.import_module(boot.TARGET);fn=module.AscendParallelEngramEmbedding.__dict__['embed_gathered']
 assert getattr(fn,'_wio_engram_metadata',False) and fn.__wrapped__.__name__=='embed_gathered' and not boot.activate()
 import audited_observer
 if audited_observer._STATE is not None:audited_observer._STATE.finish()
 print(json.dumps({'status':'PASS','default_disabled':'PASS','natural_import_only':'PASS','duplicate_activation':'refused','npu_operations':0}))
