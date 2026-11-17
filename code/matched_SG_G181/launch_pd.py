from pathlib import Path
import os,sys,json,subprocess,hashlib
r=Path('/task/sg_pd_g181');runtime=Path('/task/runtime_sglang_g32');os.chdir(r);c=json.loads((r/'candidate.json').read_text())
for k,v in c['environment'].items():
 if k!='PYTHONPATH':os.environ[k]=v
assert os.environ['PYTHONPATH'].split(os.pathsep)[:2]==[str(runtime),str(r)]
for k in ['GLM53_LOAD_LEDGER_DIR']:
 Path(os.environ[k]).mkdir(parents=True,exist_ok=True)
from glm53_sglang_adapter.checkpoint_bindings import IDENTITY_SHA256,INDEX_SHA256
assert hashlib.sha256((runtime/'nextn_checkpoint_identity.json').read_bytes()).hexdigest()==IDENTITY_SHA256
proof=json.loads((runtime/'nextn_checkpoint_receipt.json').read_text());assert proof['identity_sha256']==IDENTITY_SHA256 and proof['index_sha256']==INDEX_SHA256 and proof['tensor_bytes_sha_verified'] is True
assert hashlib.sha256(Path('/models/glm53/quant_model_weights.safetensors.index.json').read_bytes()).hexdigest()==INDEX_SHA256
q=subprocess.run([sys.executable,str(r/'native_readiness.py'),'--model-path','/models/glm53','--output',str(r/'native_readiness.json')],cwd=r,timeout=120);assert q.returncode==0
with (r/'server_help.txt').open('x') as f:q=subprocess.run([sys.executable,'-m','sglang.launch_server','--help'],cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=120)
assert q.returncode==0;helptext=(r/'server_help.txt').read_text()
for arg in c['argv']:
 if arg.startswith('--'):assert arg in helptext,'Native help lacks flag '+arg
from sglang.srt.disaggregation.ascend.transfer_engine import import_error
assert import_error is None,repr(import_error)
assert os.environ['ASCEND_MF_STORE_URL']=='tcp://<HOST>:8678'
argv=[sys.executable,*c['argv'][1:]]
m={'argv':argv,'candidate_sha256':hashlib.sha256((r/'candidate.json').read_bytes()).hexdigest(),'runtime_overlay_manifest_sha256':hashlib.sha256((runtime/'g32_sinkhorn_overlay_manifest.json').read_bytes()).hexdigest(),'runtime_manifest_sha256':hashlib.sha256((runtime/'integrated_g18_manifest.json').read_bytes()).hexdigest(),'scope':'Fullcontext263296 PP1TP8EP8MTP3 nativePD/autoDeepEP with graph+async candidate; actualeffectivecapacity/wire/graph/async/bench NOT ACCEPTED','environment':{k:os.environ[k] for k in c['environment'] if k!='PYTHONPATH'},'PYTHONPATH':os.environ['PYTHONPATH']};m['runtime_overlay_receipt_sha256']=hashlib.sha256((runtime/'g29_runtime_source_delta.json').read_bytes()).hexdigest();(r/'launch_manifest.json').write_text(json.dumps(m,indent=2))
os.execvpe(argv[0],argv,os.environ.copy())
