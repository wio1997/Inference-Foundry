"""Read-only metadata inventory; emits counts/signatures, never log lines or values."""
from pathlib import Path
import ast,hashlib,json,subprocess
base=Path('/work/vllm-ascend-reproduce-172-on-169-20261028')
logs=[base/'runtime/logs/service.log',base/'analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic-selective2/runtime/logs/service.log']
keys=('engram','stride','storage_offset','contiguous','enqueue','dequeue','inplacecopy','slice_tiling');out=[]
for p in logs:
 if not p.is_file():out.append({'path':str(p),'exists':False});continue
 a=p.stat();assert a.st_size<64*1024*1024
 data=p.read_bytes();b=p.stat();assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 lines=data.lower().splitlines();out.append({'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'stable_during_read':True,'line_counts_only':{k:sum(k.encode() in line for line in lines) for k in keys}})
source=Path('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/embedding.py');raw=source.read_bytes();tree=ast.parse(raw);entry=[]
for z in ast.walk(tree):
 if isinstance(z,ast.ClassDef):
  for f in z.body:
   if isinstance(f,ast.FunctionDef) and f.name=='embed_gathered':entry.append({'class':z.name,'line':f.lineno,'signature':ast.unparse(f.args)})
q=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/include/torch_npu/csrc/core/npu/NPUQueue.h');a=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/include/torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h');c=Path('/usr/local/Ascend/cann-9.2.0/aarch64-linux/include/aclnnop/aclnn_copy.h')
libs=['/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/lib/libtorch_npu.so','/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/lib/libnpu_profiler.so','/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so']
symbols={}
for lib in libs:
 r=subprocess.run(['nm','-D','-C',lib],capture_output=True,text=True,check=True)
 needles=('Repository::Enqueue','Repository::Dequeue','aclnnInplaceCopyGetWorkspaceSize','aclnnInplaceCopy')
 symbols[lib]=[line for line in r.stdout.splitlines() if any(x in line for x in needles)][:20]
print(json.dumps({'service_log_inventory':out,'entry':entry,'entry_source_sha256':hashlib.sha256(raw).hexdigest(),'header_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (q,a,c)},'selected_dynamic_symbols':symbols,'limitations':['Substring search cannot exclude differently named evidence.','Header declarations and exported symbols do not establish actual dynamic call paths, interception, queue correlation or ABI safety.','No NPU calls, no log line contents.']},indent=2))
