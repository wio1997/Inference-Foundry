#!/usr/bin/env python3
"""Small real-NPU support probe, no weights, service or performance claims."""
import datetime,hashlib,importlib.metadata,json,platform,subprocess,sys,traceback
v={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':platform.platform(),'python':sys.version,
   'device_compute_executed':False,'model_loaded':False,'PD_MTP_executed':False,'performance':'NOT_MEASURED','checks':[]}
try:
 import torch
 import torch_npu
 v['packages']={n:importlib.metadata.version(n) for n in ['torch','torch-npu','sglang']}
 v['torch_file']=torch.__file__;v['torch_npu_file']=torch_npu.__file__
 torch.npu.set_device(0)
 v['device_name']=torch.npu.get_device_name(0)
 v['device_compute_executed']=True
 x=torch.arange(4096,dtype=torch.int32).reshape(64,64)
 ix=torch.tensor([63,0,1,31],dtype=torch.int64)
 actual=(x.to('npu:0').index_select(0,ix.to('npu:0'))+1).cpu()
 torch.npu.synchronize()
 expected=x.index_select(0,ix)+1
 v['checks'].append({'name':'int32 H2D/gather/add/D2H agrees with CPU golden','pass':torch.equal(actual,expected),'shape':list(actual.shape),'sha256':hashlib.sha256(actual.numpy().tobytes()).hexdigest()})
 for dtype in [torch.float16,torch.bfloat16]:
  x=torch.arange(256,dtype=torch.float32).to(dtype).reshape(16,16)
  result=(x.to('npu:0')*2).cpu();torch.npu.synchronize()
  v['checks'].append({'name':str(dtype)+' exact power-of-two scale roundtrip','pass':torch.equal(result,x*2),'shape':list(result.shape)})
 v['memory_allocated_bytes']=torch.npu.memory_allocated(0)
 p=subprocess.run(['git','-C','/sgl-workspace/sglang','rev-parse','HEAD'],capture_output=True,text=True,timeout=10)
 v['image_source_commit']=p.stdout.strip() if p.returncode==0 else None
 v['status']='PASS' if all(c['pass'] for c in v['checks']) else 'FAIL'
except Exception as e:v['status']='UNAVAILABLE';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
print('L039_RESULT='+json.dumps(v));sys.exit(0 if v['status']=='PASS' else 1)
