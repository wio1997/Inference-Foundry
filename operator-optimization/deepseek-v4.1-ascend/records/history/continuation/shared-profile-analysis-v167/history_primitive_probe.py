"""One tiny owned-device probe for epoch history and native replay marking."""
import json,os,resource,time,traceback
from pathlib import Path
ROOT=Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis/history167-primitives')
ROOT.mkdir(parents=True,exist_ok=True)
result={'status':'STARTING','pid':os.getpid(),'started_unix':time.time(),
        'scope':'Synthetic tensors only;no model load/inference;one device;existing service not changed.',
        'declared_application_tensor_bytes':968,'runtime_overhead_not_in_tensor_bytes':True}
def save():
 (ROOT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
save()
try:
 resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
 import torch,torch_npu
 torch.set_num_threads(1);torch_npu.npu.set_device(0)
 torch_npu.npu.set_per_process_memory_fraction(0.01,0)
 src=torch.arange(24,dtype=torch.int64,device='npu:0')
 idx=torch.zeros(1,dtype=torch.int64,device='npu:0')
 history=torch.full((4,24),-1,dtype=torch.int64,device='npu:0')
 def record():history.scatter_(0,idx.expand(1,24),src.view(1,24))
 record();torch_npu.npu.synchronize()
 graph=torch_npu.npu.NPUGraph()
 with torch_npu.npu.graph(graph):record()
 torch_npu.npu.synchronize();history.fill_(-1);torch_npu.npu.synchronize()
 exp=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
       aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,mstx=True,export_type=[torch_npu.profiler.ExportType.Db])
 prof=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
       record_shapes=False,profile_memory=False,with_stack=False,
       experimental_config=exp,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
        str(ROOT/'raw'),worker_name='history167-'+str(os.getpid()),analyse_flag=False))
 prof.start();marks=[]
 try:
  for epoch in [1,2]:
   src.add_(100);idx.fill_(epoch)
   rid=torch_npu.npu.mstx.range_start('wio_history167/replay/'+str(epoch))
   graph.replay()
   torch_npu.npu.mstx.range_end(rid)
   marks.append({'epoch':epoch,'range_id':rid})
 finally:prof.stop()
 torch_npu.npu.synchronize();observed=history.cpu().tolist()
 expected=[[-1]*24,list(range(100,124)),list(range(200,224)),[-1]*24]
 result.update(status='HISTORY_PRIMITIVE_CHECK_COMPLETE',history_equal=observed==expected,observed=observed,
  expected=expected,native_marks=marks,profiler_configuration=exp.__dict__,
  allocated_torch_bytes=torch_npu.npu.memory_allocated(0),reserved_torch_bytes=torch_npu.npu.memory_reserved(0),
  profiles=[str(x) for x in (ROOT/'raw').glob('*_ascend_pt')])
 if observed!=expected:result['status']='HISTORY_PRIMITIVE_MISMATCH'
except BaseException:
 result.update(status='HISTORY_PRIMITIVE_ERROR',error=traceback.format_exc())
finally:
 result['finished_unix']=time.time();save()
 print('WIO_RESULT='+json.dumps(result),flush=True)
