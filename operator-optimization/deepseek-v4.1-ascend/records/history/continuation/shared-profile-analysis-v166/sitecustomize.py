"""Opt-in diagnostic constructor patch; no dispatch/model/kernel modification."""
import os
if os.environ.get('WIO_DEVICE_ONLY_PROFILE')=='v166':
 import functools,json,time,uuid,resource
 from pathlib import Path
 import torch_npu
 ROOT=Path(os.environ['WIO_DEVICE_PROFILE_ROOT'])
 (ROOT/'hook-records').mkdir(parents=True,exist_ok=True)
 incarnation=uuid.uuid4().hex;calls=0
 def emit(kind,data):
  row={'kind':kind,'pid':os.getpid(),'incarnation':incarnation,'monotonic_ns':time.monotonic_ns(),**data}
  p=ROOT/'hook-records'/(str(os.getpid())+'-'+incarnation+'-'+kind+'-'+str(calls)+'.json')
  p.write_text(json.dumps(row,indent=2)+'\n')
 original=torch_npu.profiler.profile
 @functools.wraps(original)
 def device_profile(*args,**kwargs):
  global calls
  if args:raise RuntimeError('Unexpected positional profiler construction;not silently changed')
  calls+=1
  kwargs.update(activities=[torch_npu.profiler.ProfilerActivity.NPU],record_shapes=False,
                profile_memory=False,with_stack=False,with_modules=False)
  exp=torch_npu.profiler._ExperimentalConfig(
   profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
   aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
   export_type=[torch_npu.profiler.ExportType.Db])
  kwargs['experimental_config']=exp
  # This factory is invoked by start_profile after model load; don't constrain
  # Engram shared backing creation during startup.
  resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,256*1024**2))
  result=original(**kwargs)
  emit('profile_constructed',{'activities':['NPU'],'record_shapes':False,'profile_memory':False,
    'with_stack':False,'with_modules':False,'experimental':exp.__dict__,'result_type':str(type(result))})
  return result
 original_handler=torch_npu.profiler.tensorboard_trace_handler
 @functools.wraps(original_handler)
 def raw_handler(*args,**kwargs):
  if len(args)>2:raise RuntimeError('Unexpected positional trace-handler settings')
  kwargs['analyse_flag']=False
  return original_handler(*args,**kwargs)
 torch_npu.profiler.profile=device_profile
 torch_npu.profiler.tensorboard_trace_handler=raw_handler
 emit('hook_installed',{'original_profile_module':original.__module__,'patch':'NPU-only-Level0-no-PMU-no-auto-analysis'})
