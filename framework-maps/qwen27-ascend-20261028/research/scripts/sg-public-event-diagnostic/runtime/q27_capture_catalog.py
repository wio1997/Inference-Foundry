"""Startup-only capture catalogue. Diagnostic graph construction, never scored."""
import functools,os,json,time
from pathlib import Path
def install(module):
 kind=os.environ.get('Q27_CAPTURE_CATALOG');generation=os.environ.get('Q27_LINEAGE_GENERATION','initial');name=getattr(module,'__name__','')
 if kind=='vllm' and name=='vllm_ascend.worker.model_runner_v1':cls=module.NPUModelRunner;method='capture_model'
 elif kind=='sglang' and name=='sglang.srt.model_executor.model_runner':cls=module.ModelRunner;method='init_cuda_graphs'
 else:return
 orig=getattr(cls,method)
 if getattr(orig,'_q27_catalog',False):return
 @functools.wraps(orig)
 def observed(self,*args,**kwargs):
  import torch_npu
  base=Path('/task/full-map-20261025/capture-catalog-lineage')/kind/generation
  from install_npugraph_raw_dump import install as install_dump
  install_dump(dump_dir=Path('/task/full-map-20261025/graph-lineage')/kind/generation/'graphs')
  from weight_ptr_catalog import collect_model_catalog,write_catalog
  model=self.model
  from torch import distributed as dist
  catalog=collect_model_catalog(model,model_id=kind+'-pid'+str(os.getpid()))
  catalog.update(worker_pid=os.getpid(),rank_id=int(dist.get_rank()) if dist.is_initialized() else None,capture_id=generation,generation=generation)
  import hashlib
  catalog.pop('catalog_sha256',None);catalog['catalog_sha256']=hashlib.sha256(json.dumps(catalog,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
  write_catalog(catalog,Path('/task/full-map-20261025/graph-lineage')/kind/generation/('weight-ptrs-pid'+str(os.getpid())+'.json'))
  base.mkdir(parents=True,exist_ok=True)
  marker=base/('worker-'+str(os.getpid())+'.json')
  # Weight pointers and debug_dump topology need no startup native profile.
  # Serving-wave profiling supplies actual timestamps; avoid capturing redundant
  # startup graphs once per variant and exhausting disk during long studies.
  profile_capture=os.environ.get('Q27_CAPTURE_NATIVE_PROFILE')=='1';prof=None
  if profile_capture:
   config=torch_npu.profiler._ExperimentalConfig(export_type=torch_npu.profiler.ExportType.Text,profiler_level=torch_npu.profiler.ProfilerLevel.Level1,data_simplification=False)
   handler=torch_npu.profiler.tensorboard_trace_handler(str(base),worker_name='capture-'+str(os.getpid()),analyse_flag=False)
   prof=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,with_stack=False,experimental_config=config,on_trace_ready=handler)
  row={'pid':os.getpid(),'framework':kind,'method':method,'status':'capturing','start_ns':time.time_ns(),'native_startup_profile_enabled':profile_capture,'weight_catalog_and_raw_topology_enabled':True};marker.write_text(json.dumps(row))
  if prof is not None:prof.start()
  try:
   result=orig(self,*args,**kwargs);row['status']='complete';return result
  except BaseException as e:row.update(status='failed',error=repr(e));raise
  finally:
   if prof is not None:prof.stop()
   row['end_ns']=time.time_ns();marker.write_text(json.dumps(row))
 observed._q27_catalog=True;setattr(cls,method,observed)
