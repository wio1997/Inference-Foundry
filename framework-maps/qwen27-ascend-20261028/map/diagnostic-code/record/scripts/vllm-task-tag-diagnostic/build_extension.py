"""Derived CPU compiler entry. Does not start a service or initialize a device."""
from pathlib import Path
import argparse, hashlib, json, os, platform, time

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.2.0'))
 a=p.parse_args();root=Path(__file__).resolve().parent
 source=root/'historical-build/task_anchor.cpp';env=json.loads((root/'environment.json').read_text())
 assert platform.machine()==env['Host Architecture'],'Only frozen aarch64 build is described'
 def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
 bindings=json.loads((root/'source-bindings.json').read_text())['files']
 expected=next(x['sha256']for x in bindings if x['destination']=='historical-build/task_anchor.cpp')
 assert sha(source)==expected
 import torch,torch_npu,torch_npu.version
 assert str(torch.__version__)==env['historical_build_receipt']['torch']
 assert str(torch_npu.__version__)==env['torch_npu']
 assert torch_npu.version.git_version==env['torch_npu_git']
 assert not torch.npu.is_initialized()
 header=a.cann_root/'include/acl/acl_rt.h';pkg=Path(torch_npu.__file__).resolve().parent
 assert sha(header)==env['historical_build_receipt']['header_sha256']
 assert 'aclrtCacheLastTaskExtendInfo' in header.read_text()
 a.output.mkdir(parents=True,exist_ok=False)
 state={'status':'CPU_building','started_ns':time.time_ns(),'Host Architecture':platform.machine(),
  'torch':str(torch.__version__),'torch_npu':str(torch_npu.__version__),
  'npu_git':torch_npu.version.git_version,'header_sha256':sha(header),
  'cpp_sha256':sha(source),'device_initialized':False,'derived_entry':True}
 def save(): (a.output/'build-receipt.json').write_text(json.dumps(state,indent=2)+'\n')
 save();os.environ['MAX_JOBS']='1';os.environ['OMP_NUM_THREADS']='1'
 from torch.utils.cpp_extension import load
 build=a.output/'build';build.mkdir();flags=[]
 for lib in [pkg/'lib',a.cann_root/'lib64',Path(torch.__file__).resolve().parent/'lib']:
  flags.extend(['-L'+str(lib),'-Wl,-rpath,'+str(lib)])
 flags+=['-ltorch_npu','-lascendcl']
 try:
  module=load(name=env['historical_extension_module'],sources=[str(source)],build_directory=str(build),
   extra_include_paths=[str(pkg/'include'),str(a.cann_root/'include')],
   extra_cflags=['-O0','-g0','-std=c++17','-DQ27_HAS_ACLRT_CACHE_LAST_TASK_EXTEND_INFO=1'],
   extra_ldflags=flags,with_cuda=False,verbose=True)
  assert not torch.npu.is_initialized()
  state.update(status='CPU_build_complete_no_device_init',module_path=str(module.__file__),
   module_sha256=sha(Path(module.__file__)),finished_ns=time.time_ns())
 except BaseException as exc:
  state.update(status='failed',error_type=type(exc).__name__,finished_ns=time.time_ns());save();raise
 save();print(json.dumps(state))

if __name__=='__main__':main()
