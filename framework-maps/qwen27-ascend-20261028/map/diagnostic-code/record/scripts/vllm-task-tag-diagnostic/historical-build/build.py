from pathlib import Path
import os,json,hashlib,time
P=Path('/task/full-map-20261025/vllm-task-tag-build-gen78-root');D=P/'build-output';D.mkdir(exist_ok=False)
import torch,torch_npu,torch_npu.version
assert str(torch_npu.__version__)=='2.10.0.post7'and torch_npu.version.git_version=='1d88b8608fd1e44545b5804026c7a57c44240bb4'
assert not torch.npu.is_initialized()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
H=Path('/usr/local/Ascend/cann-9.2.0/include/acl/acl_rt.h');pkg=Path(torch_npu.__file__).resolve().parent
state={'status':'CPU_building','started_ns':time.time_ns(),'torch':str(torch.__version__),'torch_npu':str(torch_npu.__version__),'npu_git':torch_npu.version.git_version,'header_sha256':sha(H),'cpp_sha256':sha(P/'task_anchor.cpp'),'device_initialized':False}
(D/'state.json').write_text(json.dumps(state,indent=2));os.environ['MAX_JOBS']='1';os.environ['OMP_NUM_THREADS']='1'
from torch.utils.cpp_extension import load
b=D/'build';b.mkdir();flags=[]
for lib in [pkg/'lib',H.parents[2]/'lib64',Path(torch.__file__).resolve().parent/'lib']:flags.extend(['-L'+str(lib),'-Wl,-rpath,'+str(lib)])
flags+=['-ltorch_npu','-lascendcl']
m=load(name='q27_vllm_task_anchor_tag_root78',sources=[str(P/'task_anchor.cpp')],build_directory=str(b),extra_include_paths=[str(pkg/'include'),str(H.parent.parent)],extra_cflags=['-O0','-g0','-std=c++17','-DQ27_HAS_ACLRT_CACHE_LAST_TASK_EXTEND_INFO=1'],extra_ldflags=flags,with_cuda=False,verbose=True)
assert not torch.npu.is_initialized();state.update(status='CPU_build_complete_no_device_init',module_path=str(m.__file__),module_sha256=sha(m.__file__),finished_ns=time.time_ns());(D/'state.json').write_text(json.dumps(state,indent=2));(D/'build-receipt.json').write_text(json.dumps(state,indent=2));print(json.dumps(state))
