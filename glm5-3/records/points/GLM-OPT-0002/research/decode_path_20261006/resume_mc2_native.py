"""CPU-only resume of isolated same-version build; no installed file edits."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path
import torch,torch_npu
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'MC2-NATIVE-BUILD-CPU-20261007-D'
PROJECT=PREVIOUS/'source/pytorch-5dd8ef3f9b375b5ae4a83538d5785754148c3302'
assert not torch.npu.is_initialized()
INSTALLED=Path(torch_npu.__file__).parent
assert hashlib.sha256((INSTALLED/'lib/libtorch_npu.so').read_bytes()).hexdigest()=='83fb9a0eb249aef6bca7f8463047fcb4d3062cc4849f17ddf31a5e050c838842'
assert json.loads((PREVIOUS/'build_state.json').read_text())['error']=="RuntimeError('compile exit=1')"
def status(phase,**extra):
 row=dict(phase=phase,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),NPU_initialized=False,model_request=False,installed_library_modified=False,**extra)
 p=ROOT/'build_state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(row,indent=2)+'\n');t.replace(p)
try:
 target=PROJECT/'third_party/acl/inc/acl/super_kernel.h'
 source=Path('/usr/local/Ascend/ascend-toolkit/latest/include/super_kernel/super_kernel.h')
 assert source.is_file() and not target.exists()
 before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [PREVIOUS/'native_candidate_identity.json',PREVIOUS/'compile.stdout.log',PREVIOUS/'compile.stderr.log',PROJECT/'CMakeLists.txt']}
 target.symlink_to(source)
 (ROOT/'resume_identity.json').write_text(json.dumps(dict(previous=before,added_header=str(target),header_source=str(source),header_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),existing_sources_and_failure_logs_unchanged=True),indent=2)+'\n')
 status('compile',resume_from=str(PREVIOUS))
 with (ROOT/'compile.stdout.log').open('xb') as out,(ROOT/'compile.stderr.log').open('xb') as err:
  proc=subprocess.run(['/usr/local/python3.12.13/bin/cmake','--build',str(PROJECT/'build'),'--parallel','4'],env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',MAX_JOBS='4'),stdout=out,stderr=err,timeout=5400)
 assert proc.returncode==0, 'compile exit='+str(proc.returncode)
 assert not torch.npu.is_initialized()
 library=PREVIOUS/'output/libtorch_npu.so';assert library.is_file()
 status('compiled',library=str(library),bytes=library.stat().st_size,sha256=hashlib.sha256(library.read_bytes()).hexdigest(),model_correctness=False,matched_AB=False,E2E_gain=False)
except BaseException as error:
 status('failed',error=repr(error),model_correctness=False,matched_AB=False);raise
