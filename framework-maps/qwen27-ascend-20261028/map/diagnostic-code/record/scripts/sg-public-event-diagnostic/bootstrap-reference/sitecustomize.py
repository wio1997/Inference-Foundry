"""Task-only verified source overlay, enabled by an explicit environment path."""
import hashlib,importlib.abc,importlib.machinery,json,os,sys
from pathlib import Path
_manifest=os.environ.get('Q27_COMPAT_MANIFEST')
if _manifest:
 _doc=json.loads(Path(_manifest).read_text());_rows={r['module']:r for r in _doc['modules']}
 class _Loader(importlib.abc.Loader):
  def __init__(self,row):self.row=row
  def create_module(self,spec):return None
  def exec_module(self,module):
   r=self.row;code_path=r['candidate_path'] if r.get('compiled_kernel_overlay') else r['original_path'];module.__file__=code_path;original=Path(r['original_path']).read_bytes();candidate=Path(r['candidate_path']).read_bytes();assert hashlib.sha256(original).hexdigest()==r['original_sha256'];assert hashlib.sha256(candidate).hexdigest()==r['candidate_sha256'];exec(compile(candidate,code_path,'exec'),module.__dict__)
   if r['module']=='sglang.srt.models.qwen3_5':
    from q27_weight_gate import install
    install(module)
   if r.get('hook'):
    from q27_diag_hooks import install
    install(r['hook'],module)
   if os.environ.get('Q27_FULL_PROBE')=='vllm':
    import q27_vllm_probe
    q27_vllm_probe.install()
   if os.environ.get('Q27_CAPTURE_CATALOG'):
    import q27_capture_catalog
    q27_capture_catalog.install(module)
   extra={}
   if r.get('compiled_kernel_overlay'):
    kernel=module._fused_sigmoid_gating_delta_rule_update_decode_kernel
    for _ in range(8):
     if hasattr(kernel,'src'):break
     kernel=getattr(kernel,'fn',None)
     if kernel is None:raise RuntimeError('Q27 JIT source unwrap failed: no .fn/.src after %d wrappers' % (_+1))
    if not hasattr(kernel,'src'):raise RuntimeError('Q27 JIT source unwrap failed: no .src after 8 wrappers')
    if '+ o_v[None, :] * K' not in kernel.src or '+ o_k[:, None] * V' in kernel.src:raise RuntimeError('Q27 JIT source layout assertion failed')
    extra={'jit_source_sha256':hashlib.sha256(kernel.src.encode()).hexdigest(),'jit_state_layout':'HV,V,K','inspect_source_file':code_path}
   root=Path(os.environ.get('Q27_RUN_RECEIPT_DIR','/task/full-map-20261025'))/'compat-receipts';root.mkdir(parents=True,exist_ok=True);p=root/(r['module']+'.pid'+str(os.getpid())+'.json');p.write_text(json.dumps({'pid':os.getpid(),'module':r['module'],'original_sha256':r['original_sha256'],'candidate_sha256':r['candidate_sha256'],**extra},indent=2))
 class _Finder(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname in _rows:return importlib.machinery.ModuleSpec(fullname,_Loader(_rows[fullname]),origin=_rows[fullname]['original_path'])
 sys.meta_path.insert(0,_Finder())

if os.environ.get("Q27_FULL_PROBE")=="vllm":
 import q27_vllm_probe
 q27_vllm_probe.register_import_finder()

if os.environ.get("Q27_SG_PROBE")=="1":
 import q27_sg_probe
 q27_sg_probe.install_finder()
 q27_sg_probe.install_ipc_observer()
