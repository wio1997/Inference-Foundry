"""CPU import/readiness gate: no model construction, weight or NPU allocation."""
import argparse,hashlib,importlib,importlib.util,inspect,json,os,sys,traceback
from pathlib import Path

ARCH='Glm5NextForConditionalGeneration';MODULE='sglang.srt.models.glm5_next'
RUNTIME=Path('/task/runtime_sglang_g32')

def assert_native_class():
    module=importlib.import_module(MODULE)  # Deliberately strict: keep full traceback.
    cls=getattr(module,ARCH)
    expected=RUNTIME/'sglang/srt/models/glm5_next.py'
    if cls.__module__!=MODULE or Path(inspect.getfile(cls)).resolve()!=expected:
        raise RuntimeError('Native GLM class is not from the prebuilt task runtime')
    loader=cls.load_weights;original=inspect.unwrap(loader)
    ledger=RUNTIME/'glm53_sglang_adapter/ledger.py'
    if not hasattr(loader,'__wrapped__') or Path(loader.__code__.co_filename).resolve()!=ledger or Path(inspect.getfile(original)).resolve()!=expected:
        raise RuntimeError('Strict audited_load wrapper missing or resolved from another package')
    return cls

def check(model_path=None):
    if os.environ.get('SGLANG_EXTERNAL_MODEL_PACKAGE')!='glm53_native_gate':
        raise RuntimeError('Per-process strict native registry package is required')
    cls=assert_native_class()
    from sglang.srt.models.registry import ModelRegistry
    resolved,arch=ModelRegistry.resolve_model_cls([ARCH])
    if resolved is not cls or arch!=ARCH:raise RuntimeError('Registry did not resolve the exact native class')
    if model_path:
        config=json.loads((Path(model_path)/'config.json').read_text())
        if config.get('architectures')!=[ARCH]:raise RuntimeError('Unexpected model architectures; refusing fallback')
    return {'resolved_arch':arch,'class_module':resolved.__module__,'class_file':inspect.getfile(resolved),
            'class_sha256':hashlib.sha256(Path(inspect.getfile(resolved)).read_bytes()).hexdigest(),
            'loader_wrapper_file':resolved.load_weights.__code__.co_filename,'loader_original_file':inspect.getfile(inspect.unwrap(resolved.load_weights)),
            'external_package':os.environ['SGLANG_EXTERNAL_MODEL_PACKAGE'],'fallback_allowed':False,
            'model_constructed':False,'NPU_kernel_invoked':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--model-path');a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise RuntimeError('Use a new gate evidence file')
    spec=importlib.util.find_spec('acl')
    d={'PYTHONPATH':os.environ.get('PYTHONPATH'),'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH'),
       'ASCEND_HOME_PATH':os.environ.get('ASCEND_HOME_PATH'),'sys_path':sys.path,'acl_spec':None if spec is None else spec.origin,
       'gate_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        d.update(check(a.model_path));d['passed']=True
    except Exception as exc:
        d.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,out)
    print(json.dumps(d,indent=2),flush=True);sys.exit(0 if d['passed'] else 1)

if __name__=='__main__':main()
