"""Validate frozen SG run order and evidence before any resource call."""
import hashlib,importlib.util,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTROLLER_SHA='d070d50197036420fb07f05a2a595155db3360de736d6c93cfc0847c40fa3404'
def checked(root,relative,sha):
    assert re.fullmatch('[0-9a-f]{64}',sha),'Missing actual frozen SHA'
    path=(root/relative).resolve();assert path.is_relative_to(root.resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,(relative,'SHA mismatch')
    return path

def validate(root,settings_path,expected_sha):
    path=checked(root,settings_path,expected_sha);settings=json.loads(path.read_text())
    assert settings['project_id']=='glm53' and settings['framework']=='sglang'
    assert settings['status']=='FROZEN_READY_FOR_SINGLE_EXECUTION'
    C=settings['concurrency'];assert C in [16,32,64]
    rid=settings['run_id'];assert Path(rid).name==rid and rid.startswith('20261017T') and f'-sglang-formal-C{C}-' in rid
    assert settings['namespace']==f'glm53-formal-{rid}-C{C}'
    required={f'vllm-C{c}':('vllm',c) for c in [16,32,64]}
    required.update({f'sglang-C{c}':('sglang',c) for c in [16,32,64] if c<C})
    assert set(settings['predecessor_formal_certificates'])==set(required),'All vLLM series and lower SG series required'
    for label,(fw,concurrency) in required.items():
        item=settings['predecessor_formal_certificates'][label]
        proof=json.loads(checked(root,item['path'],item['sha256']).read_text())
        assert proof['project_id']=='glm53' and proof['framework']==fw and proof['concurrency']==concurrency and proof['measurement_valid'] is True
    proof=json.loads(checked(root,settings['native_qualification_path'],settings['native_qualification_sha256']).read_text())
    assert proof['pass'] is True and proof['project_id']=='glm53' and proof['framework']=='sglang'
    assert proof['shared_run_id']==settings['shared_run_id']
    assert proof['runtime_CIDs']=={n:item['container_id'] for n,item in settings['runtime_containers'].items()}
    assert proof['source_evidence_sha256']
    for relative,sha in proof['source_evidence_sha256'].items():checked(root,relative,sha)
    assert settings['resource_controller_sha256']==CONTROLLER_SHA
    controller=checked(root,'scripts/glm53_formal_resource_controller_v1.py',CONTROLLER_SHA)
    return settings,controller

def execute(settings_path,expected_sha):
    settings,path=validate(ROOT,settings_path,expected_sha)
    run=ROOT/'runs'/settings['run_id'];run.mkdir(parents=True,exist_ok=True)
    with (run/'execution-once.lock').open('x') as lock:lock.write(str(__import__('os').getpid()))
    spec=importlib.util.spec_from_file_location('controller',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.execute(settings);print('GLM53_FORMAL_RESOURCE_COMPLETE='+json.dumps(result),flush=True)
    return result
