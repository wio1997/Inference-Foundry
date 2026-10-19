"""Accepted prefix relay pipe with explicit framework-specific owned destination."""
import ast,hashlib,importlib.util,re,shlex
from pathlib import Path
source=Path(__file__).with_name('glm53_prefix_transfer_v1.py');assert hashlib.sha256(source.read_bytes()).hexdigest()=='688bf6a58d263c0ba1b897e9815a8ed8c0b2acc93a66d6669d541a6057d448b5'
spec=importlib.util.spec_from_file_location('old_transfer',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def render(framework,concurrency,manifest_sha,run_id,source_pod,destination_pod):
    assert framework in ['sglang','vllm'] and concurrency in [16,32,64] and re.fullmatch('[a-f0-9]{64}',manifest_sha)
    assert re.fullmatch('[A-Za-z0-9-]+',run_id) and f'-{framework}-formal-C{concurrency}-' in run_id
    shared,host=(('20261017T1754-vllm-shared-pool-v2','B4-15U-Atlas900RCKA3-30') if framework=='vllm' else ('20261017T1720-sglang-shared-pool-v2','B5-15U-Atlas900RCKA3-34'))
    destination='/data/wio/inference-foundry/glm53/runs/'+shared+'/formal-gates/'+run_id+'/data'
    receiver=old.RECEIVER.replace("assert socket.gethostname()=='B5-15U-Atlas900RCKA3-34'",'assert socket.gethostname()==DESTINATION_HOST')
    values={'CONCURRENCY':concurrency,'MANIFEST_SHA':manifest_sha,'DESTINATION':destination,'DESTINATION_HOST':host}
    def replace(text):
        for key,value in values.items():text=re.sub(r'\b'+key+r'\b',lambda _:repr(value),text)
        ast.parse(text);return text
    def command(pod,code,stdin=False):
        return shlex.join(['kubectl','exec',*(['-i'] if stdin else []),'-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',code])
    pipeline=command(source_pod,replace(old.SENDER))+' | '+command(destination_pod,replace(receiver),True)
    return shlex.join(['bash','-o','pipefail','-c',pipeline]),destination
