"""Once-only real AIS C16 run; no profile, baseline requires postrun native audit."""
import hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1];RUN=ROOT/'runs/20261017T1790-vllm-formal-C16-v1'
SETTINGS=ROOT/'config/formal-runs-v1/vllm-C16.json';SETTINGS_SHA='3e2106002d52de829b7fc1fcee6d352703afcb69411e65c80dc36cfb6f839877'
def main():
 assert hashlib.sha256(SETTINGS.read_bytes()).hexdigest()==SETTINGS_SHA
 settings=json.loads(SETTINGS.read_text());assert settings['run_id']==RUN.name and settings['framework']=='vllm' and settings['concurrency']==16
 cert=ROOT/settings['native_qualification_path'];assert hashlib.sha256(cert.read_bytes()).hexdigest()==settings['native_qualification_sha256'];proof=json.loads(cert.read_text());assert proof['pass']
 for file,sha in proof['evidence_sha256'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha
 path=ROOT/'scripts/glm53_formal_resource_controller_v1.py';assert hashlib.sha256(path.read_bytes()).hexdigest()==settings['resource_controller_sha256']
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as lock:lock.write(str(__import__('os').getpid()))
 spec=importlib.util.spec_from_file_location('controller',path);controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
 result=controller.execute(settings);print('GLM53_FORMAL_RESOURCE_COMPLETE='+json.dumps(result),flush=True)
if __name__=='__main__':main()
