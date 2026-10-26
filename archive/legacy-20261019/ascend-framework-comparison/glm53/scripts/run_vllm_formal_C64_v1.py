"""Once-only real AIS C64 run; no profile, baseline requires postrun native audit."""
import hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1];RUN=ROOT/'runs/20261017T1810-vllm-formal-C64-v1'
SETTINGS=ROOT/'config/formal-runs-v1/vllm-C64.json';SETTINGS_SHA='bdfe7710399a76e82ebbdcdd6ac59e20fdefb5b26c411df3da20aa35c8b2f5cb'
def main():
 assert hashlib.sha256(SETTINGS.read_bytes()).hexdigest()==SETTINGS_SHA
 settings=json.loads(SETTINGS.read_text());assert settings['run_id']==RUN.name and settings['framework']=='vllm' and settings['concurrency']==64
 cert=ROOT/settings['native_qualification_path'];assert hashlib.sha256(cert.read_bytes()).hexdigest()==settings['native_qualification_sha256'];proof=json.loads(cert.read_text());assert proof['pass']
 for file,sha in proof['evidence_sha256'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha
 path=ROOT/'scripts/glm53_formal_resource_controller_v1.py';assert hashlib.sha256(path.read_bytes()).hexdigest()==settings['resource_controller_sha256']
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as lock:lock.write(str(__import__('os').getpid()))
 spec=importlib.util.spec_from_file_location('controller',path);controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
 result=controller.execute(settings);print('GLM53_FORMAL_RESOURCE_COMPLETE='+json.dumps(result),flush=True)
if __name__=='__main__':main()
