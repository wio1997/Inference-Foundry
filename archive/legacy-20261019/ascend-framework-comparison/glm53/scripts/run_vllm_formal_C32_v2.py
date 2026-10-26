"""Once-only uniform v2 vLLM baseline; no profiling."""
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'config/formal-runs-v2/vllm-C32.json';assert hashlib.sha256(path.read_bytes()).hexdigest()=='f441ef4c68e8f92752c99d35171a73be184a7bf7854145ef0b57c3dc28c16021';settings=json.loads(path.read_text());assert settings['harness_revision']=='v2-uniform-fresh-upstream-HTTP-no-retries'
 cert=ROOT/settings['native_qualification_path'];assert hashlib.sha256(cert.read_bytes()).hexdigest()==settings['native_qualification_sha256'];proof=json.loads(cert.read_text());assert proof['pass']
 for file,sha in proof['evidence_sha256'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha
 path=ROOT/'scripts/glm53_formal_resource_controller_v1.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='d070d50197036420fb07f05a2a595155db3360de736d6c93cfc0847c40fa3404'
 run=ROOT/'runs'/settings['run_id'];run.mkdir(parents=True,exist_ok=True)
 with (run/'execution-once.lock').open('x') as file:file.write(str(__import__('os').getpid()))
 spec=importlib.util.spec_from_file_location('controller',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);print('GLM53_FORMAL_RESOURCE_COMPLETE='+json.dumps(m.execute(settings)),flush=True)
if __name__=='__main__':main()
