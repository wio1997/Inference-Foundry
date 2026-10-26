"""Preserve exact bounded read-only SSH/pod mapping failures, no host or service mutations."""
import hashlib,importlib.util,json,shlex,socket,stat
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1];RUN=ROOT/'runs/20261017T2069-C32-control-channel-readonly-v1'
p=P.with_name('inventory_readonly.py');assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('inventory',p);base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 control=Path('/Users/wio/work/sglang-glm53-pd/ssh-control')
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'control_socket_exists':control.exists(),'control_socket_is_socket':control.exists() and stat.S_ISSOCK(control.stat().st_mode),'model_requests':0,'SDK_init_calls':0,'NPU_calls':0,'service_mutations':0,'commands':{}}
 try:
  for name,command in [('gateway-hostname','hostname'),('node34-pod-mapping','kubectl get pod -n default '+shlex.quote(base.NODES[34][0])+" -o jsonpath='{.status.hostIP}'")]:
   try:result=base.invoke([command],timeout=40)
   except Exception as error:result={'exception':repr(error)}
   (RUN/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n');receipt['commands'][name]=result
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
