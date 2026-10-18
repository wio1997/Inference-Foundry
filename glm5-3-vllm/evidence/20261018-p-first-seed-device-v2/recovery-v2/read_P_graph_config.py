import pathlib,json,socket,subprocess,urllib.request,re
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
cid='4cb8a338da590f0d0fd624892b2af7a20884029c7418c05ada75f5dbf3bcb157'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running']
logs=subprocess.check_output(['docker','logs',cid],stderr=subprocess.STDOUT,timeout=20).decode(errors='replace');selected=[l for l in logs.splitlines() if any(k in l.lower() for k in ['cudagraph','compilation_config','graph capture','capturing acl','capture sizes'])];assert sum(map(len,selected))<1000000
server={}
try:
 with urllib.request.urlopen('http://172.16.12.32:29481/server_info?config_format=json',timeout=5) as response:
  raw=json.load(response)
  if isinstance(raw,dict):
   for key in ['vllm_config','compilation_config','parallel_config','scheduler_config']:
    if key in raw:server[key]=raw[key]
except Exception as e:server={'available':False,'error':type(e).__name__}
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'graph_log_lines':selected,'server_config':server,'actual_container_args':d['Args']})+'\nGLM53_RESULT_JSON_END')
