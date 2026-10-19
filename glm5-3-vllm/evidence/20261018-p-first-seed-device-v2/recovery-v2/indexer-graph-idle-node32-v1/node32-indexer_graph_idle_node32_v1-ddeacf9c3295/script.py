from pathlib import Path
import json,socket,subprocess
node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[socket.gethostname()]
cid={32:'1a7c22de29006ffd698632527dacd522e746350770710e7f45a2ebacded857dc',33:'163c7646c30ab819514f112e0af510934aa592305f3ec37bd7e3b77a3edc2bdc'}[node]
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['Id']==cid and not d['State']['Running'] and not d['State']['OOMKilled']
if node==32:
 assert d['State']['ExitCode']==124
 p=json.loads(subprocess.check_output(['docker','inspect','9941c407aa419dd82372f8d92906180a21e0633edc0f782d15ecfc7de03841b4']))[0]
 assert not p['State']['Running'] and p['State']['ExitCode']==124 and not p['State']['OOMKilled']
s=subprocess.check_output(['npu-smi','info'],text=True,timeout=30);assert s.count('No running processes found in NPU')==8
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'completed':True,'node':node,'old_exact_CID':cid,'model_running':False,'model_exit':d['State']['ExitCode'],'empty_physical_cards':8,'no_stop_issued':True})+'\nGLM53_RESULT_JSON_END')
