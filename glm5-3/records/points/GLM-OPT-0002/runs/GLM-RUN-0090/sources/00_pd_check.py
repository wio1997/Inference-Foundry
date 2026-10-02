import asyncio,json,sys
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json
from loadgen import execute
root=Path(__file__).parent;d=root/'pd_compat';d.mkdir();dataset=Path("/data/tiankuan/wio/glm52-pd/deploy/results/standard_20261001_141018_3154006/c3_round1/formal_round_1/dataset/GSM8K-in81920-num12-GLM-5.2-w8a8-repeatRate0.9.jsonl")
prompt=json.loads(dataset.open().readline())['question'];payload={'model':'glm-52','messages':[{'role':'user','content':prompt}],'temperature':0,'seed':20260930,'ignore_eos':True,'max_tokens':256,'stream':True,'stream_options':{'include_usage':True}};body=d/'request.body.json';atomic_json(body,payload)
plan={'kind':'diagnostic','endpoint':'http://172.16.10.166:8000/v1/chat/completions','requests':[{'id':'pd0','arrival_s':0,'body_path':str(body),'timeout_s':240,'expected':{'prompt_tokens':81932,'output_tokens':256}}]};atomic_json(d/'plan.json',plan);result=asyncio.run(execute(plan,d));print(json.dumps({k:result[k] for k in ['valid','elapsed_s','effective_tps']}));sys.exit(0 if result['valid'] else 1)
