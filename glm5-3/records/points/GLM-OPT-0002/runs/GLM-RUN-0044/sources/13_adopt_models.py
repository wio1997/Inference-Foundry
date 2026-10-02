import json,subprocess,sys,shlex,urllib.request
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
root=Path(__file__).parent;identities={}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for rank,node,port in [(0,"166","9081"),(1,"167","9081")]:
 def command(argv):
  if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)]
  p=subprocess.run(argv,capture_output=True,text=True,timeout=30);p.check_returncode();return p.stdout
 rows=[l for l in command(["docker","top","glm52-single","-eo","pid,ppid,comm,args"]).splitlines() if "/bin/vllm serve " in l and "--port "+port in l]
 assert len(rows)==1,rows
 pid=int(rows[0].split()[0]);code="import pathlib,json;p=pathlib.Path('/proc/"+str(pid)+"');s=(p/'stat').read_text();print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':s[s.rfind(')')+2:].split()[19],'argv':[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]}))"
 x=json.loads(command(["python3","-c",code]));args=x["argv"]
 fields={"--port":port,"--max-model-len":"144384","--max-num-seqs":"8","--max-num-batched-tokens":"16384","--gpu-memory-utilization":"0.8","--data-parallel-size":"1","--data-parallel-size-local":"1","--tensor-parallel-size":"32","--decode-context-parallel-size":"1","--prefill-context-parallel-size":"1","--pipeline-parallel-size":"1","--nnodes":"2","--node-rank":str(rank),"--data-parallel-address":"172.16.10.166","--data-parallel-rpc-port":"32620","--master-port":"32630","--tool-call-parser":"glm47_contract","--reasoning-parser":"glm45","--quantization":"ascend","--tool-parser-plugin":"/data/tiankuan/wio/glm52-pd/deploy/scripts/glm_tool_contract_run44.py"}
 assert all(args[args.index(k)+1]==v for k,v in fields.items())
 assert "--kv-transfer-config" not in args and "--data-parallel-external-lb" not in args and "--data-parallel-rank" not in args and args[args.index("--worker-cls")+1]=="atomic_mq_worker.AtomicMQWorker" and ("--headless" in args)==bool(rank) and "--enable-expert-parallel" in args
 assert json.loads(args[args.index("--speculative-config")+1])=={"num_speculative_tokens":5,"method":"deepseek_mtp","enforce_eager":True}
 assert json.loads(args[args.index("--compilation-config")+1])["cudagraph_mode"]=="FULL_DECODE_ONLY"
 additional=json.loads(args[args.index("--additional-config")+1]);assert additional["mc2_comm_alg"]=="hierarchy" and additional["enable_fused_mc2"]==0 and additional["enable_dsa_cp"] is False and additional["enable_reduce_sample"] is True
 identities["Node"+str(rank)]={"host":node,"pid":pid,"identity":{k:x[k] for k in ["boot_id","start_ticks"]},"argv":args,"execution_group":"DP1-TP32-DCP1-EP32-run44","native_role":"API and headless physical TP/EP members; one scheduler, not independent endpoints"}
 if rank==0:
  with opener.open("http://172.16.10."+node+":"+port+"/health",timeout=5) as response:assert response.status==200
atomic_json(root/"adopted_model_identities.json",identities)
print(json.dumps({"at":utc(),"identities":identities}))

prior=json.loads((root/"startup_model_identities.json").read_text())
for key,item in identities.items():
 assert all(item[k]==prior[key][k] for k in ["host","pid","identity","argv"]),"exact Run44 physical member changed"

epoch=__import__("hashlib").sha256(json.dumps({k:{f:v[f]for f in ["host","pid","identity","argv"]}for k,v in identities.items()},sort_keys=True).encode()).hexdigest()
atomic_json(root/"execution_groups.json",[{"id":"TP32-EP32-native-Run44","epoch":epoch,"members":["TP32"]}])
