"""Bind native producer IDs to owned API/worker epochs using readonly GET_META."""
import json,subprocess,shlex
from pathlib import Path
from phase_runner import atomic_json,utc
QUERY="from pathlib import Path\nimport ast,json,hashlib,zmq,msgspec,sys\nf=Path(\"/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py\");raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()==\"f30572890014d0244bf31d8e15dc5d93fb84147df37e0b2c1d1b01cbba6dd533\"\nt=ast.parse(raw);msg=next(ast.literal_eval(n.value)for n in t.body if isinstance(n,ast.Assign)and any(isinstance(z,ast.Name)and z.id==\"GET_META_MSG\"for z in n.targets));assert msg==b\"get_meta_msg\"\nx=json.loads(sys.argv[1]);dest=Path(sys.argv[2]);ctx=zmq.Context();rows=[]\ntry:\n for tp in range(x[\"tp\"]):\n  port=x[\"base\"]+tp;sock=ctx.socket(zmq.REQ);sock.setsockopt(zmq.RCVTIMEO,10000);sock.setsockopt(zmq.SNDTIMEO,10000)\n  try:\n   sock.connect(\"tcp://172.16.10.\"+x[\"node\"]+\":\"+str(port));sock.send(msgspec.msgpack.encode((msg,)));data=sock.recv()\n   target=dest/(x[\"key\"]+\"_TP\"+str(tp)+\".agent.bin\");target.write_bytes(data);m=msgspec.msgpack.decode(data)\n   assert m[\"handshake_port\"]==port and m[\"local_ip\"]==\"172.16.10.\"+x[\"node\"]\n   rows.append(dict(tp=tp,port=port,engine_id=m[\"engine_id\"],num_blocks=m[\"num_blocks\"],block_size=m[\"block_size\"],raw=dict(path=str(target),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),native_source_sha256=hashlib.sha256(raw).hexdigest()))\n  finally:sock.close(linger=0)\nfinally:ctx.term()\nprint(json.dumps(rows))\n"
CHECK="from pathlib import Path\nimport json,subprocess,sys\no=json.loads(sys.argv[1]);ports=json.loads(sys.argv[2]);p=Path(\"/proc/\"+str(o[\"pid\"]));z=(p/\"stat\").read_text();boot=Path(\"/proc/sys/kernel/random/boot_id\").read_text().strip()\nassert dict(boot_id=boot,start_ticks=z[z.rfind(\")\")+2:].split()[19])==o[\"identity\"]\nassert [v.decode()for v in(p/\"cmdline\").read_bytes().split(bytes([0]))if v]==o[\"argv\"]\nls=subprocess.check_output([\"ss\",\"-ltnp\"],text=True).splitlines();rows=[]\nfor port in ports:\n hits=[l for l in ls if len(l.split())>3 and l.split()[3].endswith(\":\"+str(port))];assert len(hits)==1\n pid=int(hits[0].split(\"pid=\",1)[1].split(\",\",1)[0]);cur=pid;seen=set()\n while cur!=o[\"pid\"]:\n  assert cur>1 and cur not in seen;seen.add(cur);s=Path(\"/proc/\"+str(cur)+\"/stat\").read_text();cur=int(s[s.rfind(\")\")+2:].split()[1])\n s=Path(\"/proc/\"+str(pid)+\"/stat\").read_text()\n rows.append(dict(port=port,pid=pid,boot_id=boot,start_ticks=s[s.rfind(\")\")+2:].split()[19],P_API_root=o[\"pid\"]))\nprint(json.dumps(rows))\n"
VERIFY="from pathlib import Path\nimport json,sys\nboot=Path(\"/proc/sys/kernel/random/boot_id\").read_text().strip()\nfor x in json.loads(sys.argv[1]):\n p=Path(\"/proc/\"+str(x[\"pid\"]));s=(p/\"stat\").read_text();assert boot==x[\"boot_id\"] and s[s.rfind(\")\")+2:].split()[19]==x[\"start_ticks\"],\"registered P worker identity changed\"\nprint(\"verified\")\n"
def host(node,argv,timeout=40):
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)]
 return subprocess.check_output(argv,timeout=timeout)
def collect(run,owners,planned):
 r=Path(run);dest=r/"native_registered_producers";dest.mkdir(exist_ok=True)
 producers={}
 for key,o in owners.items():
  if o["role"]!="P":continue
  x=next(z for z in planned if z["role"]=="P"and z["rank"]==o["rank"])
  cfg=json.loads(x["argv"][x["argv"].index("--kv-transfer-config")+1]);tp=int(x["argv"][x["argv"].index("--tensor-parallel-size")+1]);base=cfg["kv_port"]+o["rank"]*tp
  rows=json.loads(subprocess.check_output(["docker","exec","glm52-single","python3","-c",QUERY,json.dumps(dict(tp=tp,base=base,node=o["host"],key=key)),str(dest)],timeout=80))
  physical=json.loads(host(o["host"],["python3","-c",CHECK,json.dumps(o),json.dumps([base+i for i in range(tp)])]))
  assert len(rows)==len(physical)==tp and len({z["engine_id"]for z in rows})==1 and len({z["num_blocks"]for z in rows})==1 and len({z["block_size"]for z in rows})==1
  actual=rows[0]["engine_id"];assert actual and actual.startswith(cfg["engine_id"]+"-")and actual.endswith("_dp"+str(o["rank"])),"native observed producer ID outside ownseed/rank"
  producers[key]=dict(at=utc(),engine_id=actual,CLI_engine_id_seed=cfg["engine_id"],API_owner=o,num_blocks=rows[0]["num_blocks"],block_size=rows[0]["block_size"],worker_handshakes=rows,physical_workers=physical,contract="Actualreadonly nativeGET_META acrossallTP workers plus socketPID/APIroot ancestry; no metadata identity rewrite/inference/transfer/DONE/free")
 assert set(producers)=={"P0","P1"}and producers["P0"]["engine_id"]!=producers["P1"]["engine_id"]
 atomic_json(r/"registered_P_engine_identities.json",producers);return producers
def verify_registered(producers):
 for p in producers.values():
  assert host(p["API_owner"]["host"],["python3","-c",VERIFY,json.dumps(p["physical_workers"])]).strip()==b"verified"
