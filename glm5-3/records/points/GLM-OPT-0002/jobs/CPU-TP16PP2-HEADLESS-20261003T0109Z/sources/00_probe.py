import sys,json,hashlib,signal
from pathlib import Path
from vllm.utils.argparse_utils import FlexibleArgumentParser
import vllm.entrypoints.cli.serve as serve
from vllm.tool_parsers import ToolParserManager
argv=json.loads(sys.argv[1])
def parsed(a):
 p=FlexibleArgumentParser(); sub=p.add_subparsers(dest="subparser")
 cmd=serve.ServeSubcommand();cmd.subparser_init(sub);args=p.parse_args(a[3:]);cmd.validate(args)
 if args.tool_parser_plugin:ToolParserManager.import_tool_parser(args.tool_parser_plugin)
 return cmd,args
class CPUStop(Exception):pass
observations=[]
class NoWorkerExecutor:
 def __init__(self,config,monitor_workers):
  pc=config.parallel_config
  assert monitor_workers is False and pc.node_rank_within_dp==1
  assert pc.nnodes==pc.nnodes_within_dp==2 and pc.world_size==32 and pc.local_world_size==16
  assert pc.tensor_parallel_size==pc.decode_context_parallel_size==16 and pc.pipeline_parallel_size==2 and pc.data_parallel_size==1
  assert not pc.enable_expert_parallel and config.kv_transfer_config is None
  assert config.speculative_config.num_speculative_tokens==3
  observations.append(dict(branch="headless_executor",node_rank=pc.node_rank,world_size=pc.world_size,local_world_size=pc.local_world_size,master_addr=pc.master_addr,master_port=pc.master_port,monitor_workers=monitor_workers))
  raise CPUStop()
async def no_server(args,**kwargs):
 observations.append(dict(branch="API_server",node_rank=args.node_rank,headless=args.headless))
 raise CPUStop()
serve.MultiprocExecutor=NoWorkerExecutor
serve.run_server=no_server
def forbidden(*a,**k):raise AssertionError("EngineCore creation forbidden in CPU branch fixture")
serve.CoreEngineProcManager=forbidden
for rank,headless in [(0,False),(1,True),(1,False)]:
 a=argv[:];a[a.index("--node-rank")+1]=str(rank)
 if headless:a.append("--headless")
 cmd,args=parsed(a)
 old={s:signal.getsignal(s)for s in[signal.SIGTERM,signal.SIGINT]}
 try:
  try:cmd.cmd(args)
  except CPUStop:pass
  else:raise AssertionError("native CLI did not reach guarded CPU boundary")
 finally:
  for s,h in old.items():signal.signal(s,h)
assert [v["branch"]for v in observations]==["API_server","headless_executor","API_server"]
a=argv[:];a[a.index("--node-rank")+1]="1";a+=["--headless","--api-server-count","1"]
cmd,args=parsed(a)
try:cmd.cmd(args)
except ValueError as e:assert "cannot be" in str(e)and "--headless" in str(e)
else:raise AssertionError("headless/API conflict must fail before engine")
sources={}
for n in [serve.__file__,"/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py","/vllm-workspace/vllm/vllm/distributed/parallel_state.py","/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py"]:
 sources[n]=hashlib.sha256(Path(n).read_bytes()).hexdigest()
print(json.dumps(dict(event="native_multinode_CLI_branch_CPU_valid",observations=observations,headless_API_conflict_rejected=True,native_sources=sources,workers=0,EngineCore=0,ports=0,communicators=0,requests=0,limits=["Executor/API are intercepted CPU boundaries, no live distributed engine or communication proof","Both node configurations run inside166; remote167 engine readiness untested"])))
