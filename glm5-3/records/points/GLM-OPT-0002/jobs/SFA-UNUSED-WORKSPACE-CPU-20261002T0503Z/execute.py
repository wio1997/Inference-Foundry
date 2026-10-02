from pathlib import Path
import sys,json,types,copy,hashlib
j=Path(__file__).parent;sys.path.insert(0,str(j))
import sfa_workspace_guard as task
import torch
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.engine.arg_utils import EngineArgs
from vllm.config import set_current_vllm_config
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder
import vllm.model_executor.layers.attention.mla_attention as mla
import vllm_ascend.attention.context_parallel.common_cp as cp
p=j.parents[1];planned=json.loads((p/"runs/GLM-RUN-0052/planned_launch.json").read_text())
def config(role):
 argv=next(x["argv"]for x in planned if x["role"]==role and x["rank"]==0)
 parser=FlexibleArgumentParser();sub=parser.add_subparsers(dest="subparser");ServeSubcommand().subparser_init(sub);args=parser.parse_args(argv[3:])
 if args.model_tag is not None:args.model=args.model_tag
 return EngineArgs.from_cli_args(args).create_engine_config()
configs={x:config(x)for x in["P","D"]}
class Clone:
 def clone(self):return self
def setup(cfg):
 cfg.compilation_config.static_forward_context["CPU.fixture"]=types.SimpleNamespace(prefill_backend=Clone())
 cfg.cache_config.block_size=cfg.cache_config.block_size or 128
 return cfg
cfg=setup(configs["D"]);fake=types.SimpleNamespace(world_size=16,rank_in_group=0);old_mla=mla.get_dcp_group;old_cp=cp.get_dcp_group;mla.get_dcp_group=lambda:fake;cp.get_dcp_group=lambda:fake
def construct(cfg,cls):
 spec=MLAAttentionSpec(block_size=cfg.cache_config.block_size,num_kv_heads=1,head_size=cfg.model_config.get_head_size(),dtype=cfg.model_config.dtype)
 with set_current_vllm_config(cfg):return cls(spec,["CPU.fixture"],cfg,torch.device("cpu"))
try:
 before=construct(cfg,AscendSFADCPMetadataBuilder);native_bytes=before.chunked_prefill_workspace.numel()*before.chunked_prefill_workspace.element_size();assert native_bytes>100*1024**2
 task.install()
 after=construct(cfg,AscendSFADCPMetadataBuilder);buf=vars(after)["chunked_prefill_workspace"];assert buf.numel()==0 and vars(after)["chunked_prefill_workspace_size"]==0
 for field in task.FIELDS:
  try:getattr(after,field)
  except RuntimeError as e:assert str(e).startswith("GLM_UNUSED_SFA_WORKSPACE_READ:")
  else:raise AssertionError("workspace runtime poison readguard missing")
 original=construct(cfg,mla.MLACommonMetadataBuilder);assert original.chunked_prefill_workspace.numel()>0
 pcfg=setup(configs["P"]);pafter=construct(pcfg,AscendSFADCPMetadataBuilder);assert vars(pafter)["chunked_prefill_workspace"].numel()==0
 draft=configs["D"].speculative_config.draft_model_config
 out=dict(native_sources=task.proof(),baseline_DCP_CPU_workspace_bytes=native_bytes,DCP_fixture=dict(world_size=16,rank=0),actual_NPU_workers=0,device_tensor_allocations=0,models=0,inference=0,signals=0,positive=dict(D_empty=buf.numel()==0,P_empty=vars(pafter)["chunked_prefill_workspace"].numel()==0),negative=dict(ordinary_MLA_preserved=original.chunked_prefill_workspace.numel()>0,unexpected_runtime_workspace_reads_fail_closed=True),architectures=dict(P=configs["P"].model_config.hf_config.architectures,D=configs["D"].model_config.hf_config.architectures,draft=draft.hf_config.architectures,draft_model_type=draft.hf_config.model_type),limits=["Actual native constructors on CPU withsyntheticDCPgroup/staticforwardfixture; not nativeGPUfit/Graph/PD/functional certificate","Guard source/liveness scopedSFA; ordinaryMLA readsbuffer andretainsnativeallocation","No native operator source or KVdata/layout changed"])
 (j/"reduction.json").write_text(json.dumps(out,indent=2)+"\n")
 b=(j/"reduction.json").read_bytes();(j/"result.json").write_text(json.dumps(dict(schema_version=1,job_id=j.name,status="completed",summary="NativeSFA CPU constructors: targetP/D zero unusedworkspace, poisonreads failclosed, ordinaryMLA retainsallocation; syntheticDCP16/noNPUworkers",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualnativeCPUconstructors/shape/source/readguard/MLAnegative")],unknowns=out["limits"],decision_request=None,next_check_at=None),indent=2)+"\n");print(json.dumps(out))
finally:mla.get_dcp_group=old_mla;cp.get_dcp_group=old_cp

