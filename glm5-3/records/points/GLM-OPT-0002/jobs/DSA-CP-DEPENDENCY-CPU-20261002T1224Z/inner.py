from pathlib import Path
import json,sys,hashlib,ast
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.entrypoints.openai.api_server import validate_api_server_args
from vllm.tool_parsers import ToolParserManager
from vllm.engine.arg_utils import EngineArgs
from vllm.config import set_current_vllm_config
j=Path(__file__).parent
paths=["/vllm-workspace/vllm-ascend/vllm_ascend/ascend_config.py","/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/sfa_cp.py","/vllm-workspace/vllm-ascend/vllm_ascend/attention/sfa_v1.py","/vllm-workspace/vllm-ascend/vllm_ascend/platform.py","/vllm-workspace/vllm-ascend/docs/source/user_guide/feature_guide/context_parallel.md","/vllm-workspace/vllm-ascend/docs/source/tutorials/models/GLM5.2.md"]
rows=[]
(j/"native_sources").mkdir()
for n,name in enumerate(paths):
 f=Path(name);raw=f.read_bytes();snapshot=j/"native_sources"/(str(n)+"_"+f.name);snapshot.write_bytes(raw)
 lines=[dict(line=i,text=l)for i,l in enumerate(raw.decode().splitlines(),1)if any(key in l for key in["enable_dsa_cp","AscendSFADSADCP","Speculative Decoding + PCP","SFA |","world_size_with_pcp","requires_tp_aligned_capture_sizes"])]
 rows.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),snapshot=str(snapshot),selected_lines=lines))
planned=json.loads(Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0069/planned_launch.json").read_text());configs=[]
for x in planned:
 argv=list(x["argv"]);idx=argv.index("--additional-config")+1;a=json.loads(argv[idx]);a["enable_dsa_cp"]=True;argv[idx]=json.dumps(a,separators=(",",":"));idx=argv.index("--compilation-config")+1;c=json.loads(argv[idx]);c["cudagraph_capture_sizes"]=[48];c["max_cudagraph_capture_size"]=48;argv[idx]=json.dumps(c,separators=(",",":"))
 parser=FlexibleArgumentParser();sub=parser.add_subparsers(dest="subparser");cmd=ServeSubcommand();cmd.subparser_init(sub);args=parser.parse_args(argv[3:]);cmd.validate(args)
 if args.model_tag is not None:args.model=args.model_tag
 ToolParserManager.import_tool_parser(args.tool_parser_plugin);validate_api_server_args(args);config=EngineArgs.from_cli_args(args).create_engine_config()
 with set_current_vllm_config(config):
  from vllm_ascend.utils import register_ascend_customop,enable_dsa_cp
  register_ascend_customop(config)
  from vllm_ascend.ascend_config import get_ascend_config
  asc=get_ascend_config()
  from vllm_ascend.attention.context_parallel.sfa_cp import resolve_sfa_metadata_builder,resolve_sfa_impl
  builder=resolve_sfa_metadata_builder();impl=resolve_sfa_impl(config)
  assert asc.enable_dsa_cp and enable_dsa_cp()and config.parallel_config.use_sequence_parallel_moe
  assert builder.__name__=="AscendSFADSADCPMetadataBuilder"and impl.__name__=="AscendSFADSADCPImpl"
  from sfa_workspace_guard import proof,eligible
  wp=proof();zero=eligible(object.__new__(builder),config);assert not zero
  from issue_budget_scheduler_v3 import proof as budget_proof
  bp=budget_proof(config)
  config.compilation_config.adjust_cudagraph_sizes_for_spec_decode(6,16)
  assert config.compilation_config.cudagraph_capture_sizes==[48]
  assert config.parallel_config.tensor_parallel_size==16 and config.parallel_config.decode_context_parallel_size==16 and config.parallel_config.prefill_context_parallel_size==1 and config.parallel_config.data_parallel_size==2
  assert config.speculative_config.num_speculative_tokens==5 and config.scheduler_config.async_scheduling
  assert config.scheduler_config.max_num_batched_tokens==16384 and config.cache_config.kv_cache_memory_bytes==13743895347
  configs.append(dict(rank=x["rank"],argv=argv,DSA_CP=asc.enable_dsa_cp,sequence_parallel_moe=config.parallel_config.use_sequence_parallel_moe,all2all_backend=config.parallel_config.all2all_backend,metadata_builder=builder.__module__+"."+builder.__name__,implementation=impl.__module__+"."+impl.__name__,capture_sizes=config.compilation_config.cudagraph_capture_sizes,native_budget_proof=bp,unused_workspace_guard_eligible=zero,workspace_sources=wp["sources"],native_models_created=0,communicators_created=0))
out=dict(sources=rows,configs=configs,models_started=0,inference_requests=0,model_signals=0,limits=["CPU nativeCLI/effectiveconfig/source/backendclass selection only, not instantiatedmetadata/GPU/HBMfit/E2E/KEEP","LegacyDSA_CP native composition DCP16 selected; experimentalPCP SFA/MTP/PD incompatible in installedguide and excluded from currentcandidate","Existing unusedSFAworkspace guard does not whitelist newDSADCPbuilder; candidate retains originalnative inheritedMLAworkspace allocation. No promisedzero-workspace memorygain","DSA_CP enables nativeSP and captureTP-alignment; [48] legalK5Geometry, realmultirank graph/metadata/correctness stillunproven"])
(j/"dependency_reduction.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(dict(event="DSA_CP_CPU_configuration_verified",configs=len(configs),DSADCP=True,capture=[48],workspace_guard_eligible=False,models=0,requests=0)))
