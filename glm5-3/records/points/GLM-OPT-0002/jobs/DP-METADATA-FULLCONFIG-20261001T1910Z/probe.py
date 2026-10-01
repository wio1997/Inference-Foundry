import json,pathlib,hashlib
from vllm.engine.arg_utils import EngineArgs
from vllm_ascend.ascend_config import get_ascend_config
rows=[]
for rank in [0,1]:
 a=EngineArgs(worker_cls="coupled_dp_worker.CoupledMetadataWorker",model="/data/tiankuan/wio/GLM-5.2-w8a8",trust_remote_code=True,quantization="ascend",seed=1024,max_model_len=144384,max_num_seqs=8,max_num_batched_tokens=4096,gpu_memory_utilization=.87,tensor_parallel_size=16,decode_context_parallel_size=16,prefill_context_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=2,data_parallel_size_local=1,data_parallel_rank=rank,data_parallel_external_lb=True,data_parallel_address="172.16.10.166",data_parallel_rpc_port=32620,data_parallel_backend="mp",distributed_executor_backend="mp",enable_expert_parallel=True,nnodes=2,node_rank=rank,master_addr="172.16.10.166",master_port=32630,enable_prefix_caching=True,enable_chunked_prefill=True,compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"},speculative_config={"num_speculative_tokens":5,"method":"deepseek_mtp","enforce_eager":True},additional_config={"multistream_overlap_shared_expert":True,"enable_dsa_cp":False,"enable_fused_mc2":0,"mc2_comm_alg":"hierarchy"})
 c=a.create_engine_config()
 pc=c.parallel_config
 # Config-only: no LLM/AsyncLLM/EngineCore/worker constructed.
 rows.append({"rank":rank,"worker_cls":pc.worker_cls,"parallel_hash":pc.compute_hash(),"world":pc.world_size_across_dp,"local_world":pc.local_world_size,"model_architectures":c.model_config.architectures,"max_model_len":c.model_config.max_model_len,"max_num_batched_tokens":c.scheduler_config.max_num_batched_tokens,"max_num_seqs":c.scheduler_config.max_num_seqs,"kv_transfer_config":None if c.kv_transfer_config is None else str(c.kv_transfer_config),"additional_config":c.additional_config,"num_speculative_tokens":c.speculative_config.num_speculative_tokens,"graph_mode":str(c.compilation_config.cudagraph_mode)})
assert all(row["worker_cls"]=="coupled_dp_worker.CoupledMetadataWorker" for row in rows)
assert rows[0]["parallel_hash"]==rows[1]["parallel_hash"] and all(r["world"]==32 and r["local_world"]==16 and r["kv_transfer_config"] is None and r["num_speculative_tokens"]==5 for r in rows)
print(json.dumps({"full_native_engine_config":rows,"model_requests":0,"weights_loaded":0,"EngineCore_or_worker_created":False,"limitations":["Actual installed EngineArgs.create_engine_config validated; no engine/NPU collective/model weight or inference created","Live memory/HCCL/Graph/API behavior still untested"]}))
