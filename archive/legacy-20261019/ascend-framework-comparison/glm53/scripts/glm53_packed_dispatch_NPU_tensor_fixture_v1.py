"""Bounded synthetic NPU byte-view/routing output qualification; never model tensors."""
import hashlib,json,sys,types
import torch,torch_npu
assert hashlib.sha256(PACK_SOURCE.encode()).hexdigest()=='336a64b8f6657541f24daa49588891072d350b4a0d9d94b383c6f3fe08cea5cb'
pack=types.ModuleType('glm53_NPU_synthetic_pack');pack.__file__='<task-owned-336a64b8-source>';exec(compile(PACK_SOURCE,pack.__file__,'exec'),pack.__dict__)
assert torch.npu.device_count()==1;torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(0.01,0)
assert not torch.distributed.is_initialized()
checks=[]
def raw(t):return t.detach().cpu().contiguous().view(torch.uint8)
def check(q,s,label):
 beforeq,before_s=raw(q).clone(),raw(s).clone()
 wire=pack.pack_payload(torch,q,s);tokens,scales=pack.unpack_payload(torch,wire,6144);torch.npu.synchronize()
 assert q.device.type=='npu' and wire.device==tokens.device==q.device and scales.device==s.device
 assert torch.equal(raw(tokens),beforeq) and torch.equal(raw(scales),before_s)
 assert torch.equal(raw(q),beforeq) and torch.equal(raw(s),before_s)
 assert tokens.shape==q.shape and tokens.dtype==q.dtype and scales.shape==(q.shape[0],) and scales.dtype==s.dtype
 checks.append({'case':label,'rows':q.shape[0],'hidden':6144,'quantized_token_dtype':str(q.dtype),'dynamic_scale_dtype':str(s.dtype),'token_strides':list(q.stride()),'token_contiguous':q.is_contiguous(),'wire_device':str(wire.device),'restored_token_device':str(tokens.device),'restored_scale_device':str(scales.device),'wire_shape':list(wire.shape),'pass':True,'before_token_bytes_SHA256':hashlib.sha256(beforeq.numpy().tobytes()).hexdigest(),'before_scale_bytes_SHA256':hashlib.sha256(before_s.numpy().tobytes()).hexdigest(),'inputs_not_mutated':True})
for n in [0,1,2,6,17]:
 q=((torch.arange(n*6144,dtype=torch.int64)*79+17)%256-128).to(torch.int8).reshape(n,6144).to('npu:0')
 unsigned=[0,0x80000000,0x3f800000,0xbf800000,0x7f800000,0xff800000,0x7fc00001,0x7fa12345,1,0x80000001,0x007fffff,0x7f7fffff,0xff7fffff,0x00800000,0xdeadbeef]
 bits=[unsigned[i%len(unsigned)] for i in range(n)];s=torch.tensor([v if v<2**31 else v-2**32 for v in bits],dtype=torch.int32).view(torch.float32).to('npu:0')
 check(q,s,'direct_special_bits_n'+str(n))
for n in [1,2,6]:
 x=(((torch.arange(n*6144,dtype=torch.float32)%31)-15)/8).to(torch.bfloat16).reshape(n,6144).to('npu:0');indices=torch.arange(n*8,dtype=torch.int32).reshape(n,8).to('npu:0')
 q,reverse,counts,s=torch_npu.npu_moe_init_routing_v2(x,indices,quant_mode=1,expert_num=256,expert_tokens_num_type=1,expert_tokens_num_flag=True,row_idx_type=0,active_expert_range=[0,256])
 assert q.shape==(n*8,6144) and q.dtype==torch.int8 and s.dtype==torch.float32
 check(q,s,'actual_native_MoeInitRoutingV2_int8_n'+str(n))
assert not any(n.startswith(('memcache_hybrid','memfabric_hybrid','sglang','deep_ep')) for n in sys.modules)
assert not torch.distributed.is_initialized();torch.npu.synchronize()
result={'project_id':'glm53','pass':True,'qualification':'SYNTHETIC_NPU_BYTE_VIEW_AND_NATIVE_ROUTING_OUTPUT_ONLY','checks':checks,'cases':8,'external_model_HTTP_requests':0,'SDK_init_calls':0,'model_tensor_values_observed':False,'native_HCCL_collectives_executed':False,'whole_model_correctness':'PENDING','torch_version':torch.__version__,'torch_npu_version':torch_npu.__version__,'helper_SHA256':'336a64b8f6657541f24daa49588891072d350b4a0d9d94b383c6f3fe08cea5cb','probe_process_Torch_allocator_fraction':0.01,'probe_process_peak_Torch_allocated_bytes':torch.npu.max_memory_allocated(0),'probe_process_peak_Torch_reserved_bytes':torch.npu.max_memory_reserved(0),'memory_limit_scope':'This probe process Torch caching allocator only; not total CANN/driver workspace or model allocation policy.','system_settings_changed':False}
print('GLM53_PACKED_NPU_TENSOR_FIXTURE='+json.dumps(result),flush=True)
