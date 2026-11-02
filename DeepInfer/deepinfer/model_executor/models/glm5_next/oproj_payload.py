# SPDX-License-Identifier: Apache-2.0
"""Bounded signed-INT8 input-column O-projection payload;no CPU INT8 compute.

Checkpoint layout is [output,input]. Input columns follow contiguous rank heads;
output rows and per-output scales remain complete. No dequantization,GEMM,
activation quantization,TP sum,native import/registration or device transfer.
"""
from dataclasses import asdict,dataclass
import ast
import hashlib
import json
import torch
from .sfa_inputs import SFAHeadPartition,NoPESFAContract,NoPESFAInputPlan


def _positive(v,name):
    if type(v) is not int or v<1:raise ValueError('positive '+name+' required')


def _desc(reader,name,dtype,shape):
    path,start,d=reader.descriptor(name);offsets=d.get('data_offsets');element=1 if dtype=='I8' else 4
    if d.get('dtype')!=dtype or tuple(d.get('shape',()))!=shape or any(type(v) is not int for v in d.get('shape',())):raise ValueError('payload descriptor dtype/shape differs')
    if type(start) is not int or start<0 or not isinstance(offsets,(list,tuple)) or len(offsets)!=2 or any(type(v) is not int for v in offsets) or not 0<=offsets[0]<=offsets[1] or offsets[1]-offsets[0]!=shape[0]*shape[1]*element:raise ValueError('payload descriptor byte range differs')
    return path,start,d


def read_i8_columns_cpu(reader,name,first,stop,*,max_bytes=8*1024**2):
    """Read only the requested byte interval per row;never read full-width rows.

    Generic CPU byte reader,not a matrix()/FP4 decode path. max_bytes and the
    existing reader limit bound the TOTAL selected bytes,not just one I/O call.
    Partial reads fail without returning a partial payload. No files are written.
    """
    _positive(max_bytes,'column byte budget');_positive(reader.max_read_bytes,'reader bound')
    path,start,d=reader.descriptor(name);shape=d.get('shape')
    if not isinstance(shape,(list,tuple)) or len(shape)!=2 or any(type(v) is not int or v<1 for v in shape):raise ValueError('positive matrix shape required')
    path,start,d=_desc(reader,name,'I8',tuple(shape));rows,width=shape
    if type(first) is not int or type(stop) is not int or not 0<=first<stop<=width:raise ValueError('explicit nonempty input column interval required')
    size=rows*(stop-first)
    if size>min(max_bytes,reader.max_read_bytes):raise ValueError('column payload byte budget exceeded')
    data=bytearray();begin=start+d['data_offsets'][0]
    with path.open('rb') as f:
        for row in range(rows):
            f.seek(begin+row*width+first);part=f.read(stop-first)
            if len(part)!=stop-first:raise EOFError('incomplete selected INT8 columns')
            data.extend(part)
    return torch.frombuffer(data,dtype=torch.int8).reshape(rows,stop-first)


@dataclass(frozen=True)
class OProjColumnManifest:
    layer_id: int
    heads: SFAHeadPartition
    contract: NoPESFAContract
    prefix: str
    full_shape: tuple[int,int]
    first_column: int
    stop_column: int
    weight_bytes: int
    checkpoint_bytes: int
    runtime_bytes: int
    descriptor_sha256: str

    def to_dict(self):return asdict(self)


def o_proj_column_manifest(reader,config,layer_id,heads,contract,*,max_checkpoint_bytes=8*1024**2,max_runtime_bytes=8*1024**2):
    """All three tensor descriptors and budgets before any payload is read."""
    _positive(max_checkpoint_bytes,'checkpoint bound');_positive(max_runtime_bytes,'runtime bound');_positive(reader.max_read_bytes,'reader bound')
    if type(heads) is not SFAHeadPartition or type(contract) is not NoPESFAContract or heads.total_heads!=contract.total_heads:raise ValueError('explicit matching head/source contract required')
    if heads.local_heads>4:raise ValueError('bounded CPU payload supports at most4 local heads;no full matrix read')
    if NoPESFAContract.from_config(config,reference_source_sha256=contract.reference_source_sha256)!=contract or config.hidden_size!=4096 or config.mla_dimensions['v_head_dim']!=256:raise ValueError('GLM O-projection geometry differs')
    if type(layer_id) is not int or not 0<=layer_id<len(config.layers) or config.layers[layer_id].layer_id!=layer_id or config.layers[layer_id].attention!='deepseek_sparse_attention':raise ValueError('backbone DSA layer required')
    prefix=f'model.language_model.layers.{layer_id}.self_attn.o_proj';shape=(4096,64*256);first=heads.head_ids[0]*256;stop=(heads.head_ids[-1]+1)*256
    weight_bytes=shape[0]*(stop-first);checkpoint=weight_bytes+shape[0]*8;resident=weight_bytes+shape[0]*2
    if checkpoint>max_checkpoint_bytes or resident>max_runtime_bytes or weight_bytes>reader.max_read_bytes or shape[0]*4>reader.max_read_bytes:raise ValueError('O-projection checkpoint/runtime byte budget exceeded')
    source=[]
    for suffix,dtype,dims in (('weight','I8',shape),('weight_scale','F32',(4096,1)),('weight_offset','F32',(4096,1))):
        name=prefix+'.'+suffix
        if reader.quant.get(name)!='W8A8_DYNAMIC':raise ValueError('explicit W8A8_DYNAMIC weight/channel recipe required')
        path,start,d=_desc(reader,name,dtype,dims);source.append({'name':name,'path':str(path),'start':start,'dtype':dtype,'shape':list(dims),'offsets':list(d['data_offsets'])})
    digest=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return OProjColumnManifest(layer_id,heads,contract,prefix,shape,first,stop,weight_bytes,checkpoint,resident,digest)


@dataclass(frozen=True)
class PreparedOProjColumns:
    manifest: OProjColumnManifest
    weight_nk: torch.Tensor
    weight_scale: torch.Tensor
    weight_sha256: str
    source_scale_sha256: str
    source_offset_sha256: str

    @torch.inference_mode()
    def prepare_input_cpu(self,head_values,*,plan,max_input_bytes=2*1024**2):
        """Copy BF16 rank head values to[T,Klocal];no dynamic quant or GEMM."""
        m=self.manifest;_positive(max_input_bytes,'input byte bound')
        if type(plan) is not NoPESFAInputPlan or (plan.layer_id,plan.heads,plan.contract)!=(m.layer_id,m.heads,m.contract):raise ValueError('O-projection layer/head/source differs from SFA plan')
        shape=(plan.candidate.tokens,m.heads.local_heads,256)
        if not isinstance(head_values,torch.Tensor) or head_values.device.type!='cpu' or head_values.dtype!=torch.bfloat16 or head_values.shape!=shape:raise ValueError('plan-matched CPU BF16 head values required')
        if head_values.numel()*2>max_input_bytes or not bool(torch.isfinite(head_values).all()):raise ValueError('input byte bound/nonfinite input')
        return head_values.reshape(shape[0],m.stop_column-m.first_column).contiguous().clone()


@torch.inference_mode()
def load_o_proj_columns_cpu(reader,config,layer_id,heads,contract,*,max_checkpoint_bytes=8*1024**2,max_runtime_bytes=8*1024**2):
    """Validate small symmetric channels BEFORE reading selected INT8 columns.

    Full output-channel scales round once to BF16 and stay unsharded. Zero
    offsets are verified then discarded;this matches the prepared symmetric
    compute contract,not proof of native quantization/TP numerical equivalence.
    Metadata recheck detects descriptor/recipe drift,not same-metadata mutation.
    """
    bounds=dict(max_checkpoint_bytes=max_checkpoint_bytes,max_runtime_bytes=max_runtime_bytes)
    m=o_proj_column_manifest(reader,config,layer_id,heads,contract,**bounds)
    scale,sd=reader.tensor(m.prefix+'.weight_scale');offset,od=reader.tensor(m.prefix+'.weight_offset')
    if any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' or t.dtype!=torch.float32 or t.shape!=(4096,1) for t in (scale,offset)) or sd!='F32' or od!='F32':raise ValueError('full F32 output channel scale/offset required')
    if not bool(torch.isfinite(scale).all()) or not bool((scale>0).all()) or not bool(torch.isfinite(offset).all()) or bool((offset!=0).any()):raise ValueError('positive finite scale and symmetric zero offset required')
    rounded=scale.flatten().bfloat16().clone()
    if not bool(torch.isfinite(rounded).all()) or not bool((rounded>0).all()):raise ValueError('channel scale invalid after BF16 rounding')
    weight=read_i8_columns_cpu(reader,m.prefix+'.weight',m.first_column,m.stop_column,max_bytes=m.weight_bytes)
    if weight.device.type!='cpu' or weight.dtype!=torch.int8 or weight.shape!=(4096,m.stop_column-m.first_column):raise ValueError('selected signedINT8 column payload differs')
    if o_proj_column_manifest(reader,config,layer_id,heads,contract,**bounds)!=m:raise ValueError('checkpoint metadata changed during column load')
    def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    return PreparedOProjColumns(m,weight,rounded,digest(weight),digest(scale),digest(offset))


def inspect_row_parallel_sources(linear_source,quant_source):
    """AST-only conservative source-role ledger;no imports or ABI attestation."""
    def cls(tree,name):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    def method(c,name):return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    def calls(tree,name):return [n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
    r=cls(ast.parse(linear_source),'RowParallelLinear');q=cls(ast.parse(quant_source),'AscendW8A8DynamicLinearMethod')
    load=method(r,'weight_loader');n=calls(load,'loaded_weight.narrow')
    if len(n)!=1 or [ast.unparse(v) for v in n[0].args]!=['input_dim','start_idx','shard_size']:raise ValueError('row-parallel input shard loader source drift')
    f=method(r,'forward');apply=calls(f,'self.quant_method.apply');reduce=calls(f,'tensor_model_parallel_all_reduce')
    if len(apply)!=1 or [ast.unparse(v) for v in apply[0].args]!=['self','input_parallel','bias_'] or len(reduce)!=1 or [ast.unparse(v) for v in reduce[0].args]!=['output_parallel'] or apply[0].lineno>=reduce[0].lineno:raise ValueError('rank apply/reduce source roles drift')
    weight=calls(method(q,'get_weight'),'torch.empty')
    if len(weight)!=1 or [ast.unparse(v) for v in weight[0].args]!=['output_size','input_size'] or {k.arg:ast.unparse(k.value) for k in weight[0].keywords}!={'dtype':'torch.int8'}:raise ValueError('INT8 checkpoint NK layout source drift')
    channels=calls(method(q,'get_perchannel_param'),'torch.empty')
    if len(channels)!=2 or any([ast.unparse(v) for v in n.args]!=['output_size','1'] for n in channels):raise ValueError('per-output channel source drift')
    f=method(q,'apply');dynamic=calls(f,'torch_npu.npu_dynamic_quant');matmul=calls(f,'torch_npu.npu_quant_matmul')
    if len(dynamic)!=1 or [ast.unparse(v) for v in dynamic[0].args]!=['x'] or len(matmul)!=1 or [ast.unparse(v) for v in matmul[0].args]!=['quantized_x','layer.weight','layer.weight_scale']:raise ValueError('W8A8 dynamic source roles drift')
    if not any(isinstance(n,ast.Call) and ast.unparse(n.func)=='isinstance' and [ast.unparse(v) for v in n.args]==['x','PreparedLinearInput'] for n in ast.walk(f)):raise ValueError('prepared input bypass source drift')
    return {'source_only':True,'native_ABI_validated':False,'checkpoint_layout':'NK signedINT8','prepared_payload_shard':'input columns;all output rows/channels retained','rank_apply_input':'input_parallel','sum':'conditional after rank apply','prepared_quantized_input_bypass':True,'activation_quantization_numerics':'unverified;supports dynamic rank input or PreparedLinearInput','parameter_input_dim_binding':'not inspected;column axis inferred from NK layout and partitioned input size','native_dispatch':False}
