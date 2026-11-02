# SPDX-License-Identifier: Apache-2.0
"""Bounded q_b output-row payload and explicit CPU rank-query call contract.

Reuse ModelSlim.linear_payload(output_slice). No CPU INT8 math, native factory,
normalization, activation quantization, TP gather/sum or automatic backend. The
caller declares qr1536 already RMS-normalized; tensor content cannot attest it.
"""
from dataclasses import dataclass,asdict
import ast,hashlib,json
import torch
from .sfa_inputs import NoPESFAContract,NoPESFAInputPlan,SFAInputIdentity,SFAHeadPartition


def _positive(value,name):
    if type(value) is not int or value<1:raise ValueError('positive '+name+' required')


def _digest(value):return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class RankQBManifest:
    layer_id: int
    heads: SFAHeadPartition
    contract: NoPESFAContract
    prefix: str
    full_shape: tuple[int,int]
    first_row: int
    stop_row: int
    weight_bytes: int
    checkpoint_bytes: int
    runtime_bytes: int
    descriptor_sha256: str
    def to_dict(self):return asdict(self)


def rank_qb_manifest(reader,config,layer_id,heads,contract,*,max_checkpoint_bytes=4*1024**2,max_runtime_bytes=4*1024**2):
    """Preflight all descriptors/recipes/bounds before existing loader reads.

    Existing linear_payload reads full16384 F32 scale/offset channels,then slices
    scale rows. Budget counts those full channel reads;no replacement reader.
    Its value checks follow selected weight read,not before it.
    """
    for v,n in ((max_checkpoint_bytes,'checkpoint bytes'),(max_runtime_bytes,'runtime bytes'),(reader.max_read_bytes,'reader bytes')):_positive(v,n)
    if type(heads) is not SFAHeadPartition or type(contract) is not NoPESFAContract or heads.total_heads!=contract.total_heads or heads.local_heads>4:raise ValueError('matching bounded rank/source contract required')
    if config.hidden_size!=4096 or config.mla_dimensions['q_lora_rank']!=1536 or config.mla_dimensions['v_head_dim']!=256 or NoPESFAContract.from_config(config,reference_source_sha256=contract.reference_source_sha256)!=contract:raise ValueError('GLM q_b config geometry differs')
    if type(layer_id) is not int or not 0<=layer_id<len(config.layers) or config.layers[layer_id].layer_id!=layer_id or config.layers[layer_id].attention!='deepseek_sparse_attention':raise ValueError('backbone DSA layer required')
    prefix=f'model.language_model.layers.{layer_id}.self_attn.q_b_proj';shape=(16384,1536);first=heads.head_ids[0]*256;stop=(heads.head_ids[-1]+1)*256;weight=(stop-first)*1536;checkpoint=weight+16384*8;resident=weight+(stop-first)*2
    if checkpoint>max_checkpoint_bytes or resident>max_runtime_bytes or max(weight,16384*4)>reader.max_read_bytes:raise ValueError('rank q_b byte budget exceeded')
    source=[]
    for suffix,dtype,dims in (('weight','I8',shape),('weight_scale','F32',(16384,1)),('weight_offset','F32',(16384,1))):
        name=prefix+'.'+suffix
        if reader.quant.get(name)!='W8A8_DYNAMIC':raise ValueError('explicit W8A8_DYNAMIC weight/channel recipe required')
        path,start,d=reader.descriptor(name);offsets=d.get('data_offsets');element=1 if dtype=='I8' else 4
        if d.get('dtype')!=dtype or tuple(d.get('shape',()))!=dims or any(type(v) is not int for v in d.get('shape',())) or type(start) is not int or start<0 or not isinstance(offsets,(list,tuple)) or len(offsets)!=2 or any(type(v) is not int for v in offsets) or not 0<=offsets[0]<=offsets[1] or offsets[1]-offsets[0]!=dims[0]*dims[1]*element:raise ValueError('q_b descriptor dtype/shape/range differs')
        source.append({'name':name,'path':str(path),'start':start,'dtype':dtype,'shape':list(dims),'offsets':list(offsets)})
    descriptor=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return RankQBManifest(layer_id,heads,contract,prefix,shape,first,stop,weight,checkpoint,resident,descriptor)


@dataclass(frozen=True)
class CPURankQuery:
    identity: SFAInputIdentity
    query: torch.Tensor  # [T,local_heads,256], not full64 heads or absorbed512


@dataclass(frozen=True)
class RankQBCallSpec:
    identity: SFAInputIdentity
    input_shape: tuple[int,int]
    output_rows: tuple[int,int]
    rank: int
    world_size: int
    input_policy: str
    quantization_policy: str
    bias_policy: str
    descriptor_sha256: str
    weight_sha256: str
    runtime_scale_sha256: str


@dataclass(frozen=True)
class PreparedRankQB:
    manifest: RankQBManifest
    weight_nk: torch.Tensor
    weight_scale: torch.Tensor
    weight_sha256: str
    runtime_scale_sha256: str

    @torch.inference_mode()
    def project_cpu(self,query_rank,*,plan,expected_identity,backend,input_policy,quantization_policy,bias_policy,
                    max_input_bytes=512*1024,max_output_bytes=256*1024,max_payload_bytes=4*1024**2):
        """Already-normalized full qr1536 -> supplied rank backend -> Hlocalx256.

        Private input/codes/scales and owned output;no allgather/allreduce or RMS
        normalization. Only full_qr_dynamic_int8/absent bias declared here. The
        callback semantics/effects are not verified/rolled back by copies.
        Empty plans skip payload/input tensor access and backend after metadata
        and policy/budget checks. Logical bounds exclude transient/callback peak.
        """
        if type(plan) is not NoPESFAInputPlan or type(expected_identity) is not SFAInputIdentity or expected_identity!=plan.identity:raise ValueError('stale/foreign rank query identity')
        if input_policy!='rms_normalized_qr1536' or quantization_policy!='full_qr_dynamic_int8' or bias_policy!='absent':raise ValueError('explicit normalized full qr/dynamic INT8/absent bias policies required')
        if not callable(backend):raise ValueError('explicit rank q_b backend required')
        for v,n in ((max_input_bytes,'input bytes'),(max_output_bytes,'output bytes'),(max_payload_bytes,'payload bytes')):_positive(v,n)
        m=self.manifest;h=plan.heads.local_heads;t=plan.candidate.tokens;n=h*256
        if type(m) is not RankQBManifest or (m.layer_id,m.heads,m.contract)!=(plan.layer_id,plan.heads,plan.contract) or h>4:raise ValueError('q_b layer/partition/source differs')
        if (m.full_shape,m.first_row,m.stop_row,m.prefix,m.weight_bytes,m.checkpoint_bytes,m.runtime_bytes)!=((16384,1536),plan.heads.head_ids[0]*256,(plan.heads.head_ids[-1]+1)*256,f'model.language_model.layers.{plan.layer_id}.self_attn.q_b_proj',n*1536,n*1536+16384*8,n*1536+n*2):raise ValueError('rank q_b manifest geometry differs')
        if t*1536*2>max_input_bytes or t*n*2>max_output_bytes or m.runtime_bytes>max_payload_bytes:raise ValueError('rank q_b call byte budget exceeded')
        if plan.skip_operator:return CPURankQuery(plan.identity,torch.empty((0,h,256),dtype=torch.bfloat16))
        def require(value,shape,dtype,name):
            if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=dtype or tuple(value.shape)!=shape:raise ValueError('CPU '+name+' shape/dtype differs')
            if dtype!=torch.int8 and not bool(torch.isfinite(value).all()):raise ValueError('nonfinite '+name)
        require(query_rank,(t,1536),torch.bfloat16,'normalized qr');require(self.weight_nk,(n,1536),torch.int8,'q_b weight');require(self.weight_scale,(n,),torch.bfloat16,'local scale')
        if not bool((self.weight_scale>0).all()):raise ValueError('positive local scale required')
        x=query_rank.contiguous().clone();weight=self.weight_nk.contiguous().clone();scale=self.weight_scale.contiguous().clone()
        if _digest(weight)!=self.weight_sha256 or _digest(scale)!=self.runtime_scale_sha256:raise ValueError('q_b payload digest changed')
        spec=RankQBCallSpec(plan.identity,(t,1536),(m.first_row,m.stop_row),m.heads.rank,m.heads.world_size,input_policy,quantization_policy,bias_policy,m.descriptor_sha256,self.weight_sha256,self.runtime_scale_sha256)
        output=backend(x,weight,scale,spec=spec);require(output,(t,n),torch.bfloat16,'local q_b output')
        return CPURankQuery(plan.identity,output.contiguous().clone().reshape(t,h,256))


@torch.inference_mode()
def load_rank_qb_cpu(reader,config,layer_id,heads,contract,*,max_checkpoint_bytes=4*1024**2,max_runtime_bytes=4*1024**2):
    bounds=dict(max_checkpoint_bytes=max_checkpoint_bytes,max_runtime_bytes=max_runtime_bytes);m=rank_qb_manifest(reader,config,layer_id,heads,contract,**bounds)
    weight,scale=reader.linear_payload(m.prefix,output_slice=slice(m.first_row,m.stop_row),scale_dtype=torch.bfloat16)
    if not isinstance(weight,torch.Tensor) or not isinstance(scale,torch.Tensor) or weight.device.type!='cpu' or scale.device.type!='cpu' or weight.dtype!=torch.int8 or weight.shape!=(m.stop_row-m.first_row,1536) or scale.dtype!=torch.bfloat16 or scale.shape!=(m.stop_row-m.first_row,) or not bool(torch.isfinite(scale).all()) or not bool((scale>0).all()):raise ValueError('rank q_b returned payload differs')
    if rank_qb_manifest(reader,config,layer_id,heads,contract,**bounds)!=m:raise ValueError('q_b metadata changed during payload load')
    weight=weight.contiguous().clone();scale=scale.contiguous().clone()
    return PreparedRankQB(m,weight,scale,_digest(weight),_digest(scale))


def inspect_qb_sources(sources):
    """Conservative AST source-role ledger,never imports code or attests ABI."""
    def need(ok,msg):
        if not ok:raise ValueError(msg)
    def cls(name,file):
        found=[n for n in ast.parse(sources[file]).body if isinstance(n,ast.ClassDef) and n.name==name];need(len(found)==1,'source class drift');return found[0]
    def method(c,name):
        found=[n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name];need(len(found)==1,'source method drift');return found[0]
    def calls(node,name):return [n for n in ast.walk(node) if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
    def assignments(node,target):return [n.value for n in ast.walk(node) if isinstance(n,ast.Assign) and any(ast.unparse(v)==target for v in n.targets)]
    glm=method(cls('Glm5NextMLAAttention','glm_attention.py'),'__init__');qb=assignments(glm,'self.q_b_proj');need(len(qb)==1 and isinstance(qb[0],ast.Call) and ast.unparse(qb[0].func)=='ColumnParallelLinear' and [ast.unparse(v) for v in qb[0].args]==['self.q_lora_rank','self.num_heads * self.qk_head_dim'],'GLM q_b source geometry drift')
    kw={v.arg:ast.unparse(v.value) for v in qb[0].keywords};need(kw.get('bias')=='False' and 'gather_output' not in kw and 'disable_tp' not in kw,'GLM q_b partition/bias source drift')
    need([ast.unparse(v) for v in assignments(glm,'self.num_local_heads')]==['num_heads // tp_size'],'GLM local-head source drift')
    wrapper=calls(glm,'MultiHeadLatentAttentionWrapper');need(len(wrapper)==1 and ast.unparse(wrapper[0].args[1])=='self.num_local_heads' and {v.arg:ast.unparse(v.value) for v in wrapper[0].keywords}.get('fuse_qkv_rmsnorm')=='True','GLM wrapper/local heads/norm source drift')
    fused=method(cls('DeepSeekV2FusedQkvAProjLinear','deepseek_v2.py'),'__init__');need(any({v.arg:ast.unparse(v.value) for v in n.keywords}.get('disable_tp')=='True' for n in calls(fused,'super().__init__')),'replicated fused q_a/kv_a source drift')
    column=cls('ColumnParallelLinear','linear.py');init=method(column,'__init__');defaults={v.arg:ast.unparse(d) for v,d in zip(init.args.args[-len(init.args.defaults):],init.args.defaults)};need(defaults.get('gather_output')=='False','column default gather source drift')
    need([ast.unparse(v) for v in assignments(init,'self.input_size_per_partition')]==['input_size'] and [ast.unparse(v) for v in assignments(init,'self.output_size_per_partition')]==['divide(output_size, self.tp_size)'],'column input/output partition drift')
    narrow=calls(method(column,'weight_loader'),'loaded_weight.narrow');need(len(narrow)==1 and [ast.unparse(v) for v in narrow[0].args]==['output_dim','start_idx','shard_size'],'column output-row loader drift')
    adapter=method(cls('AscendLinearMethod','method_adapters.py'),'create_weights');attrs=calls(adapter,'set_weight_attrs');dicts=[ast.literal_eval(n.args[1]) for n in attrs if len(n.args)>1 and isinstance(n.args[1],ast.Dict) and all(isinstance(v,ast.Constant) for v in n.args[1].values)];need({'input_dim':1,'output_dim':0} in dicts and {'output_dim':0} in dicts,'Ascend weight/channel output_dim metadata drift')
    quant=cls('AscendW8A8DynamicLinearMethod','w8a8_dynamic.py');empty=calls(method(quant,'get_weight'),'torch.empty');need(len(empty)==1 and [ast.unparse(v) for v in empty[0].args]==['output_size','input_size'],'W8A8 NK shape drift')
    forward=method(cls('MultiHeadLatentAttentionWrapper','mla.py'),'forward');need(bool(calls(forward,'self.q_a_layernorm')) and bool(calls(forward,'fused_q_kv_rmsnorm')) and bool(calls(forward,'q_proj_layer')),'normalized qr/q_b wrapper roles drift')
    return {'source_only':True,'native_ABI_validated':False,'q_b':'ColumnParallel NK output rows;full qr input;default gather false;bias false','fused_q_a_kv_a':'disable_tp=True;replicated rank output','normalization':'wrapper has unfused and fused q/kv RMSNorm before q_b;GLM requests fused branch','heads':'wrapper receives num_heads//tp_size','parameter_metadata':'Ascend weight input_dim1/output_dim0;per-output-channel output_dim0','active_runtime_or_overlap_branch':'unverified;PreparedLinearInput/native quant numerics not attested'}
