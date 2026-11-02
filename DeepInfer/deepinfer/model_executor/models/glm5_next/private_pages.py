# SPDX-License-Identifier: Apache-2.0
"""CPU-only private DSA page plan/staging; no allocation from a live allocator.

Reservations/occupied sets are caller declarations, not verified leases. Compact
staged pages are NOT resident at their declared global IDs. Never hand the read
table to a native operator until an independently owned allocator has installed
all bytes and completed its producers. Nothing here publishes tables/state,
claims resources, writes a serving cache or promises async/crash rollback.
"""
import ast
from dataclasses import dataclass
import torch
from .cann_cache import view_main_index,view_tail


def _integer(v,name,minimum=0):
    if type(v) is not int or v<minimum:raise ValueError(f'invalid {name}')


@dataclass(frozen=True)
class PrivateDSAPlan:
    start: int
    tokens: int
    block_size: int
    pool_size: int
    num_main_pages: int
    num_tail_pages: int
    committed_table: tuple[int,...]
    read_table: tuple[int,...]
    changed_logical_pages: tuple[int,...]
    private_ids: tuple[int,...]
    copy_sources: tuple[int | None,...]
    tail_source: int | None
    tail_private: int | None

    def __post_init__(self):
        for name in ('start','tokens','num_main_pages','num_tail_pages'):_integer(getattr(self,name),name)
        _integer(self.block_size,'block_size',1);_integer(self.pool_size,'pool_size',2)
        if self.block_size%128 or self.block_size>4096 or 128%self.pool_size:raise ValueError('invalid bounded plan kernel geometry')
        fields=(self.committed_table,self.read_table,self.changed_logical_pages,self.private_ids,self.copy_sources)
        if any(not isinstance(x,tuple) for x in fields):raise ValueError('immutable plan fields required')
        if len(self.committed_table)>4096:raise ValueError('plan metadata table bound exceeded')
        expected=tuple(range(self.start//self.block_size,(self.end-1)//self.block_size+1)) if self.tokens else ()
        if self.changed_logical_pages!=expected or len(self.private_ids)!=len(expected) or len(self.copy_sources)!=len(expected):raise ValueError('plan touched-page structure differs')
        if len(self.committed_table)!=(self.start+self.block_size-1)//self.block_size or len(self.read_table)!=(self.end+self.block_size-1)//self.block_size:raise ValueError('plan table lengths differ')
        for ids in (self.committed_table,self.read_table,self.private_ids):
            if any(type(i) is not int or not 0<=i<self.num_main_pages for i in ids) or len(set(ids))!=len(ids):raise ValueError('invalid plan primary IDs')
        if set(self.private_ids)&set(self.committed_table):raise ValueError('private pages alias committed prefix')
        table=list(self.committed_table)
        for logical,dest,source in zip(expected,self.private_ids,self.copy_sources,strict=True):
            original=self.committed_table[logical] if logical<len(self.committed_table) else None
            if source!=original:raise ValueError('plan copy source differs')
            if logical==len(table):table.append(dest)
            else:table[logical]=dest
        if tuple(table)!=self.read_table:raise ValueError('plan candidate table differs')
        for page in (self.tail_source,self.tail_private):
            if page is not None and (type(page) is not int or not 0<=page<self.num_tail_pages):raise ValueError('invalid plan tail ID')
        if self.start%self.pool_size and self.tail_source is None:raise ValueError('plan lacks incomplete-pool history')
        if self.tokens and (self.tail_private is None or self.tail_private==self.tail_source):raise ValueError('plan tail must be private')
        if not self.tokens and self.tail_private is not None:raise ValueError('empty plan reserves tail')
        if self.read_table and max(self.kernel_read_table)>2147483647:raise ValueError('kernel table exceeds INT32')

    @property
    def end(self):return self.start+self.tokens
    @property
    def new_pools(self):return self.end//self.pool_size-self.start//self.pool_size
    @property
    def kernel_read_table(self):
        factor=self.block_size//128
        return tuple(page*factor+i for page in self.read_table for i in range(factor))


def plan_private_dsa_pages(start,tokens,committed_table,private_ids,*,geometry,num_main_pages,num_tail_pages,
                           occupied_main,occupied_tail,tail_source=None,tail_private=None,max_tokens=128):
    """Replace every append-touched main/index page as one coupled page pair.

    Prefix pages remain read-only. Caller declares ALL unavailable IDs, not just
    this request's prefix; tail IDs use an independent namespace. Empty work must
    carry no new reservation. This validates geometry/declarations only.
    """
    for v,n in ((start,'start'),(tokens,'tokens'),(num_main_pages,'main pages'),(num_tail_pages,'tail pages')):_integer(v,n)
    _integer(max_tokens,'max_tokens',1)
    if tokens>max_tokens:raise ValueError('append exceeds token budget')
    b=geometry.layout.logical_block_size;p=geometry.layout.pool_size
    if b%128 or b>4096 or 128%p:raise ValueError('bounded kernel128/pool-compatible pages required')
    if not isinstance(committed_table,tuple) or not isinstance(private_ids,tuple) or not isinstance(occupied_main,frozenset) or not isinstance(occupied_tail,frozenset):
        raise ValueError('immutable explicit tables/reservation/occupied sets required')
    for ids,limit in ((committed_table,num_main_pages),(private_ids,num_main_pages),(occupied_main,num_main_pages),(occupied_tail,num_tail_pages)):
        if any(type(i) is not int or not 0<=i<limit for i in ids) or len(set(ids))!=len(ids):raise ValueError('invalid/duplicate page IDs')
    if len(committed_table)!=(start+b-1)//b or not set(committed_table)<=occupied_main:raise ValueError('committed prefix must match visible length and occupied set')
    if tail_source is not None and (type(tail_source) is not int or tail_source not in occupied_tail):raise ValueError('tail history must be an occupied source page')
    if start%p and tail_source is None:raise ValueError('incomplete pool requires owned tail history')
    changed=tuple(range(start//b,(start+tokens-1)//b+1)) if tokens else ()
    if len(private_ids)!=len(changed) or set(private_ids)&occupied_main:raise ValueError('one disjoint private page pair per touched logical page required')
    if tokens:
        if type(tail_private) is not int or not 0<=tail_private<num_tail_pages or tail_private in occupied_tail:raise ValueError('disjoint private tail page required')
    elif tail_private is not None:raise ValueError('empty work must reserve no tail page')
    table=list(committed_table);sources=[]
    for logical,dest in zip(changed,private_ids,strict=True):
        sources.append(committed_table[logical] if logical<len(committed_table) else None)
        if logical==len(table):table.append(dest)
        else:table[logical]=dest
    return PrivateDSAPlan(start,tokens,b,p,num_main_pages,num_tail_pages,committed_table,tuple(table),changed,private_ids,tuple(sources),tail_source,tail_private)


@dataclass(frozen=True)
class PrivateDSAStage:
    plan: PrivateDSAPlan
    main: torch.Tensor  # compact UINT8 [private pairs,main_page_bytes]
    index: torch.Tensor  # compact UINT8 [private pairs,index_page_bytes]
    tail: torch.Tensor  # compact UINT8 [0 or1,tail_page_bytes]


def _cpu(*tensors):
    if any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' for t in tensors):raise ValueError('private page staging is CPU-only')


@torch.inference_mode()
def prepare_private_dsa_cpu(main_raw,index_raw,tail_raw,plan,geometry,latent_append,compressed_append,tail_after,*,max_private_bytes=16*1024**2):
    """Stage completed BF16 main/index producers and complete FP32 tail payload.

    Raw committed backings and plan tables remain unchanged, even on failure.
    Tail page padding is copied from its source or zeroed for a fresh request.
    Payload arithmetic/compression and history-position validation are caller
    responsibilities. This does not install bytes at global private page IDs.
    """
    _cpu(main_raw,index_raw,tail_raw,latent_append,compressed_append)
    if tail_after is not None:_cpu(tail_after)
    _integer(max_private_bytes,'max_private_bytes',1)
    if (plan.block_size,plan.pool_size)!=(geometry.layout.logical_block_size,geometry.layout.pool_size):raise ValueError('plan/geometry differs')
    view_main_index(main_raw,index_raw,plan.num_main_pages,geometry);view_tail(tail_raw,plan.num_tail_pages,geometry)
    l=geometry.layout.latent_dim;d=geometry.layout.index_dim;n=len(plan.private_ids)
    if latent_append.shape!=(plan.tokens,l) or latent_append.dtype!=torch.bfloat16:raise ValueError('completed BF16 latent append required')
    if compressed_append.shape!=(plan.new_pools,d) or compressed_append.dtype!=torch.bfloat16:raise ValueError('completed BF16 compressed append required')
    if plan.tokens:
        if tail_after is None or tail_after.shape!=(geometry.layout.tail_capacity,2*d) or tail_after.dtype!=torch.float32:raise ValueError('complete FP32 concatenated tail payload required')
    elif tail_after is not None:raise ValueError('empty work must carry no tail payload')
    size=n*(geometry.main_page_bytes+geometry.index_page_bytes)+(geometry.tail_page_bytes if plan.tokens else 0)
    if size>max_private_bytes:raise ValueError('private page byte budget exceeded')
    if any(not bool(torch.isfinite(t).all()) for t in (latent_append,compressed_append,*(() if tail_after is None else (tail_after,)))):raise ValueError('nonfinite completed producer payload')
    main=torch.zeros(n,geometry.main_page_bytes,dtype=torch.uint8);index=torch.zeros(n,geometry.index_page_bytes,dtype=torch.uint8)
    tail=torch.zeros(int(bool(plan.tokens)),geometry.tail_page_bytes,dtype=torch.uint8)
    for row,source in enumerate(plan.copy_sources):
        if source is not None:
            main[row].copy_(main_raw.reshape(plan.num_main_pages,-1)[source]);index[row].copy_(index_raw.reshape(plan.num_main_pages,-1)[source])
    if plan.tokens and plan.tail_source is not None:tail[0].copy_(tail_raw.reshape(plan.num_tail_pages,-1)[plan.tail_source])
    mapping={logical:i for i,logical in enumerate(plan.changed_logical_pages)}
    main_view=main.view(torch.bfloat16).reshape(n,plan.block_size,l);index_view=index.view(torch.bfloat16).reshape(n,plan.block_size//plan.pool_size,d)
    for offset,pos in enumerate(range(plan.start,plan.end)):main_view[mapping[pos//plan.block_size],pos%plan.block_size]=latent_append[offset]
    for offset,pool in enumerate(range(plan.start//plan.pool_size,plan.end//plan.pool_size)):
        token=pool*plan.pool_size;index_view[mapping[token//plan.block_size],pool%(plan.block_size//plan.pool_size)]=compressed_append[offset]
    if plan.tokens:tail.view(torch.float32).reshape(1,geometry.tail_page_rows,2*d)[0,:geometry.layout.tail_capacity].copy_(tail_after)
    return PrivateDSAStage(plan,main,index,tail)


@torch.inference_mode()
def gather_private_latent_cpu(main_raw,stage,geometry,token_ids,*,max_selected=4096,max_output_bytes=8*1024**2):
    """Resolve selected logical tokens against read-only prefix + staged pages.

    The result is a CPU gather, not a paged native cache or allocator mapping.
    -1 padding returns zeros; caller separately checks per-query causality.
    """
    _cpu(main_raw,stage.main,token_ids);_integer(max_selected,'max_selected',1);_integer(max_output_bytes,'max_output_bytes',1)
    p=stage.plan;l=geometry.layout.latent_dim
    if (p.block_size,p.pool_size)!=(geometry.layout.logical_block_size,geometry.layout.pool_size):raise ValueError('read geometry differs')
    if main_raw.dtype!=torch.uint8 or main_raw.ndim!=1 or main_raw.storage_offset()!=0 or not main_raw.is_contiguous() or main_raw.numel()!=p.num_main_pages*geometry.main_page_bytes:raise ValueError('committed main backing differs')
    if token_ids.ndim!=1 or token_ids.dtype not in (torch.int32,torch.int64) or len(token_ids)>max_selected:raise ValueError('bounded logical token vector required')
    if len(token_ids)*l*2>max_output_bytes:raise ValueError('gather output byte budget exceeded')
    if bool(((token_ids < -1)|(token_ids>=p.end)).any()):raise ValueError('selected token outside candidate visible prefix')
    if stage.main.dtype!=torch.uint8 or stage.main.shape!=(len(p.private_ids),geometry.main_page_bytes):raise ValueError('compact staged main pages differ')
    private=stage.main.view(torch.bfloat16).reshape(len(p.private_ids),p.block_size,l);old=main_raw.view(torch.bfloat16).reshape(p.num_main_pages,p.block_size,l)
    changed={logical:i for i,logical in enumerate(p.changed_logical_pages)};out=torch.zeros(len(token_ids),l,dtype=torch.bfloat16)
    for row,token in enumerate(token_ids.tolist()):
        if token<0:continue
        logical,offset=divmod(token,p.block_size)
        out[row]=private[changed[logical],offset] if logical in changed else old[p.committed_table[logical],offset]
    return out



@torch.inference_mode()
def gather_private_index_cpu(index_raw,stage,geometry,pool_ids,*,max_selected=4096,max_output_bytes=8*1024**2):
    """Copy logical completed-pool rows from shared prefix or private index pages.

    Pool IDs use the compressed namespace (one row per pool), not raw tokens.
    Candidate-wide completeness is checked here; per-query causal visibility is
    checked by the explicit selection reference. -1 padding returns zeros.
    """
    _cpu(index_raw,stage.index,pool_ids);_integer(max_selected,'max_selected',1);_integer(max_output_bytes,'max_output_bytes',1)
    p=stage.plan;d=geometry.layout.index_dim
    if (p.block_size,p.pool_size)!=(geometry.layout.logical_block_size,geometry.layout.pool_size):raise ValueError('index read geometry differs')
    if index_raw.dtype!=torch.uint8 or index_raw.ndim!=1 or index_raw.storage_offset()!=0 or not index_raw.is_contiguous() or index_raw.numel()!=p.num_main_pages*geometry.index_page_bytes:
        raise ValueError('committed index backing differs')
    if pool_ids.ndim!=1 or pool_ids.dtype not in (torch.int32,torch.int64) or len(pool_ids)>max_selected:raise ValueError('bounded logical pool vector required')
    if len(pool_ids)*d*2>max_output_bytes:raise ValueError('index gather output byte budget exceeded')
    if bool(((pool_ids < -1)|(pool_ids>=p.end//p.pool_size)).any()):raise ValueError('pool outside candidate completed prefix')
    if stage.index.dtype!=torch.uint8 or stage.index.shape!=(len(p.private_ids),geometry.index_page_bytes) or not stage.index.is_contiguous() or stage.index.storage_offset()!=0:
        raise ValueError('compact staged index pages differ')
    rows=p.block_size//p.pool_size;private=stage.index.view(torch.bfloat16).reshape(len(p.private_ids),rows,d)
    old=index_raw.view(torch.bfloat16).reshape(p.num_main_pages,rows,d);changed={logical:i for i,logical in enumerate(p.changed_logical_pages)}
    out=torch.zeros(len(pool_ids),d,dtype=torch.bfloat16)
    for row,pool in enumerate(pool_ids.tolist()):
        if pool<0:continue
        logical,offset=divmod(pool,rows)
        out[row]=private[changed[logical],offset] if logical in changed else old[p.committed_table[logical],offset]
    return out


def inspect_ascend_wrapper_source(source,utils_source):
    """AST-only registration/context ledger; no import or ABI validation."""
    tree=ast.parse(source);utils=ast.parse(utils_source)
    def function(name,parent):
        found=[n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(found)!=1:raise ValueError('missing source function '+name)
        return found[0]
    def cls(name):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    def calls(root,name):return [n for n in ast.walk(root) if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
    mapped=any(isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='MultiHeadLatentAttentionWrapper' and ast.unparse(v)=='AscendMultiHeadLatentAttention' for k,v in zip(n.keys,n.values)) for n in ast.walk(utils))
    if not mapped:raise ValueError('OOT wrapper registration mapping differs')
    registrations=calls(utils,'CustomOp.register_oot')
    if len(registrations)!=1 or {k.arg:ast.unparse(k.value) for k in registrations[0].keywords}!={'_decorated_op_cls':'op_cls','name':'name'}:
        raise ValueError('OOT registration application differs')
    constructor=function('__init__',cls('AscendMultiHeadLatentAttention'));inners=calls(constructor,'AscendMLAAttention')
    if len(inners)!=1:raise ValueError('inner attention construction differs')
    wiring={k.arg:ast.unparse(k.value) for k in inners[0].keywords}
    for key,value in {'indexer':'ascend_indexer','fused_qkv_a_proj':'mla_modules.fused_qkv_a_proj',
                      'q_a_layernorm':'mla_modules.q_a_layernorm','kv_a_layernorm':'mla_modules.kv_a_layernorm',
                      'kv_b_proj':'mla_modules.kv_b_proj','o_proj':'mla_modules.o_proj','use_sparse':'mla_modules.is_sparse'}.items():
        if wiring.get(key)!=value:raise ValueError('inner attention module role differs:'+key)

    outer=function('forward',cls('AscendMultiHeadLatentAttention'));dispatch=calls(outer,'torch.ops.vllm.mla_forward')
    if len(dispatch)!=1 or [ast.unparse(a) for a in dispatch[0].args]!=['hidden_states','output','self.prefix'] or dispatch[0].keywords:raise ValueError('raw-hidden wrapper dispatch differs')
    if any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in ('kv_cache','attn_metadata') for n in ast.walk(outer)):
        raise ValueError('outer optional cache/metadata are no longer ignored')
    inner=function('mla_forward',tree);assignments={ast.unparse(n.targets[0]):ast.unparse(n.value) for n in ast.walk(inner) if isinstance(n,ast.Assign) and len(n.targets)==1}
    if assignments.get('self')!='forward_context.no_compile_layers[layer_name]' or assignments.get('kv_cache')!='self.mla_attn.kv_cache':raise ValueError('context/cache owner binding differs')
    plain=[n for n in calls(inner,'self.mla_attn.impl.forward') if not n.keywords]
    if len(plain)!=1 or [ast.unparse(a) for a in plain[0].args]!=['self.mla_attn.layer_name','hidden_states','kv_cache','attn_metadata','output']:raise ValueError('backend hidden/cache handoff differs')
    wrapper=function('forward',cls('IndexerWrapper'));index=calls(wrapper,'self.impl')
    if len(index)!=1 or [ast.unparse(a) for a in index[0].args]!=['hidden_states','q_c','k_hidden_states','indexer_metadata','compute_topk']:raise ValueError('indexer role handoff differs')
    registration=[n for n in calls(tree,'direct_register_custom_op') if any(k.arg=='op_name' and isinstance(k.value,ast.Constant) and k.value.value=='mla_forward' for k in n.keywords)]
    if len(registration)!=1:raise ValueError('custom op registration differs')
    kw={k.arg:ast.unparse(k.value) for k in registration[0].keywords}
    if kw.get('mutates_args')!="['output']" or kw.get('dispatch_key')!="'PrivateUse1'":raise ValueError('custom op mutation/dispatch declaration differs')
    return {'source_only':True,'native_ABI_validated':False,'raw_hidden_dispatch':True,'optional_cache_metadata_ignored':True,
            'cache_owner':'forward_context.no_compile_layers[layer_name].mla_attn.kv_cache','declared_mutates_args':['output'],
            'implicit_cache_writes':'backend may mutate cache;output-only schema is not a transaction guarantee','native_binding_implemented':False}
