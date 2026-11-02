# SPDX-License-Identifier: Apache-2.0
"""CPU-only CANN PKI sentinel/expanded-token packing and source-call contracts.

No native operation, Triton/vLLM import, binary ABI check or model integration.
Inputs/outputs are private bounded CPU references. Native scores/ties, dummy
execution safety, graph replay and operator output ABI remain unverified.
"""
import ast
import math
from dataclasses import asdict,dataclass

import torch


@dataclass(frozen=True)
class CANNIndexerContract:
    heads: int
    head_dim: int
    pool_size: int
    topk_tokens: int

    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.heads,self.head_dim,self.pool_size,self.topk_tokens)) or self.pool_size<2 or 128%self.pool_size or self.topk_tokens%self.pool_size:
            raise ValueError('positive indexer dimensions and kernel128-compatible pool/topk required')

    @classmethod
    def from_config(cls,config):
        from .cache import DSACacheLayout
        layout=DSACacheLayout.from_config(config,logical_block_size=128)
        return cls(config.indexer_config['index_n_heads'],layout.index_dim,layout.pool_size,layout.topk_tokens)

    @property
    def output_width(self):return self.topk_tokens+self.pool_size-1

    def to_dict(self):
        result=asdict(self);result.update(output_width=self.output_width,layout_q='TND',layout_k='PA_BBND',mask_mode=3,mode='CPU_reference_only',native_ABI_validated=False)
        return result


def _cpu(*tensors):
    if any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' for t in tensors):raise ValueError('CANN indexer preparation reference is CPU-only')


def _ends(ends,tokens):
    if ends.ndim!=1 or ends.dtype not in (torch.int32,torch.int64) or bool(((ends<0)|(ends>tokens)).any()) or bool((ends[1:]<ends[:-1]).any()):
        raise ValueError('query ends must be cumulative nondecreasing integers within token rows')
    return int(ends[-1]) if len(ends) else 0


@torch.inference_mode()
def prepare_nonempty_pki_cpu(query,weights,ends,lengths,tails,table,contract,*,num_kernel_blocks,
                             max_tokens=128,max_requests=8,max_table_entries=4096,max_output_bytes=2*1024*1024):
    """Match installed sentinel copies on a conservative valid metadata domain.

    lengths are COMPRESSED pool counts, tails are raw-token remainders. Table
    IDs address kernel128 cache blocks (32 pool rows for GLM pool4), already
    expanded by the caller when logical pages are larger. Even zero key lengths
    need one valid physical pool because CANN receives max(length,1).
    Table rows must equal real request count here; graph/CP variants need their
    own validation. Preserve padded query rows; packing later masks them using
    ORIGINAL ends/positions. No sentinel candidate is a real historical token.
    """
    _cpu(query,weights,ends,lengths,tails,table)
    if not isinstance(contract,CANNIndexerContract):raise ValueError('explicit CANN indexer contract required')
    if any(type(x) is not int or x<1 for x in (max_tokens,max_requests,max_table_entries,max_output_bytes,num_kernel_blocks)):raise ValueError('positive explicit preparation/cache bounds required')
    if query.ndim!=3 or query.shape[1:]!=(contract.heads,contract.head_dim) or query.dtype!=torch.bfloat16 or weights.shape!=query.shape[:2] or weights.dtype!=torch.bfloat16:
        raise ValueError('BF16 TND query and BF16 per-head weights required')
    tokens=query.shape[0];actual=_ends(ends,tokens);n=len(ends)
    if tokens>max_tokens or n>max_requests or lengths.shape!=(n,) or tails.shape!=(n,) or table.ndim!=2 or table.shape[0]!=n or table.shape[1]<1 or table.numel()>max_table_entries:
        raise ValueError('bounded one-table-row-per-request metadata required')
    if any(t.dtype not in (torch.int32,torch.int64) for t in (lengths,tails,table)) or bool((lengths<0).any()) or bool(((tails<0)|(tails>=contract.pool_size)).any()):raise ValueError('invalid integer compressed-length/tail/table metadata')
    if bool(((table < -1)|(table>=num_kernel_blocks)).any()):raise ValueError('kernel block ID out of range')
    for row,length in enumerate(lengths.tolist()):
        needed=(max(length,1)+(128//contract.pool_size)-1)//(128//contract.pool_size)
        if needed>table.shape[1] or bool((table[row,:needed]<0).any()):raise ValueError('even empty real histories require valid physical pool backing')
    shapes=((tokens+1,contract.heads,contract.head_dim),(tokens+1,contract.heads),(n+1,),(n+1,),(n+1,),(n+1,table.shape[1]))
    cost=sum(math.prod(shape)*t.element_size() for shape,t in zip(shapes,(query,weights,ends,lengths,tails,table)))
    if cost>max_output_bytes:raise ValueError('padded PKI output byte bound exceeded')
    if not bool(torch.isfinite(query[:actual]).all()) or not bool(torch.isfinite(weights[:actual]).all()):raise ValueError('actual query/weight rows must be finite')
    return (torch.cat((query,query.new_zeros((1,contract.heads,contract.head_dim)))).contiguous(),
            torch.cat((weights,weights.new_zeros((1,contract.heads)))).contiguous(),
            torch.cat((ends,ends.new_tensor([tokens+1]))).contiguous(),
            torch.cat((lengths.clamp_min(1),lengths.new_ones(1))).contiguous(),
            torch.cat((tails,tails.new_zeros(1))).contiguous(),
            torch.cat((table,table.new_zeros((1,table.shape[1])))).contiguous())


@torch.inference_mode()
def pack_cann_token_indices_cpu(indices,positions,ends,contract,*,index_namespace,max_tokens=128,max_output_bytes=2*1024*1024):
    """Pack already-expanded native TOKEN IDs, never pool IDs or physical slots.

    Match installed pack kernel with an explicit output buffer wide enough for
    topk+pool-1. Copy only causal completed-history candidates, append the raw
    tail, fill padding with-1. Original cumulative ends mask padded query rows.
    No raw native-output width/dtype or binary schema is inferred from this API.
    """
    _cpu(indices,positions,ends)
    if index_namespace!='expanded_token' or not isinstance(contract,CANNIndexerContract):raise ValueError('explicit expanded_token namespace and contract required')
    if any(type(x) is not int or x<1 for x in (max_tokens,max_output_bytes)):raise ValueError('positive explicit packing bounds required')
    if indices.ndim!=2 or indices.dtype!=torch.int32 or positions.shape!=(len(indices),) or positions.dtype not in (torch.int32,torch.int64):raise ValueError('INT32 candidate rows and integer positions required')
    n=len(indices);actual=_ends(ends,n)
    if n>max_tokens or n*contract.output_width*4>max_output_bytes:raise ValueError('packed index output bound exceeded')
    if bool(((positions[:actual]<0)|(positions[:actual]>=2**31-1)).any()):raise ValueError('actual query positions must fit causal INT32 token IDs')
    output=torch.full((n,1,contract.output_width),-1,dtype=torch.int32)
    p=contract.pool_size
    for row in range(actual):
        pos=int(positions[row]);tail_start=(pos+1)//p*p;history=min(tail_start,contract.topk_tokens)
        if indices.shape[1]<history:raise ValueError('native candidates do not cover completed history')
        selected=indices[row,:history]
        if bool(((selected<0)|(selected>=tail_start)).any()) or selected.unique().numel()!=history:raise ValueError('native historical token candidates must be distinct causal IDs')
        output[row,0,:history]=selected
        output[row,0,history:history+pos+1-tail_start]=torch.arange(tail_start,pos+1,dtype=torch.int32)
    return output


_KEY_ARGS=('hidden_states','self._key_weight','self._gate_weight','self.index_kpool_compress_ape','state','indexer_metadata.keypool_state_table','indexer_metadata.keypool_start_offsets')
_KEY_KW=('norm_weight','norm_bias','cu_seqlens','cmp_ratio','norm_eps')
_PKI_ARGS=('query','indexer_cache','weights.to(torch.bfloat16)','query_metadata.pool_tail')
_KEY_VALUES=('self._norm_weight_f32','self._norm_bias_f32','indexer_metadata.query_start_loc','self.index_kpool','norm_eps')
_PKI_VALUES=('query_metadata.cum_query_lens','query_metadata.seq_lens','indexer_metadata.block_table',"'TND'","'PA_BBND'",'self.topk_tokens','self.index_kpool','3')
_PKI_KW=('actual_seq_q','actual_seq_k','block_table','layout_q','layout_k','topk','pool_size','mask_mode')


def validate_indexer_source_calls(indexer_source,adapter_source):
    """AST-only call-site checks; do NOT import sources or claim binary ABI."""
    tree=ast.parse(indexer_source);methods=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_forward_cann']
    if len(methods)!=1:raise ValueError('one installed _forward_cann source method required')
    result={}
    for name,args,keywords in (('_key_pool',_KEY_ARGS,_KEY_KW),('_pool_key_indexer',_PKI_ARGS,_PKI_KW)):
        calls=[n for n in ast.walk(methods[0]) if isinstance(n,ast.Call) and ast.unparse(n.func)=='self.'+name]
        if len(calls)!=1 or tuple(ast.unparse(a) for a in calls[0].args)!=args or tuple(k.arg for k in calls[0].keywords)!=keywords:raise ValueError('installed CANN call-site contract changed: '+name)
        values=_KEY_VALUES if name=='_key_pool' else _PKI_VALUES
        if tuple(ast.unparse(k.value) for k in calls[0].keywords)!=values:raise ValueError('installed CANN keyword roles changed: '+name)
        if name=='_pool_key_indexer':
            literals={k.arg:k.value.value for k in calls[0].keywords if isinstance(k.value,ast.Constant)}
            if literals!={'layout_q':'TND','layout_k':'PA_BBND','mask_mode':3}:raise ValueError('unsupported native PKI layout/mask literals')
        result[name]={'positional':args,'keyword_names':keywords}
    initializers=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_init_cann_ops']
    if len(initializers)!=1:raise ValueError('one native operator initializer source required')
    aliases={ast.unparse(n.targets[0]):ast.unparse(n.value) for n in ast.walk(initializers[0]) if isinstance(n,ast.Assign) and len(n.targets)==1}
    if aliases.get('self._key_pool')!='torch.ops.cann_ops_transformer.key_pool' or aliases.get('self._pool_key_indexer')!='partial(adapters.pool_key_indexer_nonempty, torch.ops.cann_ops_transformer.pool_key_indexer)':
        raise ValueError('installed native operator alias bindings changed')
    adapter=ast.parse(adapter_source)
    calls=[n for n in ast.walk(adapter) if isinstance(n,ast.Call) and ast.unparse(n.func)=='prepare_nonempty_keypool_inputs']
    if len(calls)!=1 or tuple(ast.unparse(a) for a in calls[0].args)!=('query','weights','actual_seq_q','actual_seq_k','pool_tail_k','block_table'):raise ValueError('installed nonempty adapter source contract changed')
    return {'source_only':True,'native_ABI_validated':False,'calls':result,'required_symbols':('torch.ops.cann_ops_transformer.key_pool','torch.ops.cann_ops_transformer.pool_key_indexer')}
