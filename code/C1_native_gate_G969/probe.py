"""Native C1 publish/consume gate. Execute only inside the fixed remote image.

Queries are prequantized FP4 operands with a power-of-two frequency multiplier;
this tests the indexer contract, not the model's projection or complete quality.
"""
from pathlib import Path
from types import FunctionType, MethodType, SimpleNamespace
import datetime, hashlib, json, os, sys, traceback
import torch, torch_npu
import freqs_install
import npu_c1_common as candidate
from sglang.srt.layers.attention.dsv4.v41_indexer import scoring, dense_blocks
from sglang.srt.layers.attention.dsv4.v41_indexer.types import DecodeInputs
from sglang.srt.runtime_context import get_platform
from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4

ROOT = Path(__file__).resolve().parent
CPU = '--cpu-contract' in sys.argv
CONTRACT_ROOT = Path(os.environ.get('C1_CONTRACT_ROOT', str(ROOT)))
rank = int(os.environ.get('LOCAL_RANK', '0'))
record = dict(rank=rank, pid=os.getpid(), phase='initializing', cases=[], graphs=[],
              scope='Native C1 component/changed-input graph only; not full-model quality or E2E gain')
OUT = CONTRACT_ROOT if CPU else ROOT / 'evidence'
OUT.mkdir(exist_ok=True)

def save():
    p = OUT / ('CPU_contract.json' if CPU else f'rank{rank}.json')
    t = p.with_suffix('.tmp'); t.write_text(json.dumps(record, indent=2)+'\n'); t.replace(p)

def clone(fn, replacements=None):
    env = dict(fn.__globals__)
    if replacements: env.update(replacements)
    copied = FunctionType(fn.__code__, env, fn.__name__, fn.__defaults__, fn.__closure__)
    copied.__kwdefaults__ = fn.__kwdefaults__
    return copied

def sync():
    torch.npu.synchronize()

class FixtureIndexer:
    index_topk = 512
    n_local_heads = 32
    def queries(self, q, freqs):
        # Keep inputs in the FP4-domain while making frequency changes observable.
        return q * freqs.real[:, None, :1].to(q.dtype)
    def head_weights(self, x):
        return x

class FixturePool:
    def __init__(self, table): self.table = table
    def get_index_k_with_scale_buffer(self, layer_id): return self.table

try:
    save()
    if CPU: assert not torch.npu.is_initialized()
    else:
        os.sched_setaffinity(0, {144+rank})
        torch.set_num_threads(1); torch.set_grad_enabled(False); torch.npu.set_device(rank)
    # Apply the same actual eight-site frequency installer used by the model.
    freqs_install.install(CONTRACT_ROOT)
    # Freeze functions AND their consumer globals before identity-preserving mutation.
    old_write = clone(scoring.write_decode)
    old_select = clone(scoring.select_decode, {'write_decode': old_write})
    old_decode = clone(scoring.decode_scores)
    replacements = dict(decode_scores=old_decode, write_decode=old_write, select_decode=old_select)
    old_publish = clone(dense_blocks.DenseBlocksBackend.publish_decode, replacements)
    old_consume = clone(dense_blocks.DenseBlocksBackend.consume_decode, replacements)
    identity = (scoring.decode_scores, scoring.write_decode)
    old_codes = (old_decode.__code__, old_write.__code__)
    candidate.install(CONTRACT_ROOT)
    assert identity == (scoring.decode_scores, scoring.write_decode)
    assert dense_blocks.decode_scores is scoring.decode_scores
    assert dense_blocks.write_decode is scoring.write_decode
    assert old_codes == (old_decode.__code__, old_write.__code__)
    assert old_decode.__code__ is not scoring.decode_scores.__code__
    assert old_write.__code__ is not scoring.write_decode.__code__
    assert old_publish.__globals__['decode_scores'] is old_decode
    assert old_publish.__globals__['select_decode'].__globals__['write_decode'] is old_write
    assert old_consume.__globals__['write_decode'] is old_write
    lazy_type = scoring.__dict__['_c1_scores_type']
    tensor = torch.zeros(1)
    lazy = lazy_type(bs=1,lmax=1,lens=tensor,slots=None,scores=tensor,
                     req=tensor,req_to_token=tensor,ratio=1)
    assert isinstance(lazy,scoring.DecodeScores) and not isinstance(lazy,scoring.PagedDecodeScores)
    assert lazy.slots is None
    record.update(native_alias_identity=True, original_reference_independently_frozen=True,
                  actual_frozen_subclass_slots_none=True,
                  candidate_SHA256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
                  architecture=os.uname().machine,torch=torch.__version__,torch_npu=torch_npu.__version__)
    if CPU:
        assert not torch.npu.is_initialized()
        record.update(phase='CPU_contract_passed',NPU_initialized=False,
                      at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(); print(json.dumps(record)); sys.exit(0)
    # Use actual fixed-image platform probes, with no override or fake SM facts.
    assert get_platform().is_npu and not get_platform().is_sm90
    record['platform_actual']={'is_npu':True,'is_sm90':False,'overrides':False}
    config=json.loads((ROOT/'model_config.json').read_text())['text_config']
    assert (config['index_topk'],config['candidate_topk_blocks'],config['candidate_block_size'])==(512,2048,8)
    record['actual_model_candidate_config']={k:config[k] for k in ['index_topk','candidate_topk_blocks','candidate_block_size']}
    device=f'npu:{rank}'

    def run_native(inputs, backend):
        d=old_decode(inputs=inputs, token_to_kv_pool=backend.token_to_kv_pool, req_to_token=backend.req_to_token)
        if d is None: return None
        bits=d.scores.view(torch.int32).clone()
        blocks=MethodType(old_publish,backend)(inputs)
        publish_slots=inputs.out_page_indices.clone()
        MethodType(old_consume,backend)(inputs,blocks)
        return bits,blocks.blocks.clone(),publish_slots,inputs.out_page_indices.clone()

    def run_candidate(inputs,backend):
        d=scoring.decode_scores(inputs=inputs, token_to_kv_pool=backend.token_to_kv_pool, req_to_token=backend.req_to_token)
        if d is None: return None
        assert d.slots is None and type(d) is lazy_type
        blocks=backend.publish_decode(inputs)
        publish_slots=inputs.out_page_indices.clone()
        backend.consume_decode(inputs,blocks)
        return d.scores.view(torch.int32),blocks.blocks,publish_slots,inputs.out_page_indices.clone()

    def compare(expected,actual):
        assert (expected is None)==(actual is None)
        if expected is not None:
            for label,a,b in zip(['FP32_score_bits','native_candidate_block_ids','publish_physical_slots','consume_physical_slots'],expected,actual):
                assert torch.equal(a,b),label+' differs'

    for ratio in [1,2]:
        capacity=263435 if ratio==1 else 131717
        cases=[(0,65,0,2048),(1,0,0,2048),(1,1,0,2048),
               (6,65,65,2),(16,513,513,2048),(80,4097,4097,3),
               (24,capacity,2048,2048),(96,capacity,2048,2048),(1,capacity,capacity,2048)]
        for number,(batch,L,visible,topk_blocks) in enumerate(cases):
            gen=torch.Generator().manual_seed(969000+100*rank+10*ratio+number)
            q=fake_quant_fp4((torch.randn(batch,32,128,generator=gen)*.125).bfloat16().to(device)) if batch else torch.empty((0,32,128),dtype=torch.bfloat16,device=device)
            weights=torch.rand(batch,32,generator=gen).bfloat16().to(device)
            rt=torch.randint(0,4096*ratio,(4,L*ratio),generator=gen,dtype=torch.int32).to(device)
            req=(torch.arange(batch,device=device)%4).to(torch.int64)
            table=torch.randint(0,256,(64,64*68),generator=gen,dtype=torch.uint8).to(device)
            table[:,64*64:]=127
            # Nonnegative physical positions; ratio2 position0 is zero visible.
            lens=(torch.arange(batch,device=device)*37)%max(visible+1,1)
            if batch: lens[-1]=visible
            pos=(lens*ratio-1).clamp_min(0).to(torch.int64)
            if ratio==1 and batch and visible==0: pos.fill_(0) # native minimum visibility=1
            freq_size=max(L*ratio+1,1)
            f=torch.zeros((freq_size,1,2),dtype=torch.float32,device=device);f[:,:,0]=1
            freqs=torch.view_as_complex(f)
            pages=torch.full((batch,512),-73,dtype=torch.int32,device=device)
            inputs=DecodeInputs(indexer=FixtureIndexer(),layer_id=20,compress_ratio=ratio,
                freqs_cis=freqs,x=weights,q_lora=q,positions=pos,req_rows=req,
                paged_metadata=SimpleNamespace(max_compressed_seq_len=L),is_verify=True,out_page_indices=pages)
            backend=dense_blocks.DenseBlocksBackend(token_to_kv_pool=FixturePool(table),req_to_token=rt,
                candidate_topk_blocks=topk_blocks,candidate_block_size=8,use_deep_gemm_prefill=False)
            copies=[t.clone() for t in [q,weights,rt,req,table,pos,f]]
            record.update(phase='case_running',current_case=dict(batch=batch,capacity=L,ratio=ratio,blocks=topk_blocks),step='native')
            save(); expected=run_native(inputs,backend);sync()
            record['step']='candidate';save();actual=run_candidate(inputs,backend);sync();compare(expected,actual)
            for t,c in zip([q,weights,rt,req,table,pos,f],copies):assert torch.equal(t,c),'input mutated'
            if expected is None: assert torch.equal(pages,torch.full_like(pages,-1))
            record['cases'].append(dict(rows=batch,capacity=L,ratio=ratio,topk_blocks=topk_blocks,
                scores_blocks_publish_consume_equal=True,inputs_unchanged=True,
                scope='Prequantized FP4 query fixture; original native C1 selection functions'))
            save()
            if L==capacity:
                for _ in range(2):warm=run_candidate(inputs,backend)
                sync();graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph): captured=run_candidate(inputs,backend)
                for replay,bound in enumerate([127,L if batch==1 else 2049,63]):
                    q.copy_(fake_quant_fp4((torch.randn(batch,32,128,generator=gen)*.125).bfloat16().to(device)))
                    weights.copy_(torch.rand(batch,32,generator=gen).bfloat16().to(device))
                    lengths=(torch.arange(batch,device=device)*53)%(bound+1);lengths[-1]=bound
                    pos.copy_((lengths*ratio-1).clamp_min(0))
                    rt.copy_(torch.randint(0,4096*ratio,rt.shape,generator=gen,dtype=torch.int32).to(device))
                    req.copy_(((torch.arange(batch,device=device)+replay+1)%4).to(torch.int64))
                    table[:,:64*64].bitwise_xor_(torch.tensor(17,dtype=torch.uint8,device=device))
                    f[:,:,0]=2 if replay%2==0 else 1
                    f[:,:,1]=replay+1
                    expected=run_native(inputs,backend);sync();graph.replay();sync();compare(expected,captured)
                    record['graphs'].append(dict(rows=batch,capacity=L,ratio=ratio,replay=replay,visible_bound=bound,
                        changed_q_weights_positions_frequencies_request_rows_table_and_K=True,
                        native_eager_scores_blocks_publish_consume_equal=True));save()
                del graph,captured,warm
            del inputs,backend,expected,actual,copies,q,weights,rt,req,table,lens,pos,f,freqs,pages
            sync()
    record.update(phase='passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except SystemExit:
    raise
except BaseException as exc:
    record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
