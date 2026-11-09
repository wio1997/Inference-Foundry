#!/usr/bin/env python3
"""Bind sampler branch contracts and price actual reply-rank early D2H."""
import ast
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import sys
import zlib

sys.dont_write_bytecode=True
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT=Path('audit/execution-search')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())

def early_patch(source):
    old=source.read_text();tree=ast.parse(old)
    cls,=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NPUModelRunner']
    fn,=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='sample_tokens']
    lines=old.splitlines(True)
    model,=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='model_runner_output' for t in n.targets)]
    async_,=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='async_output' for t in n.targets)]
    model_text=''.join(lines[model.lineno-1:model.end_lineno]);async_text=''.join(lines[async_.lineno-1:async_.end_lineno])
    def indent(s):return ''.join('    '+line for line in s.splitlines(True))
    early_model=model_text.replace('model_runner_output =','early_model_runner_output =',1)
    early_async=async_text.replace('async_output =','early_async_output =',1).replace('model_runner_output=model_runner_output','model_runner_output=early_model_runner_output',1)
    marker='        with record_function_or_nullcontext("draft_token"):\n            if self.speculative_config:\n'
    assert old.count(marker)==1
    guard='''        # Keep publication after proposer submission/host failures. Only the copy
        # moves; input-batch history publication below stays in its old position.
        early_model_runner_output = None
        early_async_output = None
        if (
            self.use_async_scheduling
            and self.speculative_config is not None
            and self.speculative_config.method == "dspark"
            and not use_pp_spec_decode
            and get_tp_group().rank_in_group == 0
            and self.vllm_config.kv_transfer_config is None
            and self.vllm_config.ec_transfer_config is None
            and not self.dynamic_eplb
            and not self.need_accepted_tokens
            and self.debugger is None
            and self.input_batch.sampling_metadata.all_greedy
            and not self.input_batch.sampling_metadata.output_token_ids
            and sampler_output.logprobs_tensors is None
            and not prompt_logprobs_dict
            and num_nans_device is None
        ):
'''+indent(early_model)+indent(early_async)+'\n'
    new=old.replace(model_text,'        model_runner_output = early_model_runner_output\n        if model_runner_output is None:\n'+indent(model_text),1)
    new=new.replace(async_text,'        async_output = early_async_output\n        if async_output is None:\n'+indent(async_text),1)
    new=new.replace(marker,guard+marker,1)
    after=ast.parse(new)
    def funcs(t):return {(c.name,f.name):ast.dump(f,include_attributes=False) for c in t.body if isinstance(c,ast.ClassDef) for f in c.body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
    a,b=funcs(tree),funcs(after);changed=[list(k) for k in a if a[k]!=b[k]]
    assert changed==[['NPUModelRunner','sample_tokens']],changed
    p=OUT/'sampler-early-output-copy.patch'
    p.write_text('STOPPED DESIGN RECORD. Do not apply: mutable pinned next-input staging has no generation ownership protocol.\n\n'+''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/vllm_ascend/worker/model_runner_v1.py',tofile='b/vllm_ascend/worker/model_runner_v1.py')))
    return dict(path=str(p),sha256=digest(p),base_sha256=digest(source),parse_valid=True,changed_methods=changed,deployed=False,native_accepted=False,status='STOPPED_DIRECT_RECIPE_WITHOUT_STAGING_GENERATION_OWNERSHIP',
      unchanged=['all sampled-ID/math/quant/RNG primitives','all DSpark calls and device compute','input_batch.set_async_sampled_token_ids placement','return/publication remains after proposer submission and host exceptions','non-reply TP ranks and all guarded-out branches retain original copy'],
      qualification='Stopped design record, not a usable implementation. Guards exclude known KV/EC/hybrid/EPLB/debug branches but do not protect mutable pinned CPU staging from later generations. Native H2D probe disproves automatic source snapshot. Larger early-output route requires generation ownership/read-complete fences before integration.')

def main():
    identity=read(OUT/'sampler-execution-identity.json');pins=identity['source_pins']
    assert len(pins)==28
    for r in pins:assert digest(r['local_path'])==r['runtime_sha256']
    by={r['runtime_path']:r for r in pins}
    source=Path(by['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py']['local_path'])
    patch=early_patch(source)
    native_path=OUT/'native-pinned-h2d-a2/evidence.json';native=read(native_path)
    assert native['exit_code']==0 and native['health_before']==native['health_after']=='200'
    assert native['matching_workers']==16
    assert native['result']['torch']=='2.10.0+cpu' and native['result']['torch_npu']=='2.10.0.post7'
    cases=native['result']['cases'];assert len(cases)==4
    for c in cases:
        assert c['pinned'] and c['gate_pending_before'] and c['gate_pending_after']
        if c['ownership']=='shared_staging':
            assert c['old_value_differences']==c['n'] and c['replacement_value_differences']==0
            assert c['got_bytes_sha256']==c['replacement_bytes_sha256']
        else:
            assert c['old_value_differences']==0 and c['got_bytes_sha256']==c['old_bytes_sha256']
    receipt=read(OUT/'sampler-output-handoff-receipt.json');assert digest(receipt['artifact'])==receipt['sha256']
    raw=json.loads(zlib.decompress(Path(receipt['artifact']).read_bytes()))
    windows=[];population=Counter();reply=[]
    for r in raw:
        for w in r['windows']:
            anchors={v[0]:v for v in w['anchors']}
            assert set(anchors)=={w['target_last_task'],w['draft_last_task'],w['next_target_task']}
            target=anchors[w['target_last_task']];draft=anchors[w['draft_last_task']]
            main=target[3];t=target[2];d=draft[2]
            late=[(p,q) for p,q in w['memcpy_record_pairs'] if p[3]!=main and w['CANN_API_by_connection'][str(p[4])][0][3]=='aclrtMemcpyAsyncWithCondition' and p[1]>=d]
            assert len(late)==1,(r['rank'],w['wave'],len(late))
            p,q=late[0]
            waits=[v for v in w['host_waits'] if v[2]>>32!=v[2]%2**32 and v[4]=='aclrtSynchronizeEvent' and abs(v[1]-q[2])<=250000]
            is_reply=r['rank']%8==0
            assert len(waits)==int(is_reply),(r['rank'],w['wave'],waits)
            population['reply_wait_bound' if is_reply else 'non_reply_no_wait']+=1
            record=dict(rank=r['rank'],wave=w['wave'],kind=w['kind'],ordinal=w['ordinal'],B=w['B'],target_M=w['target_M'],scheduled=w['scheduled'],main_stream=main,output_copy_stream=p[3],target_last_task=target[0],draft_last_task=draft[0],copy_task=p[0],record_task=q[0],target_ready_ns=t,draft_end_ns=d,copy_ns=p[1:3],record_ns=q[1:3],copy_API=w['CANN_API_by_connection'][str(p[4])][0],record_API=w['CANN_API_by_connection'][str(q[4])][0],is_reply_rank=is_reply)
            if is_reply:
                wait,=waits;paid=q[2]-p[1]
                # Retain CPU handoff, sampler readiness and original copy/event time.
                credit=max(0,min(wait[1]-wait[0],q[2]-max(wait[0],t+paid)))
                record.update(wait=wait,wait_end_minus_record_ns=wait[1]-q[2],paid_copy_and_record_us=paid/1000,
                    conditional_same_CPU_handoff_credit_ms=credit/1e6,
                    CPU_device_offset_tolerance_ms={str(ms):[max(0,min(wait[1]-wait[0],q[2]-max(wait[0]+ms*1e6,t+paid)))/1e6,max(0,min(wait[1]-wait[0],q[2]-max(wait[0]-ms*1e6,t+paid)))/1e6] for ms in [.25,1,2,5]})
                reply.append(record)
            windows.append(record)
    assert population=={'reply_wait_bound':9,'non_reply_no_wait':63}
    hist_path=BASE/'full-review/PREFILL-OA-JOINT-ARRIVAL.json';hist=read(hist_path)
    m_by_ordinal={i:s['M'] for i,s in enumerate(hist['sections'])}
    requests=[]
    for r in hist['requests']:
        w,=[w for w in reply if w['kind']=='E' and m_by_ordinal[w['ordinal']]==r['first_eager_M'] and w['rank']==r['DP']*8 and str(r['slot']) in w['scheduled']]
        saved=w['conditional_same_CPU_handoff_credit_ms']
        requests.append(dict(slot=r['slot'],DP=r['DP'],wave=w['wave'],original_TTFT_ms=r['TTFT_ms'],conditional_early_copy_ms=saved,conditional_TTFT_percent=100*saved/r['TTFT_ms'],
            added_nonoverlapped_cost_sensitivity={str(us):dict(saved_ms=max(0,saved-us/1000),TTFT_percent=100*max(0,saved-us/1000)/r['TTFT_ms']) for us in [100,1000]},
            offset_tolerance_ms=w['CPU_device_offset_tolerance_ms']))
    base=max(r['original_TTFT_ms'] for r in requests);gain=base-max(r['original_TTFT_ms']-r['conditional_early_copy_ms'] for r in requests)
    price=dict(requests=requests,P90_TTFT_conditional_saved_ms=gain,P90_TTFT_conditional_percent=100*gain/base,max_individual_conditional_percent=max(r['conditional_TTFT_percent'] for r in requests),
       scope='First eager reply only per request. Original target/sample readiness, CPU output-thread handoff, copy/event duration, all draft compute/submission and downstream work retained. No cross-wave cumulative or throughput/response-end credit.',
       qualification='CANN API/device timestamps and source-motif matching are approximate; event handles and copy byte counts are absent. Nine unique reply-thread waits end within35.025us of late record; offset sensitivity retained. First-ID delivery to first nonempty client chunk remains a propagation assumption until service admission. No realized speedup.')
    source_reviews={
      'sampler.py':['AscendSampler','AscendTopKTopPSampler','random_sample','_apply_top_k_top_p_pytorch'],
      'rejection_sampler.py':['AscendRejectionSampler','apply_sampling_constraints','rejection_sample','sample_recovered_tokens','rejection_greedy_sample_spec_len_1_pytorch','rejection_greedy_sample_pytorch','rejection_random_sample_pytorch','sample_recovered_tokens_pytorch','rejection_random_sample_block_verify_pytorch','generate_uniform_probs','RejectionSampler'],
      'reject_sample.py':['rejection_greedy_sample_triton','rejection_greedy_sample_spec_len_1_triton','rejection_random_sample_kernel','sample_recovered_tokens_kernel','rejection_random_sample_block_verify_kernel'],
      'penalty.py':['apply_all_penalties_kernel','apply_penalties_triton'],
      'bincount.py':['token_bin_counts_and_mask_kernel','get_token_bin_counts_and_mask_triton'],
      'builtin.py':['MinTokensLogitsProcessor','MinPLogitsProcessor','LogitBiasLogitsProcessor'],
      '__init__.py':['build_logitsprocs'],
      'thinking_budget_state.py':['ThinkingBudgetStateHolder'],
      'multiproc_executor.py':['WorkerProc.enqueue_output','WorkerProc._execute_worker_rpc','MultiprocExecutor._get_output_rank','MultiprocExecutor.sample_tokens'],
      'llm_base_proposer.py':['AscendSpecDecodeBaseProposer.prepare_next_token_ids_padded'],
      'utils.py':['copy_kv_cache_blocks_inplace','AscendKVBlockZeroer.zero_block_ids','CpuGpuBuffer'],
      'gpu_model_runner.py':['GPUModelRunner._make_buffer','GPUModelRunner._prepare_input_ids']}
    # Symbol ranges are navigation, not a claim to have audited every line in each class.
    symbols=[]
    for r in pins:
        p=Path(r['local_path']);tree=ast.parse(p.read_text());wanted=source_reviews.get(p.name,[])
        for n in tree.body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
                if n.name in wanted:symbols.append(dict(runtime_path=r['runtime_path'],symbol=n.name,start=n.lineno,end=n.end_lineno))
                if isinstance(n,ast.ClassDef):
                    for f in n.body:
                        if isinstance(f,ast.FunctionDef) and n.name+'.'+f.name in wanted:symbols.append(dict(runtime_path=r['runtime_path'],symbol=n.name+'.'+f.name,start=f.lineno,end=f.end_lineno))
    branches=[
      dict(family='processor factory',guard='speculative_config present',behavior='Only MinTokens instantiated. Custom processors rejected; min_p/logit_bias warned inactive. No hidden MinP/bias work to erase in fixed DSpark.'),
      dict(family='greedy/random/mixed',guard='all_greedy/all_random/temperature',behavior='All-greedy returns before random filtering/noise; mixed saves greedy decisions before random-only processing, merges by original temperature predicate.'),
      dict(family='score observers',guard='raw/processed logits/logprobs; max_num_logprobs/specific IDs',behavior='Raw values precede mutation; FP32 raw-target clone currently unconditional in raw mode. Requested topk/selected scores need original logsoftmax, target token score and > rank, not only argmax. Ascend rejection return path gates requested scores on max_num_logprobs.'),
      dict(family='allowed/badword/penalty',guard='metadata predicates',behavior='Target penalties then allowed mask/badword/MinTokens; bonus path has its own allowed/badword/processor/penalty order. Histories expand committed output with preceding speculative prefixes; -1 output placeholders mapped to vocab padding before histogram.'),
      dict(family='penalty sharing',guard='not no_penalties',behavior='Prompt/output int32 atomic histograms + FP32 sign-conditioned repetition scaling, frequency then presence. Integer prefix counts/prompt-mask caching is a distinct representation candidate; none of this runs under no_penalties. Do not reassociate FP32 operations or interleave histogram updates without a join.'),
      dict(family='MinTokens + grammar',guard='min length and structured outputs',behavior='Mask stop tokens in each appropriate draft prefix; save pre-mask stop logits and restore finite stops only when entire row otherwise -inf. A simple union-of-masks kernel can lose this dependency.'),
      dict(family='thinking state',guard='holder enabled/tracked',behavior='Committed outputs, draft prefixes, start/end/countdown and force-index transitions; bonus call updates state then target call uses forcing. Newest accepted state and previous output length are not speculative -1 placeholders; do not swap the two calls or reuse masks across generations.'),
      dict(family='cutoff',guard='k/p absent, present or batch-invariant',behavior='Ascend native path returns exact alias if both absent; otherwise sorts FP32 probabilities and preserves threshold ties. TopP cutoff uses unfiltered probabilities even when TopK masks logits. Batch-invariant dispatches upstream sorted-logit semantics, not an interchangeable algorithm.'),
      dict(family='RNG',guard='partial/full/no seeded rows',behavior='Bonus draws global exponential plane unless fully seeded, then seeded row overrides on global stream and explicit join. Recovered draws global BxV exponential plane unconditionally plus seeded overrides; one q per request shared across positions. Uniform requests FP64 global plane then per-request seeded slice, zero-draft seeded slices skipped; Triton callers cast to FP32. Preserve shape/order/dtype/offsets.'),
      dict(family='ordinary verifier',guard='not block verify',behavior='K1 special only when every K=1 and is_greedy None. General greedy first mismatch + target token, all match bonus; random first rejection and recovered row; K0 bonus. Suffix remains -1 and no cross-request unmasked preceding-element load.'),
      dict(family='block/entropy',guard='enable_block_verify and maxK>=3; entropy flag independent',behavior='Block scans all K, sequential pi=min(pi*p/q,1) and cumulative uniforms, records last qualifying position; cannot stop at first failure. Entropy uses raw ori probabilities when present, FP32 log/prod/reduction order retained.'),
      dict(family='recovered vocab',guard='draft_probs None/full',behavior='None means onehot correction by zeroing proposed token; full uses max(p-q,0). Triton chunks4096, strict cross-chunk > retains earliest winner; q shared across K. Lazy recovery of only selected emitted row is possible only after original verify decision and unchanged complete RNG advancement; inactive random branch under fixed loads.'),
      dict(family='fallback/synthetic',guard='HAS_TRITON; rejection_sample_method',behavior='Pytorch fallback has vector masks, first/last prefix selection and explicit q==0/isinf handling; not bit-interchangeable with Triton. Blockwise recovery helper flag defaults false and ordinary call does not forward block flag. Synthetic+block raises ValueError; synthetic rates alter acceptance and cannot be used as a performance replacement.'),
    ]
    result=dict(status='PARTIAL_SAMPLER_SOURCE_BRANCH_REVIEW_AND_ACTUAL_REPLY_PRICE_LOOP014_ACTIVE',introduction='Complete score/ID observers -> actual sampler/RNG/verify guards -> reply-thread D2H -> next-input and retirement conditions',precision='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',source_identity='sampler-execution-identity.json',source_SHA_matches=len(pins),source_symbols=symbols,branch_families=branches,
      output_handoff=dict(receipt='sampler-output-handoff-receipt.json',native_windows=windows,population=dict(population),actual_reply_ranks=[0,8],all_late_copy_pairs=72,clock_match_max_abs_us=max(abs(r['wait_end_minus_record_ns'])/1000 for r in reply),price=price,patch=patch),
      candidates=[
       dict(scope='Early reply-rank D2H together with next-input staging generation ownership',status='待确认',depth='28-source identity, immutable sampled-ID consumer, actual72 markers/9 reply waits, paid-copy/CPU-handoff price, and native source-snapshot discriminator',value='Specific short-request first-token exposure; modest P90, no throughput claim',gaps=['enumerate all mutable pinned input/metadata staging last DMA readers, lease them until per-generation read-complete events, and charge ring/snapshot/backpressure','native graph/cache/slot reuse, EOS/abort/new-request and DP-empty retirement qualification','runtime guard eligibility and actual client first-nonempty propagation','matched service A/B including ordinary loads and DMA/host contention; no implementation admission yet']),
       dict(scope='Direct early-copy patch without next-input generation protocol',status='已停止',depth='native backend counterexample plus persistent CPU staging source',reason='Pinned INT32[24]/INT64[16] copy_ did not snapshot original bytes while event pending: all24/16 elements became overwritten values. Independent-generation controls retained all original bytes. This is a deterministic ownership discriminator, not evidence of an observed race in unmodified service. Default-stream order does not order CPU writes to pinned input buffers.'),
       dict(scope='Unconditional early output removing all completion semantics',status='已停止',depth='source retirement constraint',reason='KV consumer defer_block_free treats update_from_output as GPU completion; EC/hybrid/EPLB/debug state and raw readers can add required work. Keep old route or explicit independent completion fence; a data-ready event alone cannot retire such generations.'),
       dict(scope='Independent greedy clone/cast/argmax/reject fusion under original1% budget',status='已停止',depth='whole target head+verify free budget reused',reason='Prior complete parent already includes all head GEMM/transport/score/internal gaps: max C8 individual response0.58204%, six ordinary0.52940%. New output wait outside parent is priced separately, not a reason to restart small math variants.'),
       dict(scope='Random lazy recovery / shared penalty-prefix histograms under frozen greedy no-penalty calls',status='已停止',depth='guard-qualified inactive current load',reason='No active work to remove on specified calls. Source design boundaries retained for workload change; no new RNG/penalty profile or native probe justified now.'),
      ],
      retirement=dict(source='scheduler.py:1889-1893,2603-2645; NPU worker/utils.py:15-64,239-271; CpuGpuBuffer.copy_to_gpu',constraint='Async KV loads are not default-stream ordered; processed_step_seq assumes completed writes. No KV/EC guard preserves fallback. Ascend block zero and CoW are current-stream operations; CoW snapshots GPU source before writes. CPU _ids_pinned and CpuGpuBuffer.cpu are mutable, however, and their H2D readers can remain queued behind old draft work when CPU queues another generation. Default-stream order does not protect CPU overwrites.',not_accepted='Native backend test refutes source-snapshot assumption only, not a full-service race witness. Need per-buffer lease/read-complete or immutable generation snapshots; only then qualify EOS/abort/same-ID/DP-empty and all relevant secondary streams.'),
      native_staging_discriminator=dict(path=str(native_path),sha256=digest(native_path),result=native['result'],scope='Backend ownership parameter only; no replacement primitive/model performance acceptance. a1 setup error before NPU retained, a2 one isolated NPU process with current16-worker queue/env match and health200 before/after.'),
      prior_parent_price=dict(path='terminal-cost-replay.json',sha256=digest(OUT/'terminal-cost-replay.json')),
      profile_pins={str(hist_path):digest(hist_path),receipt['artifact']:receipt['sha256']},
      next='Trace CpuGpuBuffer and block-table/metadata pinned staging to actual last H2D readers; derive a generation ownership protocol for valuable early output before another native test. Then jointly price remaining metadata/host submission parent and revisit final-layer attention/FFN row-demand producer. Do not retry the stopped direct patch, greedy micro-plans or inactive random branches.',
      analyzer_sha256=digest(__file__),new_NPU_work=True,new_NPU_processes=1,new_service_inference_requests=0,new_profile_collection=False,offline_SQL_extractions=1,remote_write_scope=['/data/tiankuan/wio/.../runtime/fusion-pinned-h2d-generation-20261109-a1','/data/tiankuan/wio/.../runtime/fusion-pinned-h2d-generation-20261109-a2'],whole_chain_complete=False)
    (OUT/'sampler-output-frontier.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(source_SHA_matches=len(pins),windows=len(windows),population=dict(population),P90_TTFT_percent=price['P90_TTFT_conditional_percent'],max_individual_TTFT_percent=price['max_individual_conditional_percent'],patch=patch['path'],native_accepted=False,whole_chain_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
