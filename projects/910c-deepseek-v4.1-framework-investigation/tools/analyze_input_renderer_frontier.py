#!/usr/bin/env python3
"""Price actual frontend encoding FIFO from completed trace, no new profile."""
from hashlib import sha256
from pathlib import Path
import itertools
import json
import zlib

OUT=Path('audit/execution-search')


def main():
    p=OUT/'frontend-output-events.json.zlib';r=json.loads((OUT/'frontend-output-trace-receipt.json').read_text());assert sha256(p.read_bytes()).hexdigest()==r['sha256']
    d=json.loads(zlib.decompress(p.read_bytes()));ns=d['names'];rows=d['rows'];by={r[5]:r for r in rows}
    def ancestors(row):
        out=[]
        while row[6] in by:
            row=by[row[6]];out.append(ns[row[0]])
        return out
    calls=sorted([r for r in rows if '/renderers/base.py(582): _tokenize_prompt' in ns[r[0]]],key=lambda r:r[1])
    native=sorted([r for r in rows if 'encode_batch' in ns[r[0]]],key=lambda r:r[1])
    submits=sorted([r for r in rows if 'async_utils.py' in ns[r[0]] and '_async_wrapper' in ns[r[0]] and any('_tokenize_singleton_prompt_async' in n for n in ancestors(r))],key=lambda r:r[1])
    assert len(calls)==len(native)==len(submits)==8
    assert len({r[4] for r in calls})==1
    for c,n in zip(calls,native):
        assert c[1]<=n[1] and n[1]+n[2]<=c[1]+c[2]
        assert ns[c[0]] in ancestors(n)
    origin=submits[0][1]
    jobs=[]
    for i,(s,c,n) in enumerate(zip(submits,calls,native)):
        jobs.append(dict(ordinal=i,submit_ms=(s[1]-origin)/1000,start_ms=(c[1]-origin)/1000,end_ms=(c[1]+c[2]-origin)/1000,
          parent_ms=c[2]/1000,native_ms=n[2]/1000,retained_parent_other_ms=(c[2]-n[2])/1000,
          queue_wait_ms=(c[1]-s[1])/1000,thread=c[4]))
    ttfts=json.loads((OUT/'frontend-output-frontier.json').read_text())['clock']['TTFTs_ms']
    batch=json.loads((OUT/'frontend-output-frontier.json').read_text())['clock']['client_profiled_batch_ms']
    permutations=list(itertools.permutations(range(8)));scenarios=[]
    for rho in [0,.5,.9,1]:
        for h in [.1,.5,2]:
            ends=[]
            for i,j in enumerate(jobs):
                oldgap=j['start_ms']-(jobs[i-1]['end_ms'] if i else 0)
                start=max(j['submit_ms'],(ends[-1] if i else 0)+oldgap)
                ends.append(start+j['parent_ms']-rho*j['native_ms']+h)
            deltas=[j['end_ms']-e for j,e in zip(jobs,ends)]
            endpoint=[]
            for perm in permutations:
                endpoint.append(max(ttfts)-max(t-deltas[k] for t,k in zip(ttfts,perm)))
            scenarios.append(dict(native_fraction_removed=rho,new_ms_per_job=h,completion_acceleration_ms_by_FIFO_ordinal=deltas,
              last_input_acceleration_ms=deltas[-1],P90_TTFT_fixed_downstream_lag_best_pairing_ms=max(endpoint),
              P90_TTFT_fixed_downstream_lag_worst_pairing_ms=min(endpoint),
              last_input_acceleration_over_profiled_batch_percent=100*deltas[-1]/batch))
    pins=[]
    for filename,key in [('input-renderer-execution-identity.json','runtime_sha256'),('input-renderer-extra-dependencies.json','sha256')]:
        for r in json.loads((OUT/filename).read_text())['sources']:
            if r.get('missing'):continue
            assert sha256(Path(r['local_path']).read_bytes()).hexdigest()==r[key];pins.append(dict(path=r['local_path'],sha256=r[key]))
    out=dict(status='LIMITED_INPUT_RENDERER_SOURCE_FIFO_VALUE_REVIEW',whole_chain_complete=False,bitwise_required=True,
      current_source_pins=pins,
      actual_path=['CompletionRequest.build_tok_params','OnlineRenderer.preprocess_completion/preprocess_cmpl','parse_model_prompt -> parse_dec_only_prompt',
        'BaseRenderer.render_cmpl_async -> render_prompts_async/tokenize_prompts_async','_tokenize_singleton_prompt_async pre-validation',
        'make_async/_executor -> _tokenize_prompt -> pooled TokenizersBackend.__call__ -> _encode_plus -> native encode_batch',
        '_build_tokens_prompt per-request list copy -> post-validation/truncation/padding/parallel fields',
        '_process_tokens_async -> tokens_input -> EngineInput arrival_time -> existing EngineCoreRequest/validation/cache/scheduler/H2D consumers'],
      fixed_branch=dict(model='decoder-only text, no multimodal data/offsets/echo or prompt embeds in frozen C8',request_max_tokens=1024,
        max_model_len=34816,source_derived_kwargs=dict(truncation=True,max_length=33793,add_special_tokens=True),
        renderer_num_workers='ModelConfig default1; saved launch has no flag per prior script identity. Completed trace actual one worker matches, not current live object introspection.',
        native_binding='v0.22.2 upstream encode_batch uses py.detach; trace shows main submitting7 more requests during first native call. Source tag not installed binary provenance.',
        pool='HfRenderer copies tokenizer then maybe_make_thread_pool(workers+1). Public calls borrow independent deep copies; raw _tokenizer/mutation methods explicitly outside thread-safe interface.'),
      FIFO=dict(jobs=jobs,submit_span_ms=jobs[-1]['submit_ms']-jobs[0]['submit_ms'],native_total_ms=sum(j['native_ms'] for j in jobs),
        parents_total_ms=sum(j['parent_ms'] for j in jobs),last_parent_end_ms=jobs[-1]['end_ms'],
        aggregate_queue_wait_ms=sum(j['queue_wait_ms'] for j in jobs),
        contract='Internal FIFO ordinal pairing:8 submissions,8 worker jobs/native calls; source maxworkers1 and no extra tokenizer job in window. No external request-ID association. Queue waits overlap across requests: never add their sum to batch or CPU work.'),
      paid_FIFO_scenarios=scenarios,
      endpoint_contract='Queue completion simulations retain observed parent-other costs and inter-job gaps, fixed task submission times and source FIFO. Endpoint pairing range assumes each downstream lag unchanged; engine admission/batching/prefill/CPU-GPU overlap and request-ID mapping are NOT known, so these are sensitivity scenarios, not measured/expected/tight causal request bounds. No cross-cohort/native-seed/metadata budget addition.',
      decisions=[dict(state='已确认可实现',scope='Existing renderer worker configuration and tokenizer-copy pool',
          qualification='Existing async capability, not a new fusion implementation or measured benefit. No service change. Shared raw backend mutable truncation/encode_special_tokens state must not replace the pool.'),
        dict(state='待确认',scope='Shared stable-prefix tokenization/pretokenization/BPE and immutable ID delivery',
          qualification='Whole native466ms is nontrivial; cached_tokenizer caches properties, not full promptIDs. Need exact stable regex/added-token boundary, immutable generation/config/flags, validation/error/offset semantics, cold insertion/warm hit/LRU/lookup/copy costs. Prefix length alone is not admissible.'),
        dict(state='已停止',scope='Blind concat of independently tokenized arbitrary common prefix/suffix, or bare concurrent calls through one mutable HF backend',
          qualification='No stable BPE/special-token boundary proof; HF explicitly mutates backend truncation/padding/split-special state. Stops these recipes, not guarded prefix reuse or independent pooled clones.'),
        dict(state='待确认',scope='Whole renderer/ID-publication/cache/next-submit joint delivery',
          qualification='Retain original perrequest ID values/ownership, EngineInput text/arrival/cache_salt and existing EngineCore/Engram/cache consumers. No new model/H2D deletion credit from tokenizer CPU optimization.')],
      remaining=['Generic HF constructor replay IDs/actual loader and service identity relation; source replay cannot replace captured serving IDs',
        'Decision-changing stable-prefix geometry and exact IDs/cold+warm CPU cost point if sufficiently large; otherwise move on',
        'Four startup warmup helpers/internal kernels and MM budget remain; do not close full scan by listing gaps'],
      analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_profile_collection=0,service_requests=0,new_explicit_NPU_work=0,remote_writes=[])
    (OUT/'input-renderer-frontier.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(submit_span_ms=out['FIFO']['submit_span_ms'],native_ms=out['FIFO']['native_total_ms'],parent_ms=out['FIFO']['parents_total_ms'],last_input_ms=jobs[-1]['end_ms'],
      half_h05=next(r for r in scenarios if r['native_fraction_removed']==.5 and r['new_ms_per_job']==.5),whole_chain_complete=False)))


if __name__=='__main__':main()
