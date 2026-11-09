#!/usr/bin/env python3
"""Audit completed frontend call tree and conditional token-to-byte prices.

No new inference. Chrome wall clock is explicit; nested costs are unions.
Favourable endpoint credits deliberately relax ownership and producer timing.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import zlib

OUT=Path('audit/execution-search')


def union(xs):
    out=[]
    for a,b in sorted(xs):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return out


def length(xs,lo,hi):
    return sum(max(0,min(b,hi)-max(a,lo)) for a,b in xs)/1000


def main():
    p=OUT/'frontend-output-events.json.zlib';packed=p.read_bytes()
    receipt=json.loads((OUT/'frontend-output-trace-receipt.json').read_text())
    assert sha256(packed).hexdigest()==receipt['sha256']
    d=json.loads(zlib.decompress(packed));names=d['names'];rows=d['rows']
    ids={r[5]:r for r in rows};assert len(ids)==len(rows)
    children=defaultdict(list);missing=[]
    for r in rows:
        parent=ids.get(r[6])
        if parent is None:missing.append(r[5]);continue
        assert parent[3:5]==r[3:5]
        assert parent[1]-.002<=r[1] and parent[1]+parent[2]+.002>=r[1]+r[2]
        children[parent[5]].append([r[1],r[1]+r[2]])
    clients=json.loads(zlib.decompress((OUT/'inactive-dp-framework-clients.json.zlib').read_bytes()))
    client=next(r['data'] for r in clients if r['path'].endswith('/profiled.json'))
    base=d['metadata']['baseTimeNanoseconds']
    rr=client['requests'];offsets=[r['start_wall_ns']-r['start_mono_ns'] for r in rr]
    # Per-client wall sample is used for its deadline; batch start takes the
    # earliest offset (favourable window, only tens of microseconds uncertainty).
    lo=(client['start_mono_ns']+min(offsets)-base)/1000
    hi=(client['end_mono_ns']+max(offsets)-base)/1000
    ends=[(r['first_token_mono_ns']+off-base)/1000 for r,off in zip(rr,offsets)]
    ttfts=[(r['first_token_mono_ns']-r['start_mono_ns'])/1e6 for r in rr]
    batch=(client['end_mono_ns']-client['start_mono_ns'])/1e6
    native=lambda n:'DecodeStream' in n and 'step' in n
    scopes={
      'detokenizer_parent':lambda n,r:'/engine/detokenizer.py' in n,
      'prompt_seed_native_large8':lambda n,r:native(n) and r[2]>5000,
      'prompt_seed_native_and_constructor':lambda n,r:(native(n) and r[2]>5000) or ('detokenizer.py' in n and '__init__' in n),
      'all_decode_stream_native':lambda n,r:native(n),
      'completion_generator':lambda n,r:'/completion/serving.py' in n and 'completion_stream_generator' in n,
      'JSON_model_dump':lambda n,r:'model_dump_json' in n,
      'output_processor_parent':lambda n,r:'output_processor.py' in n and 'process_outputs' in n,
      'uvicorn_send':lambda n,r:'httptools_impl.py' in n and 'send' in n,
      'generic_str_encode_navigation_only':lambda n,r:'str' in n and 'encode' in n,
      'input_encode_batch_navigation_only':lambda n,r:'encode_batch' in n,
      'whole_output_delivery_relaxed':lambda n,r:any(s in n for s in ['/engine/output_processor.py','/engine/detokenizer.py','/engine/logprobs.py','vllm/vllm/outputs.py','sse_keep_alive.py']) or
        ('/completion/serving.py' in n and 'completion_stream_generator' in n) or
        ('/engine/async_llm.py' in n and any(s in n for s in ['output_handler','generate'])) or
        ('/engine/core_client.py' in n and any(s in n for s in ['process_outputs_socket','get_output_async'])) or
        ('starlette/responses.py' in n and any(s in n for s in ['stream_response','wrap'])) or
        ('httptools_impl.py' in n and 'send' in n) or
        ('async_utils.py' in n and 'merge_async_iterators' in n),
    }
    prices={}
    for scope,pred in scopes.items():
        selected=[r for r in rows if pred(names[r[0]],r)]
        intervals=union([(r[1],r[1]+r[2]) for r in selected])
        credits=[length(intervals,lo,end) for end in ends]
        total=length(intervals,lo,hi)
        saved=max(ttfts)-max(t-c for t,c in zip(ttfts,credits))
        prices[scope]=dict(call_events=len(selected),inclusive_sum_ms=sum(r[2] for r in selected)/1000,
            clipped_interval_union_ms=total,favourable_fixed_schedule_batch_percent=100*total/batch,
            favourable_prefix_credit_by_first_deadline_ms=credits,
            favourable_P90_TTFT_credit_ms=saved,favourable_P90_TTFT_percent=100*saved/max(ttfts))
        if scope in ['prompt_seed_native_and_constructor','prompt_seed_native_large8']:
            large=[r for r in rows if native(names[r[0]]) and r[2]>5000]
            assert len(large)==8
            charges=[sum(lo<=r[1]<end for r in large) for end in ends]
            prices[scope]['paid_conditional']=[dict(rho=rho,new_cost_per_seed_ms=h,
              batch_credit_ms=rho*total-8*h,batch_percent=100*(rho*total-8*h)/batch,
              P90_TTFT_credit_ms=max(ttfts)-max(t-rho*c+h*k for t,c,k in zip(ttfts,credits,charges)))
              for rho in [0,.5,1] for h in [.1,.5]]
    stats=[]
    groups=defaultdict(list)
    for r in rows:groups[r[0]].append(r)
    for index,n in enumerate(names):
        rs=groups[index]
        if any(s in n for s in ['DecodeStream','decode_next','model_dump_json','encode_batch','process_outputs','make_request_output','_new_completion_output','update_from_output','flow_control.py','RequestOutput','with_sse_keep_alive']):
            self_ms=sum(max(0,r[2]/1000-length(union(children[r[5]]),r[1],r[1]+r[2])) for r in rs)
            stats.append(dict(name=n,calls=len(rs),inclusive_ms=sum(r[2] for r in rs)/1000,
              self_interval_remainder_ms=self_ms,max_call_us=max(r[2] for r in rs),threads=sorted({r[4] for r in rs})))
    large=sorted([r for r in rows if native(names[r[0]])],key=lambda r:r[2],reverse=True)[:8]
    pins=[]
    for name,key in [('frontend-output-execution-identity.json','runtime_sha256'),('frontend-output-extra-dependencies.json','sha256')]:
        x=json.loads((OUT/name).read_text())
        for r in x['sources']:
            assert sha256(Path(r['local_path']).read_bytes()).hexdigest()==r[key]
            pins.append(dict(path=r['local_path'],sha256=r[key]))
    out=dict(scope='Limited actual fast detokenizer -> collector -> completion SSE -> ASGI byte transport source/value review',
      whole_chain_complete=False,bitwise_replacement_required=True,
      evidence=dict(events=len(rows),unique_Python_IDs=len(ids),contained_same_thread_parent_edges=len(rows)-len(missing),root_or_missing_parent_events=len(missing),
        thread_counts={str(k):v for k,v in Counter((r[3],r[4]) for r in rows).items()},source_pins=pins),
      clock=dict(baseTimeNanoseconds=base,wall_formula='baseTimeNanoseconds + Chrome ts_us*1000',
        wall_minus_mono_min_ns=min(offsets),wall_minus_mono_max_ns=max(offsets),offset_spread_ns=max(offsets)-min(offsets),
        client_profiled_batch_ms=batch,P90_TTFT_ms=max(ttfts),quantile='8 requests, nearest-rank P90=max',
        TTFTs_ms=ttfts,window_ts_us=[lo,hi]),
      price_contract='Selected active call intervals are unioned (never sum parent+child). Python self interval remainder excludes recorded child intervals but is NOT sampled CPU-active time: preemption/C-extension/unrecorded waits remain. Credits erase eligible pre-endpoint intervals with ownership/producer arrivals/engine scheduling fixed and relaxed exposure; no measured speedup, causal lower bound or reliable request association. All scopes overlap. Background encode navigation is not output work or event-loop blocking.',
      scopes=prices,selected_call_stats=stats,
      eight_large_native_steps=[dict(ts_us=r[1],dur_us=r[2],thread=r[4],python_id=r[5]) for r in large],
      source_result=dict(fast_used='8 constructors,8192 scalar steps;8 large10ms steps consistent with source initial full-prompt priming, not observed perrequest args',
        actual_decoder='tokenizers0.22.2 ByteLevel; constructor exposes ids/skip only, no state clone/prefix setter API',
        seed_price='Native8 is a tighter ceiling for a candidate that retains the original constructor validation/copy. Broader constructor+native scope is an oracle including nested3x8 constructor events, not candidate-removable cost.',
        source_mechanism='Upstream v0.22.2 step_decode_stream decodes seeded prompt then prompt+newIDs. Installed binary behaviour still needs point fixture; tag source is not proof of binary provenance.',
        preserving=['Stop=min_tokens+earliest completed stop end/list-order ties; streaming holdback/final flush; special spacing; UTF8 None and error recovery',
          'Collector already merges pending DELTA, generator consumes per-delta text/IDs and exact usage/finish/error/DONE. No blanket fullhistory decode or prompt copy credit.',
          'Explicit None/exclude_unset JSON fields, UTF8/SSE framing, ASGI disconnect/backpressure/HTTP chunk framing and transport storage ownership'],
        absent='Fixed request logprobs disabled: no heavy probability-to-text materialization; retain cheap guard/metadata semantics'),
      decisions=[dict(state='待确认',name='ByteLevel guarded prompt-seed elimination/shared immutable decode boundary',
        next='Freeze and run CPU-only installed tokenizer binary point fixture: exact per-step UTF8 bytes/None/errors and guard hit, fallback for unsafe suffix; then charge guard+constructor and match real input IDs. No NPU/model/service.',
        gaps=['Whole legal byte-boundary proof and installed binary equality','Actual same-load seed IDs/flags, guard cost and exposure; source/8large calls are not request-ID correlation','SSE byte/reference admission before production replacement']),
        dict(state='已停止',name='Standalone JSON serialization removal for same C8 independent1pct budget',reason='Exact allfree prefix price below1pct TTFT; whole-window JSON <1pct batch. Does not prune whole output/seed or different ordinary load.'),
        dict(state='待确认',name='Whole output/BatchFrame/next-submit host delivery',reason='609ms selected union relaxed; significant waits/metrics/dispatch overlap, cannot claim exposure or remove retained error/counters. Preserve earlier producer/lease and whole-native gates.'),
        dict(state='待确认',name='Input tokenization batching/shared prefix state',reason='8 encode_batch calls serialized on background thread,466ms; need actual renderer call/queue/IDs and no lost BPE/special-boundary semantics. Not main-thread blocking or a benefit estimate.')],
      new_implementable_admission=[],new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[],
      remaining=['Input renderer/tokenization real caller/producer and cache/parallel execution next after bounded seed point',
        'Four warmup helpers/internal kernels and inherited startup MM budget remain; other physical/native/ordinary gaps unchanged'],
      analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'frontend-output-frontier.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(call_tree='PASS containment/identity',batch_ms=batch,P90_ms=max(ttfts),prices={k:dict(union_ms=v['clipped_interval_union_ms'],P90_ms=v['favourable_P90_TTFT_credit_ms']) for k,v in prices.items()},whole_chain_complete=False)))


if __name__=='__main__':main()
