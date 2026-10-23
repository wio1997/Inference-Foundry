#!/usr/bin/env python3
"""Count each scheduled batch and speculative observation once, never per alias."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'evidence/E33-work-inputs.json.gz';records=json.loads(gzip.decompress(p.read_bytes()))['records'];cells=[];flat=[]
    for run in sorted({x['run_id'] for x in records}):
        rows=[x for x in records if x['run_id']==run];is_di='-DI-' in run
        remaining_prefill=3328 if '-HOT-' in run else 32768
        by_rank=collections.defaultdict(lambda:dict(schedule_calls=0,scheduled_tokens=0,speculative_calls=0,stages=collections.defaultdict(lambda:dict(calls=0,scheduled_tokens=0,request_slots=0)),raw_spec=collections.Counter()))
        seen=set()
        for item in rows:
            e=item['event'];rank=e.get('rank') if is_di else e.get('dp_rank');assert rank in range(8)
            g=by_rank[rank]
            if is_di:
                w=e['extra'];event=e['event_name'];key=(rank,event,e['iteration_id'])
                assert key not in seen;seen.add(key)
                if event=='post_schedule':
                    g['schedule_calls']+=1;g['scheduled_tokens']+=w['num_batched_tokens']
                    stage=('prefill' if w['prompt_run'] else 'decode')+('_empty' if w['is_empty'] else '_nonempty')
                    h=g['stages'][stage];h['calls']+=1;h['scheduled_tokens']+=w['num_batched_tokens'];h['request_slots']+=len(w['scheduled'])
                    flat.append(dict(run_id=run,rank=rank,kind=stage,event_id=e['iteration_id'],scheduled_tokens=w['num_batched_tokens'],request_slots=len(w['scheduled']),num_generated_tokens='',num_verified_tokens='',num_draft_tokens='',num_accepted_tokens=''))
                elif event=='spec_counters':
                    g['speculative_calls']+=1
                    for k in ['num_generated_tokens','num_verified_tokens']:g['raw_spec'][k]+=w[k]
                    # scheduled_request_ids/count in these events refer to a later
                    # schedule in the async pipeline.  They MUST NOT be used for
                    # per-request attribution or for the bonus-token denominator.
                    flat.append(dict(run_id=run,rank=rank,kind='native_spec_batch_total',event_id=e['iteration_id'],scheduled_tokens='',request_slots='',num_generated_tokens=w['num_generated_tokens'],num_verified_tokens=w['num_verified_tokens'],num_draft_tokens='',num_accepted_tokens=''))
            else:
                w=e['work'];event=e['source'];key=(e['pid'],e['event_seq']);assert key not in seen;seen.add(key)
                if event=='vllm.worker.NPUWorker.execute_model':
                    g['schedule_calls']+=1;g['scheduled_tokens']+=w['total_num_scheduled_tokens']
                    stage='batch_with_spec_tokens' if w.get('num_spec_decode_tokens',0) else 'batch_without_spec_tokens'
                    h=g['stages'][stage];h['calls']+=1;h['scheduled_tokens']+=w['total_num_scheduled_tokens'];h['request_slots']+=w['num_scheduled_requests']
                    g.setdefault('scheduled_spec_tokens',0);g['scheduled_spec_tokens']+=w.get('num_spec_decode_tokens',0)
                    flat.append(dict(run_id=run,rank=rank,kind=stage,event_id=e['event_seq'],scheduled_tokens=w['total_num_scheduled_tokens'],request_slots=w['num_scheduled_requests'],num_generated_tokens='',num_verified_tokens='',num_draft_tokens='',num_accepted_tokens=''))
                elif event=='vllm.scheduler.make_spec_decoding_stats':
                    g['speculative_calls']+=1
                    for k in ['num_draft_tokens','num_accepted_tokens','num_invalid_spec_tokens_for_request']:
                        if isinstance(w.get(k),int):g['raw_spec'][k]+=w[k]
                    flat.append(dict(run_id=run,rank=rank,kind='native_spec_per_request_args',event_id=e['event_seq'],scheduled_tokens='',request_slots='',num_generated_tokens='',num_verified_tokens='',num_draft_tokens=w.get('num_draft_tokens',''),num_accepted_tokens=w.get('num_accepted_tokens','')))
        assert set(by_rank)==set(range(8))
        counts=dict(schedule_calls=sum(v['schedule_calls'] for v in by_rank.values()),scheduled_tokens=sum(v['scheduled_tokens'] for v in by_rank.values()),speculative_calls=sum(v['speculative_calls'] for v in by_rank.values()))
        spec=collections.Counter()
        for g in by_rank.values():spec.update(g['raw_spec'])
        checks={};normalization={}
        if is_di:
            checks['per_rank_prefill_tokens_match_request_work']=all(g['stages']['prefill_nonempty']['scheduled_tokens']==8*remaining_prefill for g in by_rank.values())
            checks['per_rank_decode_rows_match_native_verified_counters']=all(g['stages']['decode_nonempty']['scheduled_tokens']==g['raw_spec']['num_verified_tokens'] for g in by_rank.values())
            checks['per_rank_decode_batches_match_spec_counter_calls']=all(g['stages']['decode_nonempty']['calls']==g['speculative_calls'] for g in by_rank.values())
            bonus=sum(g['stages']['decode_nonempty']['request_slots'] for g in by_rank.values())
            accepted=spec['num_generated_tokens']-bonus;verified=spec['num_verified_tokens']-bonus
            normalization=dict(base_rows=bonus,accepted_candidate_rows=accepted,verified_candidate_rows=verified,accepted_per_verified_candidate=accepted/verified,
                               generated_before_final_truncation=spec['num_generated_tokens']+64,
                               generated_minus_delivered=spec['num_generated_tokens']+64-8192,
                               definition='Per source sampler.py: num_generated=[req_cnt]+accepted_by_step, num_verified=[req_cnt]+is_verify_by_step. Base req_cnt is summed from actual nonempty decode schedules, not the delayed spec-event request IDs.')
        else:
            normalization=dict(base_rows_reported_by_stats=counts['speculative_calls'],accepted_candidate_rows=spec['num_accepted_tokens'],draft_candidate_rows_reported=spec['num_draft_tokens'],
                               accepted_per_reported_draft=spec['num_accepted_tokens']/spec['num_draft_tokens'],
                               generated_before_final_truncation=counts['speculative_calls']+spec['num_accepted_tokens']+64,
                               generated_minus_delivered=counts['speculative_calls']+spec['num_accepted_tokens']+64-8192,
                               scheduled_spec_rows=sum(g['scheduled_spec_tokens'] for g in by_rank.values()),
                               definition='Stats hook arguments are per-request observations. Scheduled draft rows and reported stats can differ because of asynchronous completion/cancellation; the difference is retained.')
        assert all(checks.values())
        counts['remaining_prefill_request_tokens']=64*remaining_prefill
        counts['scheduled_tokens_minus_remaining_prefill']=counts['scheduled_tokens']-64*remaining_prefill
        cells.append(dict(run_id=run,framework='di' if is_di else 'vllm',counts=counts,raw_spec=dict(spec),normalization=normalization,checks=checks,ranks=dict(by_rank),
                          limits=['Empty local DP schedules still participate in EP and must not automatically be called wasted work.',
                                  'Scheduled token counts omit hidden DP/graph padding, and are not FLOP counts.',
                                  'vLLM execute_dummy_batch and sample_tokens lack direct diagnostic hooks in this capture.',
                                  'DI spec-event request IDs are temporally misassociated by the async last_meta probe; they are preserved only in raw input and excluded from attribution.',
                                  'Acceptance differences depend on actual generated sequences and implementation; no causal speedup is assigned from these ratios alone.']))
    result=dict(input=str(p.relative_to(ROOT)),input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cells=cells)
    (ROOT/'data/E33-work-ledger.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'data/E33-work-ledger.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    for c in cells:print(c['run_id'],c['counts'],c['normalization'])


if __name__=='__main__':main()
