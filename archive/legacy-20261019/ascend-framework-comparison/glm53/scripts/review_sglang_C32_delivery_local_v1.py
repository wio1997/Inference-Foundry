"""CPU43 same-clock edges from actual C32 records; no resource/model calls."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RID='20261017T2052-sglang-formal-C32-L04-control-v3'
def quantiles(values):
    values=sorted(values)
    def at(p):
        index=(len(values)-1)*p;lo=math.floor(index);hi=math.ceil(index)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    return {'min':values[0],'P50':at(.5),'P90':at(.9),'P99':at(.99),'max':values[-1]}
def main():
    path=ROOT/'runs'/RID/'formal-result.remote.json';raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='a28905da2c5c6ad4a231ab275d962d0927bd6463489d359f31e451d8fe3cb2eb'
    report=json.loads(raw);assert report['phase']=='COMPLETE' and report['concurrency']==32 and len(report['formal']['rows'])==128
    data={k:[] for k in ['gateway_first_forward_to_client_first_valid_ms','gateway_terminal_to_client_outer_terminal_ms','last_client_valid_to_forwarded_DONE_ms','P_terminal_after_first_client_valid_seconds','SSE_valid_chunks_per_request','native_verify_steps_per_request','verify_plus_correct_minus_target_tokens']}
    P_before_final=0;histogram=[0]*6;verify=0;correct=0
    for row in report['formal']['rows']:
        r=row['result'];n=r['native'];g=r['terminal']['gateway'];valid=n['extra_perf_data']['glm53_valid_output_chunk_times'];spec=n['extra_details_data']['glm53_native_sglext']['spec_tokens_details']
        assert r['logical_success'] and len(valid)>0 and g['status']=='D_STREAM_COMPLETE'
        data['gateway_first_forward_to_client_first_valid_ms'].append((valid[0]-g['first_forward_monotonic_ns']/1e9)*1000)
        data['gateway_terminal_to_client_outer_terminal_ms'].append((row['outer_terminal_perf_counter_ns']-g['gateway_terminal_monotonic_ns'])/1e6)
        data['last_client_valid_to_forwarded_DONE_ms'].append((g['native_done_forwarded_monotonic_ns']/1e9-valid[-1])*1000)
        data['P_terminal_after_first_client_valid_seconds'].append(g['prefill_terminal_monotonic_ns']/1e9-valid[0])
        P_before_final+=int(g['prefill_terminal_monotonic_ns']/1e9<=valid[-1])
        data['SSE_valid_chunks_per_request'].append(len(valid));data['native_verify_steps_per_request'].append(spec['spec_verify_ct'])
        data['verify_plus_correct_minus_target_tokens'].append(spec['spec_verify_ct']+spec['spec_num_correct_drafts']-600)
        histogram=[a+b for a,b in zip(histogram,spec['spec_correct_drafts_histogram'])];verify+=spec['spec_verify_ct'];correct+=spec['spec_num_correct_drafts']
    result=dict(project_id='glm53',run_id=RID,formal_requests=128,model_requests=0,SDK_init_calls=0,actual_resource_calls=0,
                report_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                timing={k:quantiles(v) for k,v in data.items()},P_terminal_preceded_final_output_requests=P_before_final,
                formal_MTP={'verify_steps':verify,'correct_drafts':correct,'histogram':histogram,'num_proposed_per_verify':5},
                scope='CPU43 same Linux monotonic clock. Edge observations only, not model kernel timing. Negative last-valid/DONE edges preserved. Does not exclude internal D/intermediate chunk buffering or collective contention. MTP sends multiple tokens per SSE chunk; chunk interval is not per-token interval. Draft has one KV layer versus78target layers; do not model each draft as a full target-model pass.')
    p=ROOT/'results/sglang-C32-control-delivery-timing-review-v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'timing':result['timing'],'P_terminal_preceded_final_output_requests':P_before_final,'formal_MTP':result['formal_MTP']},indent=2))
if __name__=='__main__':main()
