"""Compact diagnostic reduction; sampled counters are not full-run token counts."""
import argparse,json,re,hashlib
from pathlib import Path
from phase_runner import atomic_json
INTEREST=('num_requests_running','num_requests_waiting','kv_cache_usage_perc','prefix_cache_queries_total','prefix_cache_hits_total','spec_decode_num_drafts_total','spec_decode_num_draft_tokens_total','spec_decode_num_accepted_tokens_total','generation_tokens_total','request_prefill_time_seconds_sum','request_decode_time_seconds_sum')
def reduce(root):
    root=Path(root);state=json.loads((root/'state.json').read_text());result={'run_id':state['run_id'],'status':state['status'],'kind':'diagnostic','arms':{},'limits':['2-second metrics snapshots omit unsampled tails; no calibrated cross-host event clock','2048 output does not substitute 61440 output SLA or stable capacity','resident prefix state differs across arms; returned cached_tokens may disagree with engine metrics']}
    for arm in [s['id'] for s in json.loads((root/'controller_spec.json').read_text())['stages']]:
        path=root/arm/'probe_result.json'
        if not path.exists():continue
        j=json.loads(path.read_text());a={'valid':j['valid'],'elapsed_s':j['elapsed_s'],'effective_tps':j['effective_tps'],'requests':[{k:r.get(k) for k in ('id','usage','first_content_s','tpot_s','elapsed_s','finish_reason','done','content_sha256')} for r in j['requests']],'telemetry':{}}
        samples={}
        for line in (root/arm/'telemetry.jsonl').read_text().splitlines():
            row=json.loads(line);role=row['role'];vals={}
            for raw in row.get('text','').splitlines():
                if raw.startswith('#'):continue
                name=raw.split('{',1)[0].split(' ',1)[0]
                if name.startswith('vllm:') and name[5:] in INTEREST:
                    try:vals[name]=vals.get(name,0)+float(raw.rsplit(' ',1)[1])
                    except ValueError:pass
            if vals:samples.setdefault(role,[]).append(vals)
        for role,rows in samples.items():
            names=set().union(*(set(r) for r in rows));a['telemetry'][role]={'samples':len(rows),'metrics':{n:{'first':rows[0].get(n),'last':rows[-1].get(n),'min':min(r[n] for r in rows if n in r),'max':max(r[n] for r in rows if n in r),'sampled_delta':rows[-1].get(n,0)-rows[0].get(n,0)} for n in sorted(names)}}
            m=a['telemetry'][role]['metrics'];draft=m.get('vllm:spec_decode_num_draft_tokens_total',{}).get('sampled_delta',0);accepted=m.get('vllm:spec_decode_num_accepted_tokens_total',{}).get('sampled_delta',0);drafts=m.get('vllm:spec_decode_num_drafts_total',{}).get('sampled_delta',0)
            a['telemetry'][role]['sampled_acceptance_rate']=accepted/draft if draft>0 else None;a['telemetry'][role]['sampled_mean_acceptance_length']=1+accepted/drafts if drafts>0 else None
        result['arms'][arm]=a
    result['same_prompt0_c1_hash_equal']=result['arms'].get('c1a',{}).get('requests',[{}])[0].get('content_sha256')==result['arms'].get('c1b',{}).get('requests',[{}])[0].get('content_sha256') if 'c1b' in result['arms'] else None
    atomic_json(root/'reduction.json',result)
    print(json.dumps(result,ensure_ascii=False));return 0 if state['status']=='completed' and all(a['valid'] for a in result['arms'].values()) else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir');a=p.parse_args();raise SystemExit(reduce(a.run_dir))
