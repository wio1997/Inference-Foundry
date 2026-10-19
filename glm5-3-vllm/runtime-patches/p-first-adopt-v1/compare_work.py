#!/usr/bin/env python3
"""Join completed client rows to immutable per-rank native forward dispatch logs."""
import argparse
import collections
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace-dir',type=Path,required=True);p.add_argument('--client',type=Path,required=True);a=p.parse_args()
    records=[json.loads(line) for f in a.trace_dir.glob('*.jsonl') for line in f.read_text().splitlines() if line.strip()]
    report=json.loads(a.client.read_text());rows=[]
    for client in report['rows']:
        if not client.get('D_chunks'):continue
        public=client['D_chunks'][0]['id'];prefix=public+'-0-';n=client['input_tokens'];enabled=client['mode']=='adopt' and client['D_request']['max_tokens']>1
        per_rank=[];internal=set()
        for rank in range(16):
            selected=sorted([(event['monotonic_ns'],r) for event in records if event['tp_rank']==rank for r in event['requests'] if r['req_id'].startswith(prefix)],key=lambda x:x[0])
            assert selected,(public,rank)
            ids={r['req_id'] for _,r in selected};assert len(ids)==1;internal|=ids
            first=selected[0][1]
            assert first['computed_before']==n-(not enabled),(public,rank,first)
            assert first['output_history_count']==int(enabled),(public,rank,first)
            expected=client['P_response']['kv_transfer_params']['last_token_id'] if enabled else client['P_request']['prompt'][-1]
            assert first['host_query_token']==expected,(public,rank,first)
            per_rank.append({'rank':rank,'dispatches':len(selected),'scheduled_tokens_sum':sum(r['scheduled_tokens'] for _,r in selected),'first':first})
        assert len(internal)==1 and len({r['dispatches'] for r in per_rank})==1
        rows.append({'public_id':public,'mode':client['mode'],'input_tokens':n,'output_tokens':len(client['token_ids']),'adoption_observed_all16':enabled,'dispatches_per_rank':per_rank[0]['dispatches'],'scheduled_tokens_sum_per_rank':per_rank[0]['scheduled_tokens_sum'],'first_host_position':per_rank[0]['first']['computed_before'],'initial_host_history':per_rank[0]['first']['initial_history_head'],'matches_native_control':client.get('matches_native_control'),'numeric_pass':client.get('numeric_pass'),'all16_agree':True})
    print(json.dumps({'rows':rows,'scope':'Actual native target forward dispatch calls including queued work in the preserved completed trace; not kernel counts, device duration or graph-padded tensor rows. Input-state checks are host metadata.'},indent=2))


if __name__=='__main__':main()
