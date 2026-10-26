#!/usr/bin/env python3
"""Join Run576 scoped capture to actual measured client and Runtime cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def need(ok,why):
    if not ok:raise ValueError(why)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    arm=a.arm
    capture=json.loads((arm/'capture_admission.json').read_text())
    client=json.loads((arm/'client_admission.json').read_text())
    need(capture['status']=='scoped_diagnostic_accepted', 'capture not admitted')
    need(client['status']=='client_two_phase_admitted' and client['request_count']==96,
         'client phases not admitted')
    for phase in ('warmup','measured'):
        s=client[f'{phase}_summary']
        need(s['phase']==phase and s['n']==48 and s['success']==48 and s['fail']==0 and
             s['concurrency']==12 and s['max_tokens']==1024,
             f'{phase} frozen contract')
    need(int((arm/'server_post_count.txt').read_text())==96,'server POST count')
    ids=capture['request_ids']
    need(len(ids)==12 and len(set(ids))==12,'captured request IDs')
    index=client['request_index']
    need(len(index)==96,'client index length')
    match=[]
    for req in ids:
        row=[r for r in index if req==r['response_id'] or
             req.startswith(r['response_id']+'-')]
        need(len(row)==1,f'client request identity ambiguous: {req}')
        need(row[0]['phase']=='measured','captured cohort is not measured')
        match.append(row[0])
    need(len({r['response_id'] for r in match})==12,'duplicate client response match')
    need({r['dataset_index'] for r in match}==set(range(12)),
         'cohort5 is not first measured request set')
    runtime_paths=sorted((arm/'runtime').glob('rank*_cohort5.json'))
    need(len(runtime_paths)==8,'exactly eight Runtime cohort5 rows')
    runtime=[json.loads(p.read_text()) for p in runtime_paths]
    need(sorted(r['rank'] for r in runtime)==list(range(8)), 'Runtime TP8 ranks')
    for r in runtime:
        need(r['cohort']==5 and r['req_ids']==ids and
             r['cycles']==capture['total_cycles'], 'Runtime identity/cycle')
        need(r['cycles']>capture['cycle'] and r['host_mirror_exact'] is True,
             'Runtime cycle/Host mirror')
        need(r['target_graph_requested'] is True and r['target_graph_mode']=='FULL',
             'Runtime FULL Graph')
        need(r['generated_output_counts']==[1024]*12,
             'Runtime output counts')
        raw=json.loads((arm/'capture'/f"rank{r['rank']}_cohort5.json").read_text())
        need(raw['run_tag']==capture['run_tag'] and raw['request_ids']==r['req_ids'] and
             raw['cycles']==r['cycles'],'raw capture/runtime join')
    report=dict(status='diagnostic_capture_client_runtime_join_pass',
                run_tag=capture['run_tag'],cohort=5,cycle=capture['cycle'],
                request_ids=ids,matched_measured_dataset_indices=sorted(r['dataset_index'] for r in match),
                rank_count=8,client_requests=96,formal_tps_eligible=False,
                fixed_formal_W0=False,numeric_bound_update=False,
                limits=['One diagnostic run; no transfer to Run99 W0.',
                        'Capture reference snapshots do not certify native dynamic arguments.',
                        'Current diagnostic timing is not a Resource or Scheduling floor.'])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cohort','cycle','rank_count')}))


if __name__=='__main__':main()
