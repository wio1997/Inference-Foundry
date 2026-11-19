"""Join active Router TCP-counter windows with the next original AISBench response."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from audit_router_histogram_boundary_g1483 import bind_db, parse_client, read_packet
from compare_first_wave_g1479 import digest, load_decoder


FILES = {
    'probe': 'G1484_router_socket_boundaries/P_tcp_probe.stdout',
    'tcp_observer_source': 'G1484_router_socket_boundaries/observe_transport.py',
    'tcp_window': 'G1484_router_socket_boundaries/P_tcp_window.jsonl',
    'prior_client_receipt': 'G1483_final_client_exact_timing_2204.json',
    'prior_client_db': 'G1483_router_timing_boundary/final_19_timing.db',
    'next_client_receipt': 'G1484_final_client_exact_timing_2204.json',
    'next_client_db': 'G1484_router_socket_boundaries/final_20_timing.db',
    'next_client_event': 'G1484_router_socket_boundaries/D_wait_20.json',
    'client_clock_before': 'G1484_D_client_clock_before_2204.json',
    'client_clock_after': 'G1484_D_client_clock_after_2204.json',
    'pd_clock_before': 'G1484_PD_clock_before.json',
    'pd_clock_after': 'G1484_PD_clock_after.json',
    'decode_log': 'G1484_D_activity_2204.json',
}


def wall(value):
    return datetime.fromisoformat(value).timestamp()


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def first_socket_windows(probe, records):
    ready, = [r for r in records if r['event']=='ready']
    assert probe['event']=='probe' and probe['connection_counts']==ready['counts']=={'prefill':13,'decode':13}
    pid=probe['router_listener_PID']
    assert all(r['router_listener_PID']==pid for r in records)
    assert all(c['bytes_received']==c['data_segments_in']==0 for c in probe['connections'].values())
    finished=records[-1]
    assert finished['event']=='done' and finished['same_listener_alive']
    marker, = [r for r in records if r['event']=='observed_one_router_increment']
    samples = [r for r in records if r['event']=='sample']
    assert samples[0]['router_completed_outer_count']==101 and samples[-1]['router_completed_outer_count']==102
    pos=samples.index(next(s for s in samples if s['from_P_UTC']==marker['after']['from_P_UTC']))
    assert 0 < pos < len(samples)-1
    assert all(c['bytes_received']==0 for c in samples[0]['connections'].values() if c['role']=='prefill')
    router_edge = [wall(samples[pos-1]['from_P_UTC']), wall(samples[pos+1]['from_P_UTC'])]
    result={role:{'persistent_socket_first_nonzero_windows':[],
                  'first_seen_positive_socket_count':0,'disappearing_socket_count':0}
            for role in ('prefill','decode')}
    def add_onset(first, second, current, pre_window):
        result[current['role']]['persistent_socket_first_nonzero_windows'].append({
           'bounds_P_UTC':[first['from_P_UTC'],second['to_P_UTC']],
           'receive_onset_may_precede_dense_sample_window':pre_window,
           'received_bytes_as_of_second_sample':current['bytes_received'],
           'receive_segments_as_of_second_sample':current['data_segments_in']})

    starting=probe['connections']
    first_sample=samples[0]
    assert set(first_sample['connections'])==set(starting)
    assert wall(probe['to_P_UTC'])<wall(first_sample['from_P_UTC'])
    for key,current in first_sample['connections'].items():
        previous=starting[key]
        assert current['role']==previous['role'] and current['router_fd']==previous['router_fd']
        if current['bytes_received']>0:
            assert current['data_segments_in']>0
            add_onset(probe,first_sample,current,True)
    for before, after in zip(samples, samples[1:]):
        old, new = before['connections'],after['connections']
        assert wall(before['to_P_UTC'])<wall(after['from_P_UTC'])
        assert wall(after['to_P_UTC'])-wall(before['from_P_UTC'])<4
        for key in old.keys() - new.keys():
            result[old[key]['role']]['disappearing_socket_count']+=1
        for key in new.keys() - old.keys():
            entry = new[key]
            if entry['bytes_received']>0:
                result[entry['role']]['first_seen_positive_socket_count']+=1
        for key in old.keys() & new.keys():
            earlier, current=old[key],new[key]
            assert earlier['role']==current['role'] and earlier['router_fd']==current['router_fd']
            assert earlier['bytes_received']<=current['bytes_received']
            if earlier['bytes_received']==0<current['bytes_received']:
                assert current['data_segments_in']>earlier['data_segments_in']
                add_onset(before,after,current,False)
    return result, router_edge, len(samples)


def client_first_time(files, paths, decoder):
    prev=parse_client(files['prior_client_receipt'],19)
    latest=parse_client(files['next_client_receipt'],20)
    observation=json.loads(files['next_client_event'])
    assert observation['same_runner_live'] and observation['outer_PID']==latest['runner_PID']
    assert observation['subphase']=='cold' and observation['observed_complete_lines']==20
    before=bind_db(paths['prior_client_db'],prev,decoder)
    full=bind_db(paths['next_client_db'],latest,decoder)
    assert all(full[idx]==value for idx,value in before.items())
    new,=full.keys()-before.keys()
    mono=full[new][1]
    pin_clock = [read_packet(files['client_clock_'+part],'G1476_D_CLIENT_CLOCK_JSON') for part in ('before','after')]
    assert all(x['same_runner_live'] and x['client_container_matches_host_time_namespace']
               and x['perf_counter_impl']=='clock_gettime(CLOCK_MONOTONIC)' for x in pin_clock)
    clock_pairs=[[x['wall_seconds']-x['monotonic_bracket_s'][1],x['wall_seconds']-x['monotonic_bracket_s'][0]] for x in pin_clock]
    offset_tm=[min(x[0] for x in clock_pairs),max(x[1] for x in clock_pairs)]
    assert offset_tm[1]-offset_tm[0]<0.001
    samples=[json.loads(files['pd_clock_'+part]) for part in ('before','after')]
    assert all(len(x['samples'])==12 for snapshot in samples for x in snapshot['roles'].values())
    skew=[s['prefill_minus_decode_realtime_ns'] for s in samples]
    low,high=min(x[0] for x in skew)/1e9,max(x[1] for x in skew)/1e9
    assert 19<low<high<21
    return [mono+offset_tm[0]+low,mono+offset_tm[1]+high], [low,high]


def decode_activity_before_p(raw, p_first_lower, skew):
    d=read_packet(raw,'G1468_COLD_LOGS')
    assert d['role']=='decode' and d['epoch']==1317
    counters=[]
    for line in d['reports']['engine']['selected_lines']:
        match=re.match(r'^\[([\d-]+ [\d:]+) TP0 EP0\] Decode batch',line)
        if not match:
            continue
        stamp=datetime.fromisoformat(match[1]).replace(tzinfo=timezone.utc)
        p_upper=stamp.timestamp()+1+skew[1]
        if p_first_lower-120 < p_upper < p_first_lower:
            series={key:int(val) for key,val in re.findall(r'#([a-z-]+): (\d+)',line)}
            assert series['running-req']==1 and series['transfer-req']==12
            counters.append({'D_UTC':stamp.isoformat(),'running_requests':1,'transfer_requests':12,
                             'P_arrival_lower_bound_minus_D_activity_upper_bound_s':
                                 round(p_first_lower-p_upper,3)})
    assert len(counters)>=2
    return counters


def build(files,paths,decoder):
    probe=json.loads(files['probe'])
    records=[json.loads(row) for row in files['tcp_window'].splitlines()]
    sockets,metric_window,n=first_socket_windows(probe,records)
    sse,skew=client_first_time(files,paths,decoder)
    assert abs(sse[0]-metric_window[0])<6 and abs(sse[1]-metric_window[1])<6
    evidence={'epoch':1317,'TCP_samples':n,'initial_pending_Router_owned_connections_by_role':probe['connection_counts'],
       'original_client_complete_temp_records':20,'original_client_first_SSE_P_clock_bracket_UTC':list(map(utc,sse)),
       'observed_single_router_histogram_increment_P_UTC':list(map(utc,metric_window)),
       'role_arrival_observations':sockets,
       'scope':'Read-only socket receive counters on the original Router for the isolated 101-to-102 metric window. '
               'The then-active TCP sockets carry no request ID here. A counter increase measures receipt of '
               'TCP payload on one leg; HTTP framing, body completion and scheduler readiness require their own evidence.',
       'formal_phase_accepted':False}
    for role,row in sockets.items():
        for window in row['persistent_socket_first_nonzero_windows']:
            a,b=map(wall,window['bounds_P_UTC'])
            window['TCP_receive_minus_client_first_SSE_s']=[round(a-sse[1],3),round(b-sse[0],3)]
    p,d=[sockets[role]['persistent_socket_first_nonzero_windows'] for role in ('prefill','decode')]
    if len(p)==len(d)==1:
        pa,pb=map(wall,p[0]['bounds_P_UTC'])
        da,db=map(wall,d[0]['bounds_P_UTC'])
        evidence['decode_receive_minus_prefill_receive_s']=[round(da-pb,3),round(db-pa,3)]
        evidence['decode_payload_already_observed_before_prefill_arrival_lower_bound_s']=round(pa-db,3)
        evidence['decode_scheduler_reports_before_first_observed_prefill_payload']=decode_activity_before_p(files['decode_log'],pa,skew)
    return evidence


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--evidence-root',type=Path,required=True)
    cli.add_argument('--decoder',type=Path,required=True)
    args=cli.parse_args()
    paths={name:args.evidence_root/file for name,file in FILES.items()}
    files={name:path.read_bytes() for name,path in paths.items()}
    result=build(files,paths,load_decoder(args.decoder))
    result['private_evidence_sha256']={name:digest(raw) for name,raw in files.items()}
    result['analysis_source_sha256']=digest(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
