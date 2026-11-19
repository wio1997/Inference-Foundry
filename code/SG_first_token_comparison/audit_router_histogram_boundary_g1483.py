"""Align the active Router's next latency observation with original 32K raw client clocks."""

import argparse
from base64 import b64decode
from datetime import datetime
from gzip import decompress
import json
from pathlib import Path

from compare_first_wave_g1479 import DATA_SHA, clocks, digest, load_decoder


FILES = {
    'probe': 'G1483_router_metric_probe_2203.json',
    'watch': 'G1483_router_timing_boundary/P_watch_events.jsonl',
    'client_before': 'G1483_initial_client_exact_timing_2204.json',
    'client_at_increment': 'G1483_D_on_router_hist_increment_2204.json',
    'client_after': 'G1483_final_client_exact_timing_2204.json',
    'client_db': 'G1483_router_timing_boundary/final_19_timing.db',
    'old_db': 'G1483_router_timing_boundary/initial_18_timing.db',
    'client_clock_before': 'G1483_D_client_clock_before_2204.json',
    'client_clock_after': 'G1483_D_client_clock_after_2204.json',
    'pd_clock_before': 'G1483_PD_clock_before.json',
    'pd_clock_after': 'G1483_PD_clock_after.json',
}


def read_packet(raw, marker):
    packet = json.loads(raw)
    assert packet['exitstatus'] == 0 and packet['outcome'] == 'finished'
    prefix = marker + ' '
    content, = (line.removeprefix(prefix) for line in packet['output'].splitlines() if line.startswith(prefix))
    return json.loads(content)


def parse_client(raw, count):
    packet = read_packet(raw,'G1479_COLD_RAW_TIMING_AND_PROMPT_IDENTITY')
    observation = packet['observation']
    assert packet['epoch'] == 1317 and packet['runner_PID']==3090471 and packet['aisbench_exec_PID']==3090587
    assert observation['dataset_sha256'] == DATA_SHA and observation['all_complete_temp_rows_match_frozen_inputs']
    assert not observation['formal_terminal_acceptance']
    assert observation['configured_requests'] == 32 and observation['verified_complete_temp_rows'] == count
    assert observation['temp_rows_by_outcome'] == {'recorded_success_with_exact_usage':count}
    return packet


def bind_db(path, sample, decoder):
    bundle = sample['observation']['timing_bundle']
    raw = path.read_bytes()
    assert raw == decompress(b64decode(bundle['database_gzip_b64'],validate=True))
    assert digest(raw) == bundle['database_sha256'] and len(raw) == bundle['database_bytes']
    return clocks(path, {row['data_id']:row['reference'] for row in bundle['timing_refs']}, decoder)


def wall(stamp):
    return datetime.fromisoformat(stamp).timestamp()


def observed_timing(files, paths, decoder):
    probe = read_packet(files['probe'],'G1483_ROUTER_EXISTING_METRICS_JSON')
    assert probe['epoch'] == 1317 and probe['all_http_reads_targeted_original_listener_owners']
    metrics, = [r for r in probe['responses'] if r['status']==200]
    assert metrics['listener']=='auxiliary'
    families = sorted({line.split('{',1)[0].split()[0] for line in metrics['metric_series']})
    assert all(term not in name for name in families for term in ('prefill','decode','first_token','first_sse'))
    durations = [name for name in families if 'duration' in name]
    assert all(name.startswith(('smg_router_request_duration_seconds_', 'smg_http_request_duration_seconds_'))
               for name in durations)
    armed,event = [json.loads(line) for line in files['watch'].splitlines()]
    assert armed['event']=='watch_ready' and event['event']=='one_completion_histogram_increment'
    assert armed['router_listener_PID']==event['router_listener_PID']==probe['router_PID'] and event['still_live']
    previous,new = event['previous_snapshot'],event['current_snapshot']
    assert previous['router_count']==previous['http_count']==100
    assert new['router_count']==new['http_count']==101
    assert 0 < wall(new['sample_to_P_UTC'])-wall(previous['sample_from_P_UTC']) < 5

    before = parse_client(files['client_before'],18)
    on = parse_client(files['client_at_increment'],18)
    after = parse_client(files['client_after'],19)
    old_o,new_o = before['observation'],on['observation']
    assert old_o['complete_temp_prefix_sha256']==new_o['complete_temp_prefix_sha256']
    assert old_o['timing_bundle']['database_sha256']==new_o['timing_bundle']['database_sha256']
    old = bind_db(paths['old_db'],before,decoder)
    clocks_after = bind_db(paths['client_db'],after,decoder)
    new_id, = clocks_after.keys() - old.keys()
    assert all(clocks_after[key]==time for key,time in old.items())
    assert len(clocks_after)==19
    submit,first,*_,last = clocks_after[new_id]

    d_samples = [read_packet(files['client_clock_'+part],'G1476_D_CLIENT_CLOCK_JSON') for part in ('before','after')]
    assert all(r['same_runner_live'] and r['client_container_matches_host_time_namespace']
               and r['perf_counter_impl']=='clock_gettime(CLOCK_MONOTONIC)' for r in d_samples)
    d_stamps = [[r['wall_seconds']-r['monotonic_bracket_s'][1],r['wall_seconds']-r['monotonic_bracket_s'][0]]
                for r in d_samples]
    d_min,d_max = min(x[0] for x in d_stamps),max(x[1] for x in d_stamps)
    assert d_max-d_min<0.001
    pd = [json.loads(files['pd_clock_'+part]) for part in ('before','after')]
    assert all(len(role['samples'])==12 for sample in pd for role in sample['roles'].values())
    skew = [sample['prefill_minus_decode_realtime_ns'] for sample in pd]
    p_min,p_max = min(x[0] for x in skew)/1e9,max(x[1] for x in skew)/1e9
    assert 19 < p_min < p_max < 21
    start = wall(previous['sample_from_P_UTC'])-p_max
    end = wall(new['sample_to_P_UTC'])-p_min
    low,high = first+d_min-end,first+d_max-start
    on_read = wall(on['at_UTC'])
    assert end<=on_read and first+d_max<=on_read<last+d_min
    assert abs(low)<5 and abs(high)<5
    ttft = first-submit
    r_delta,h_delta=(new[key+'_sum']-previous[key+'_sum'] for key in ('router','http'))
    assert abs(r_delta-ttft)<1 and abs(h_delta-ttft)<1
    return {'epoch':1317,'raw_ready_rows_before_P_update':18,'raw_ready_rows_after_P_update':18,
            'raw_ready_rows_in_later_client_snapshot':19,'matched_original_client_timing_rows':len(clocks_after),
            'all_18_old_timing_arrays_unchanged':True,
            'actual_router_http_duration_metric_names':durations,
            'separate_exposed_prefill_or_decode_leg_metric_observed':False,
            'isolated_metric_count_delta_each':1,'client_submit_to_first_SSE_s':ttft,
            'client_first_to_last_SSE_s':last-first,'router_duration_sum_delta_s':r_delta,
            'HTTP_middleware_duration_sum_delta_s':h_delta,
            'client_first_SSE_minus_router_histogram_update_bracket_s':[round(low,3),round(high,3)],
            'client_end_seen_after_histogram_update_lower_bound_s':round(last+d_min-end,3),
            'same_request_join_basis':'One isolated metric increment, one added client timing row, same staged cold run and a calibrated host time window.',
            'scope':'One observed POST completions window. Both exposed durations track the first-response boundary within the measured window. '
                    'The series carry no per-leg timing or shared request identifier. Prefill-specific waiting requires independent observations.',
            'formal_phase_score_accepted':False}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--evidence-root',type=Path,required=True)
    cli.add_argument('--decoder',type=Path,required=True)
    args=cli.parse_args()
    paths={name:args.evidence_root/rel for name,rel in FILES.items()}
    files={name:p.read_bytes() for name,p in paths.items()}
    report=observed_timing(files,paths,load_decoder(args.decoder))
    report['private_receipt_sha256']={name:digest(raw) for name,raw in files.items()}
    report['source_sha256']=digest(Path(__file__).read_bytes())
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
