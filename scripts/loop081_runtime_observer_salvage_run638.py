#!/usr/bin/env python3
"""Admit each Run638 arm separately after fixed-W0 cross-arm gate rejects it."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from scripts.loop081_runtime_observer_validate_run638 import arm, read, need
from scripts import loop081_runtime_observer_validate_run638 as validator

ROOT = Path('/data/wio/Inference_Foundry')
BASE = ROOT / 'evidence/20260928_loop081_bound/run638/live'
OUT = ROOT / 'evidence/20260928_loop081_bound/run638/arm_scoped_recovery.json'
PAIRS = (
    ('cycle_begin','target_before'),
    ('target_before','target_forward_return'),
    ('target_forward_return','target_logits_return'),
    ('target_logits_return','target_after'),
    ('target_after','acceptance_after'),
    ('acceptance_after','state_advance_after'),
    ('proposer_before','proposer_after'),
    ('proposer_after','draft_commit_after'),
    ('draft_commit_after','serve_stage_after'),
    ('serve_stage_after','serve_park_after'),
    ('cycle_begin','serve_park_after'),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_cleanup():
    status = dict(line.split('=',1) for line in
                  (BASE/'cleanup_status.txt').read_text().splitlines())
    need(set(status) == {'run_exit','stop_exit','stop_verify_exit','restore_exit',
                         'source_sha_exit','source_compare_exit','script_sha_exit',
                         'script_compare_exit','final_exit'},'cleanup fields')
    need(all(status[key]=='0' for key in status
             if key not in ('run_exit','final_exit')) and
         status['run_exit']=='1' and status['final_exit']=='1',
         'expected cross-arm admission rejection and clean recovery')
    need((BASE/'source_before.sha256').read_bytes() ==
         (BASE/'source_after.sha256').read_bytes(), 'source restoration')
    need((BASE/'scripts_before.sha256').read_bytes() ==
         (BASE/'scripts_after.sha256').read_bytes(), 'script restoration')
    return status


def main():
    cleanup = check_cleanup()
    rows={name:arm(BASE/name,f'LOOP081-RUN638-{name.upper()}',name=='on')
          for name in ('off_a','on','off_b')}
    histories = {name:{str(k):v for k,v in row['cohorts'].items()}
                 for name,row in rows.items()}
    cohort_cycles={name:{c:x['cycles'] for c,x in h.items()}
                   for name,h in histories.items()}
    same_text={name:sum(a==b for a,b in zip(
                    rows['off_a']['client_text_sha256_by_dataset_index'],
                    rows[name]['client_text_sha256_by_dataset_index']))
               for name in ('on','off_b')}
    same_cohorts={name:histories[name]==histories['off_a']
                  for name in ('on','off_b')}
    need(not same_cohorts['on'] and same_text['on'] < 48,
         'cross-arm mismatch no longer present; use primary admission')
    distributions={f'{a}_to_{b}':[] for a,b in PAIRS}
    paired_other=[]
    host_wait=[]
    packet_shas={}
    for cohort in range(5,9):
        for rank in range(8):
            path=BASE/'on/packet'/f'rank{rank}_cohort{cohort}.json'
            packet=read(path)
            packet_shas[str(path)]=sha(path)
            for cycle in (63,64,65):
                events={e['label']:e['elapsed_ms_from_first_same_stream']
                        for e in packet['events'] if e['cycle']==cycle and
                        e['stream']=='current'}
                for a,b in PAIRS:
                    value=events[b]-events[a]
                    need(value>=0,'current stream phase reversed')
                    distributions[f'{a}_to_{b}'].append(value)
                paired_other.append(
                    (events['serve_park_after']-events['cycle_begin'])
                    -(events['target_forward_return']-events['target_before'])
                    -(events['proposer_after']-events['proposer_before']))
                need(paired_other[-1]>=0,'paired other interval negative')
                wait=packet['existing_host_waits'][str(cycle)][0]
                host_wait.append((wait['end_ns']-wait['start_ns'])/1e6)
    def stats(x):
        return {'n':len(x),'min':min(x),'median':statistics.median(x),
                'max':max(x)}
    result={
        'status':'arm_scoped_current_packet_pass_cross_arm_fixed_W0_rejected',
        'scope':'Each arm has independent 48+48/c12 all8 Runtime/client admission; ON current/copy Event packet is valid only for its own observed W0. The raw product_output_sha256 field hashes Runtime bulk _cohort_output.token_ids, not final Scheduler/API token IDs. Client text/usage is independently valid, but final API token-ID parity is not re-proved. Cross-arm observer-effect and timing transfer are invalid because effective trajectories and client output differ.',
        'cleanup':cleanup,
        'acquisition_validator_sha256':sha(ROOT/'evidence/20260928_loop081_bound/run638/acquisition_validator.py'),
        'posthoc_validator_sha256':sha(Path(validator.__file__)),
        'cleanup_status_sha256':sha(BASE/'cleanup_status.txt'),
        'source_before_sha256':sha(BASE/'source_before.sha256'),
        'source_after_sha256':sha(BASE/'source_after.sha256'),
        'arm_raw_sha256':{name:row['raw_sha256'] for name,row in rows.items()},
        'cohort_cycles':cohort_cycles,
        'same_cohort_effective_work_as_off_a':same_cohorts,
        'same_client_text_count_of_48_as_off_a':same_text,
        'client_product_measured_duration_s':{
            name:row['client_product_measured_duration_s'] for name,row in rows.items()},
        'on_current_stream_phase_ms':{k:stats(v) for k,v in distributions.items()},
        'on_paired_other_current_stream_ms':stats(paired_other),
        'on_existing_host_count_copy_wait_ms':stats(host_wait),
        'on_packet_raw_sha256':packet_shas,
        'whole_product_framework_only_tps_bound':None,
        'run99_timing_transfer':False,
    }
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],
                      'same_text_on':same_text['on'],
                      'on_packets':len(packet_shas)}))


if __name__=='__main__':
    main()
