"""Verify printed CPU scope clocks against actual CANN host start metadata."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HISTORY_SHA = '669cdc4ab0f540a56b235f0568d0fec2637ffa1708f72ecb6d4a89bc3ca0eefd'
PREFILL_SHA = '496c429745ce596c401f9603eed9811583ee21a27ed3818daee9b8abd0592575'
DECODE_SHA = '65d21e3e538c02accdbce5ae830268c294b2b5dfb3dc54e2d9edf0f3fc4bb5f5'
SOURCE_MARKER = 'G1486_LIVE_FULL_CLOCK_AUTHORITY_JSON'
HIST_MARKER = 'G1486_HISTORICAL_PROFILE_RAW_CLOCK_JSON'


def receipt(path, digest, marker):
    raw=path.read_bytes()
    assert sha256(raw).hexdigest() == digest
    packet=json.loads(raw)
    assert packet['outcome']=='finished' and packet['exitstatus']==0
    content,=(line.removeprefix(marker+' ') for line in packet['output'].splitlines()
              if line.startswith(marker+' '))
    return json.loads(content)


def source_contract(p,d):
    assert p['role']=='prefill' and d['role']=='decode' and p['epoch']==d['epoch']==1317
    checked = {
        'analysis/_profiler_config.py': (
            'config_offset = start_log.get("Host", "clock_monotonic_raw")',
            'config_start_cnt = start_log.get("Host", "cntvct")',
            'timestamp = int((syscnt - self._start_cnt) * ratio) + self._time_offset',
            'self._localtime_diff = CANNFileParser(profiler_path).get_localtime_diff()',
            'return int(monotonic_time + self._localtime_diff)',
        ),
        'analysis/prof_bean/_torch_op_bean.py': (
            'self._start_ns = profiler_config.get_local_time(profiler_config.get_timestamp_from_syscnt(start_syscnt))',
            'self._end_ns = profiler_config.get_local_time(profiler_config.get_timestamp_from_syscnt(end_syscnt))',
        ),
        'analysis/prof_common_func/_path_manager.py': (
            'start_info_path = os.path.join(cann_path, "host", "start_info")',
        ),
        'analysis/prof_common_func/_constant.py': (
            'CANN_BEGIN_TIME = "collectionTimeBegin"',
            'CANN_BEGIN_MONOTONIC = "clockMonotonicRaw"',
            'FWK_END_TIME = "collectionTimeEnd"',
            'FWK_END_MONOTONIC = "MonotonicTimeEnd"',
        ),
        'analysis/prof_parse/_cann_file_parser.py': (
            'localtime_diff = convert_us2ns(',
            'info_json.get(Constant.CANN_BEGIN_TIME, 0)',
            ') - int(info_json.get(Constant.CANN_BEGIN_MONOTONIC, 0))',
        ),
    }
    out={}
    for path,fragments in checked.items():
        left,right=(row['files'][path] for row in (p,d))
        assert left==right
        lines={text.strip():line for line,text in left['lines']}
        out[path]={'sha256':left['sha256'],'anchor_lines':[lines[part] for part in fragments]}
    return out


def check_capture(row):
    assert row['rank']==0
    start=row['raw_session_start_files']['host/start_info']
    sample,=start['timing_text']
    fields=json.loads(sample)
    cann_begin_epoch_ns=int(fields['collectionTimeBegin'])*1000
    cann_begin_monotonic_ns=int(fields['clockMonotonicRaw'])
    framework=row['profiler_end_info']
    end_epoch_ns=framework['collectionTimeEnd']
    end_monotonic_ns=framework['MonotonicTimeEnd']
    host_offset_ns=cann_begin_epoch_ns-cann_begin_monotonic_ns
    fallback_offset_ns=end_epoch_ns-end_monotonic_ns
    scopes,start_ns,end_ns,min_type,max_type=row['all_cpu_scope_count_min_max_type']
    markers,first_marker,last_marker=row['scheduler_scope_count_min_max']
    assert scopes>0 and markers==4
    assert cann_begin_epoch_ns<=start_ns<=first_marker<last_marker<=end_ns<=end_epoch_ns
    assert cann_begin_monotonic_ns < end_monotonic_ns
    assert row['profiler_start_info']['syscnt_enable']
    assert min_type==max_type=='text'
    return {'physical_decode_rank':0,'CPU_scope_count':scopes,'scheduler_scope_count':markers,
            'DB_startNs_storage_type':min_type,'CANN_host_start_info_sha256':start['sha256'],
            'profiler_info_sha256':row['metadata_sha256'],
            'CANN_host_clock_offset_ns':host_offset_ns,
            'framework_end_time_fallback_minus_CANN_start_offset_ns':fallback_offset_ns-host_offset_ns,
            'first_CPU_scope_after_CANN_start_ns':start_ns-cann_begin_epoch_ns,
            'framework_capture_end_after_last_CPU_scope_ns':end_epoch_ns-end_ns,
            'all_CPU_scope_times_inside_local_wall_clock_capture':True}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ('history','prefill','decode','current_p_d_clocks','join_fixture'):
        cli.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    args=cli.parse_args()
    p=receipt(args.prefill,PREFILL_SHA,SOURCE_MARKER)
    d=receipt(args.decode,DECODE_SHA,SOURCE_MARKER)
    history=receipt(args.history,HISTORY_SHA,HIST_MARKER)
    assert history['role']=='decode' and history['read_only']
    assert set(history['historical_rank_zero'])=={'G896','G981'}
    raw=args.current_p_d_clocks.read_bytes()
    clocks=json.loads(raw)
    offset=clocks['prefill_minus_decode_realtime_ns']
    assert len(offset)==2 and 19_000_000_000 < offset[0] < offset[1] < 21_000_000_000
    assert all(len(role['samples'])==12 for role in clocks['roles'].values())
    fixture_bytes=args.join_fixture.read_bytes()
    fixture=json.loads(fixture_bytes)
    assert fixture['scope_storage_types_tested']==['TEXT']
    assert fixture['joined_frozen_requests_with_synthetic_scope_times']==16
    assert 'invalid_fractional_timestamp_text' in fixture['rejected_controls']
    join_source=Path(__file__).with_name('join_PD_room_events_g1485.py')
    report={'epoch':1317,'active_image_exporter_source_pins':source_contract(p,d),
            'fixed_joiner_source_sha256':sha256(join_source.read_bytes()).hexdigest(),
            'actual_SQLite_schema_regression_fixture_sha256':sha256(fixture_bytes).hexdigest(),
            'text_timestamp_join_fixture':fixture,
            'historical_CPU_exports':{name:check_capture(row) for name,row in history['historical_rank_zero'].items()},
            'contemporaneous_P_minus_D_real_clock_offset_ns':offset,
            'current_clock_sample_sha256':sha256(raw).hexdigest(),
            'actual_new_P_D_profile_clock_aligned':False,
            'private_receipts_sha256':{'P_exporter':PREFILL_SHA,'D_exporter':DECODE_SHA,'historical_capture':HISTORY_SHA},
            'analysis_source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'The exporter translates host CPU event clocks using CANN host metadata or a framework fallback. '
                    'Each translated CPU scope remains in its host wall-clock domain. Obtain P/D samples around '
                    'the actual matched windows before cross-host time subtraction. Device and link clocks need independent evidence.'}
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
