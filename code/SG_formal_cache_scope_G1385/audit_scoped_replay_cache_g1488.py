"""Audit Prefill token sources while preserving the Decode reporting boundary."""

import argparse
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re


METRIC='sglang:prefill_effective_tokens_total'
MODES=('input','device_hit','host_hit','storage_hit')
RANKS={str(i) for i in range(8)}
DATA_SHA='2484713fafb09065e6426717d1f07fa26cd3e7de4d771e734c60ec2511b94ffe'
GATE_SHA='81b675d4f4ae9929564b68c45586868da26d0ce8883488249771e1b2db653d5d'
LIVE_PINS={
    'metrics': ('35f0912f143f136cbd1b8620f19a4fa2b404e5a6c3b7c6502be25d3b69c17adc','G1488_COLD_ALL_RANK_PREFIX_COUNTERS'),
    'prefill': ('58f3ebc93e1414142b2cd9ee4f4cfa90b04f70a1691b14e5dfa05ef6de3842df','G1488_LIVE_CACHE_USAGE_AUTHORITY_JSON'),
    'decode': ('fa8c60c0e92ca6a487a62cf6fca40492a0669f6c187dee01141000da71b3ad2d','G1488_LIVE_CACHE_USAGE_AUTHORITY_JSON'),
    'client': ('d9c6209de351f396d8f63463e3f679f7b78d0c7d849df9c31556a604dafbc179','G1488_ORIGINAL_TEMP_CLIENT_CACHE_SCHEMA'),
}


def open_receipt(path,key):
    raw=path.read_bytes()
    pin,marker=LIVE_PINS[key]
    assert sha256(raw).hexdigest()==pin
    packet=json.loads(raw)
    assert packet['exitstatus']==0 and packet['outcome']=='finished'
    payload,=(json.loads(line.removeprefix(marker+' ')) for line in packet['output'].splitlines()
             if line.startswith(marker+' '))
    assert payload['epoch']==1317
    return payload


def rank_values(series,role):
    modes={name:{} for name in MODES}
    for key,value in series.items():
        if not key.startswith(METRIC+'{'):
            continue
        tags=dict(re.findall(r'(\w+)="([^"]*)"',key))
        if tags.get('engine_type')!=role or tags.get('model_name')!='ds41flash':
            continue
        mode,rank=tags['mode'],tags['tp_rank']
        number=Decimal(str(value))
        assert number.is_finite() and number==number.to_integral_value() and number>=0
        assert rank not in modes[mode]
        modes[mode][rank]=int(number)
    assert all(set(ranks)==RANKS for ranks in modes.values())
    return modes


def delta(roles):
    assert set(roles)=={'prefill','decode'}
    result={}
    for role, samples in roles.items():
        a,b=(rank_values(samples[k],role) for k in ('previous','current'))
        result[role]={mode:{rank:b[mode][rank]-a[mode][rank] for rank in sorted(RANKS)} for mode in MODES}
    assert all(v>=0 for r in result.values() for m in r.values() for v in m.values())
    return result


def p_local_outcome(raw):
    by_mode={mode:ranks['0'] for mode,ranks in raw['prefill'].items()}
    extra_publishers=sorted({rank for mode in raw['prefill'].values() for rank,count in mode.items()
                             if rank!='0' and count})
    return {'prefill_rank0_counter_deltas':by_mode,'other_P_rank_counter_movement':extra_publishers,
            'decode_local_prefix_hits_verified':False,
            'scope':'The Prefill rank-zero metric is a Prefill-local effective-token view. Other rank series are '
                    'retained separately. Decode-labeled Prefill series cannot establish local Decode reuse.'}


def source_scope(prefill,decode,client):
    assert prefill['role']=='prefill' and decode['role']=='decode'
    assert prefill['runtime_settings']['enable_cache_report']==decode['runtime_settings']['enable_cache_report']==['True']
    assert decode['runtime_settings']['disaggregation_decode_enable_radix_cache']==['True']
    bindings={
      'disaggregation/utils.py':('self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens',
          'self.cached_tokens[req.metadata_buffer_index][1] = req.cached_tokens_device'),
      'disaggregation/decode.py':('decode_req.req.cached_tokens = cached_tokens[0].item()',
          'decode_req.req.already_computed = decode_req.req.cached_tokens',
          'decode_req.req.cached_tokens_device = cached_tokens[1].item()'),
      'disaggregation/decode_schedule_batch_mixin.py':('delta = max(0, pre_len - req.already_computed)',
          'req.cached_tokens += delta','req.cached_tokens_device += delta'),
      'observability/metrics_collector.py':('name="sglang:cached_tokens_total",',
          'self.cached_tokens_total.labels(**source_labels).inc(value)'),
    }
    source={}
    for file,snippets in bindings.items():
        a,b=(r['source_files'][file] for r in (prefill,decode))
        assert a==b
        selected={text.strip():number for number,text in a['rows']}
        source[file]={'sha256':a['sha256'],'lines':[selected[text] for text in snippets]}

    observer_gauges={}
    for role, current in (('prefill',prefill),('decode',decode)):
        families=current['metric_families']
        observer_gauges[role]={
           'request_tokens_cached_total_exposed':'sglang:cached_tokens_total' in families,
           'tokenizer_request_counter_exposed':'sglang:num_requests_total' in families,
           'scheduler_cache_hit_rate_exposed':'sglang:cache_hit_rate' in families,
        }
    ob=client['observation']
    assert client['runner_PID']==3090471 and ob['verified_complete_temp_rows']==23
    assert not ob['retained_cache_metadata_paths'] and not ob['formal_terminal_acceptance']
    return {'source_runtime_checksum_match':source,'server_cache_reporting_enabled':True,
            'observed_metrics':observer_gauges,'current_temp_client_rows_inspected':23,
            'usage_cache_breakdown_preserved_in_tmp_rows':False,
            'scope':'Decode initializes its reported total and device count from the Prefill transfer metadata. '
                    'The shown Decode prebuilt path adds only positive local-prefix excess over the already '
                    'counted window. Neither the combined total nor the device tally isolates the Decode local match.'}


def bundle_observation(folder):
    info=json.loads((folder/'preservation_inventory.json').read_bytes())
    assert info['dataset_sha256']==DATA_SHA and info['gate_sha256']==GATE_SHA
    assert info['phase'] in ('cold','hot')
    report=json.loads((folder/'audit.json').read_bytes())
    assert report['fully_validated'] and report['requests']==report['successful_requests']==32
    snapshots={}
    for key,name in (('previous','before'),('current','after')):
        raw=(folder/(name+'_server_snapshot.json')).read_bytes()
        snapshot=json.loads(raw)
        assert not snapshot.get('_capture_provenance',{}).get('posthoc',False)
        for role in ('prefill','decode'):
            metric=snapshot[role]['metrics']
            assert metric['status']==200
            lines=(item.rsplit(None,1) for item in metric['body'].splitlines() if item.startswith(METRIC+'{'))
            series={}
            for stage,value in lines:
                assert stage not in series
                series[stage]=value
            snapshots.setdefault(role,{})[key]=series
    native=delta(snapshots)
    summary=p_local_outcome(native)
    accepted_cache=json.loads((folder/'cache_audit.json').read_bytes())['prefill']
    assert accepted_cache['verified'] is True
    assert summary['prefill_rank0_counter_deltas']['device_hit']==accepted_cache['delta']
    if info['phase']=='cold':
        assert accepted_cache['delta']==0
    return {'phase':info['phase'],'raw_role_deltas_by_rank':native,**summary,
            'verified_P_rank0_device_hit_delta':accepted_cache['delta']}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for key in LIVE_PINS:
        cli.add_argument('--'+key,type=Path)
    cli.add_argument('--verified-phase-bundle',type=Path)
    args=cli.parse_args()
    if args.verified_phase_bundle:
        result=bundle_observation(args.verified_phase_bundle)
    else:
        payloads={key:open_receipt(getattr(args,key),key) for key in LIVE_PINS}
        metrics=payloads.pop('metrics')
        assert metrics['current_phase']=='cold' and metrics['runner_PID']==3090471
        changes=delta(metrics['roles'])
        result={'epoch':1317,'observed_at_D_UTC':metrics['at_UTC'],
                'live_cold_role_deltas_by_rank':changes,'P_local_cache':p_local_outcome(changes),
                'D_usage_and_export_scope':source_scope(**payloads),'cold_or_replay_formally_accepted':False,
                'private_inputs_sha256':{key:pin for key,(pin,_) in LIVE_PINS.items()},
                'analysis_source_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
