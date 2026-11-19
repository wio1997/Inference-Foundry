"""Audit G1317 live KV histogram deltas and the actual Ascend timing boundary."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re


METRICS_SHA = '548b84092ee6cfd2d1d6986bb87ef4b4ec165bcd88b8bc055baf7f2f36d7d427'
PREFILL_SOURCE_SHA = '4cdc58d3735d6f796a9b55c3adc3721aeaa98a0c6d026a657bcdfc1dc56813f5'
DECODE_SOURCE_SHA = 'ab90cd299e058f8c6f73da0d30279f5e9fcdc40cf5bbe9bd9360ba13782ce5a0'
BASELINE_SHA = '3639fe8a522b719d9e36fa0145d38ce762cfad6b3c7b01a6f1f21d64d4e2bf87'
METRICS = ('kv_transfer_latency_ms', 'kv_transfer_total_mb', 'kv_transfer_speed_gb_s',
           'kv_transfer_bootstrap_ms', 'kv_transfer_alloc_ms')
QUEUES = ('num_running_reqs', 'num_queue_reqs', 'num_prefill_bootstrap_queue_reqs',
          'num_prefill_inflight_queue_reqs', 'num_decode_prealloc_queue_reqs',
          'num_decode_transfer_queue_reqs', 'num_decode_host_receive_queue_reqs')


def read_receipt(path, digest, marker):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == digest
    receipt = json.loads(raw)
    assert receipt['outcome'] == 'finished' and receipt['exitstatus'] == 0
    line, = (row.removeprefix(marker + ' ') for row in receipt['output'].splitlines()
             if row.startswith(marker + ' '))
    return json.loads(line), receipt['script_sha256']


def verify_source(prefill, decode):
    assert prefill['epoch'] == decode['epoch'] == 1317
    assert prefill['role'] == 'prefill' and decode['role'] == 'decode'
    files = {
        'disaggregation/ascend/conn.py': ('class AscendKVManager(MooncakeKVManager):',
                                          'class AscendKVSender(MooncakeKVSender):'),
        'disaggregation/mooncake/conn.py': ('class MooncakeKVManager(StagingManagerMixin, CommonKVManager):',
                                            'class MooncakeKVSender(CommonKVSender):'),
        'disaggregation/common/conn.py': ('reports_transfer_latency: bool = False',
                                          'and self.kv_mgr.reports_transfer_latency'),
        'disaggregation/base/conn.py': ('transfer_latency_s: Optional[float] = None',),
        'disaggregation/prefill.py': ('req.time_stats.set_prefill_transfer_queue_entry_time()',
                                       'req.time_stats.set_completion_time()',
                                       'req.disagg_kv_sender.get_transfer_metric()'),
        'observability/req_time_stats.py': ('self.completion_time - self.prefill_transfer_queue_entry_time',),
        'observability/metrics_collector.py': ('self._log_histogram(self.kv_transfer_latency_ms, latency_ms)',),
    }
    result = {}
    for name, clauses in files.items():
        a, b = (item['sources'][name] for item in (prefill, decode))
        assert a['sha256'] == b['sha256']
        selected = {text.strip(): number for number, text in a['selected']}
        result[name] = {'source_sha256': a['sha256'], 'verified_lines': [selected[clause] for clause in clauses]}
    for name in ('disaggregation/ascend/conn.py', 'disaggregation/mooncake/conn.py'):
        assert not any('reports_transfer_latency' in text for _, text in prefill['sources'][name]['selected'])
    return result


def series(payload, name):
    prefix = f'sglang:{name}{{'
    return {tag: value for tag, value in payload.items() if tag.startswith(prefix)}


def unique(payload, name):
    values = series(payload, name)
    key, = values
    assert 'tp_rank="0"' in key
    return key, values[key]


def histogram(payload, metric):
    previous, current = (payload[key] for key in ('previous', 'current'))
    if not series(current, metric + '_count'):
        assert not series(previous, metric + '_count')
        return None
    delta = {}
    for name in ('count', 'sum'):
        key, value = unique(current, metric + '_' + name)
        delta[name] = value - previous[key]
        assert math.isfinite(delta[name]) and delta[name] >= 0
    count = delta['count']
    assert int(count) == count
    now, then = (series(data, metric + '_bucket') for data in (current, previous))
    assert now.keys() == then.keys()
    buckets = {}
    for tag, n in now.items():
        limit = re.search(r'\ble="([^"]+)"', tag)[1]
        increase = n - then[tag]
        assert int(increase) == increase and 0 <= increase <= count
        buckets[limit] = int(increase)
    ordered = sorted(buckets, key=float)
    assert ordered[-1] == '+Inf' and buckets['+Inf'] == count
    assert all(buckets[a] <= buckets[b] for a, b in zip(ordered, ordered[1:]))
    return {'count_delta': int(count), 'sum_delta': delta['sum'],
            'mean_in_metric_units': delta['sum'] / count if count else None,
            'delta_cumulative_buckets': buckets}


def summarize(measurement, prefill, decode):
    assert measurement['epoch'] == 1317 and measurement['runner_PID'] == 3090471
    assert measurement['run_id'] == 'sg_pd_g1317_aisbench_32k_g1324'
    assert measurement['baseline_sha256'] == BASELINE_SHA and measurement['current_phase'] == 'cold'
    definitions = verify_source(prefill, decode)
    roles = {}
    for role, payload in measurement['roles'].items():
        histograms = {name: result for name in METRICS if (result := histogram(payload, name)) is not None}
        queues = {name: unique(payload['current'], name)[1] for name in QUEUES}
        roles[role] = {'rank': 0, 'histogram_deltas': histograms, 'queue_gauges': queues}
    assert set(roles) == {'prefill', 'decode'}
    p = roles['prefill']['histogram_deltas']
    assert p['kv_transfer_latency_ms']['count_delta'] == p['kv_transfer_total_mb']['count_delta'] == 4
    assert p['kv_transfer_latency_ms']['delta_cumulative_buckets']['5000.0'] == 0
    assert roles['decode']['histogram_deltas']['kv_transfer_alloc_ms']['count_delta'] == 20
    return {
        'epoch': 1317, 'cold_phase_live_snapshot_at_UTC': measurement['at_UTC'],
        'before_first_cold_metrics_snapshot_sha256': BASELINE_SHA, 'roles': roles,
        'actual_native_source_chain_on_both_roles': definitions,
        'ascend_transfer_time_scope': 'Prefill last-chunk queue entry to sender-side completion observation.',
        'transfer_size_scope': 'Sender-derived size counters reported in MiB, including tracked state bytes.',
        'effective_link_bandwidth_or_device_transfer_time_verified': False,
        'actual_cold_or_replay_score_accepted': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metric_receipt', type=Path)
    parser.add_argument('prefill_source_receipt', type=Path)
    parser.add_argument('decode_source_receipt', type=Path)
    args = parser.parse_args()
    measure, measure_observer = read_receipt(args.metric_receipt, METRICS_SHA, 'G1467_COLD_TRANSFER_RAW')
    prefill, p_observer = read_receipt(args.prefill_source_receipt, PREFILL_SOURCE_SHA, 'G1467_SG_TRANSFER_SOURCE')
    decode, d_observer = read_receipt(args.decode_source_receipt, DECODE_SOURCE_SHA, 'G1467_SG_TRANSFER_SOURCE')
    assert p_observer == d_observer
    result = summarize(measure, prefill, decode)
    result['input_receipt_sha256'] = {'metrics': METRICS_SHA, 'prefill_source': PREFILL_SOURCE_SHA,
                                      'decode_source': DECODE_SOURCE_SHA}
    result['observer_source_sha256'] = {'metrics': measure_observer, 'both_sources': p_observer}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
