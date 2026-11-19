"""Trace the pinned Ascend KV completion paths and check active cold histograms."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from audit_cold_transfer_histograms_g1467 import BASELINE_SHA, METRICS_SHA, histogram, read_receipt


PREFILL = '7eb0a7f514599fc8f48f651aaf936d0e1874e0184d79eba134142960663a727d'
DECODE = '739607cf9c17912bff7c8c5c5cc825d74b00be7d0211c7aaf336e95167c29425'
HALFWAY = 'e4baa25d7c0e3615d68fcadad0dc19c0c800886b06531efa964c3dce2697ce4c'
SOURCES = 'G1481_ASCEND_SEND_RECV_SOURCE_JSON'
METRICS = 'G1467_COLD_TRANSFER_RAW'


def find_order(source, function, *snippets):
    rows = source['functions'][function]['rows']
    selected = []
    for text in snippets:
        line, = (number for number, raw in rows if text in raw)
        selected.append(line)
    assert selected == sorted(set(selected))
    return {'function': function, 'line_numbers_in_order': selected}


def status_path(prefill, decode):
    assert prefill['role'] == 'prefill' and decode['role'] == 'decode'
    assert prefill['epoch'] == decode['epoch'] == 1317
    files = prefill['sources']
    assert {k: v['sha256'] for k, v in files.items()} == {
        k: v['sha256'] for k, v in decode['sources'].items()}
    assert prefill['startup'] == decode['startup']
    assert decode['startup'] == {
        'disable_overlap_schedule': ['False'], 'disaggregation_decode_polling_interval': ['1'],
        'disaggregation_transfer_backend': ["'ascend'"]}
    ascend = files['disaggregation/ascend/conn.py']
    mooncake = files['disaggregation/mooncake/conn.py']
    common = files['disaggregation/common/conn.py']
    prefill_scheduler = files['disaggregation/prefill.py']
    decode_scheduler = files['disaggregation/decode.py']
    for role in ('Sender', 'Receiver', 'Manager'):
        assert ascend['class_bases']['AscendKV'+role] == ['MooncakeKV'+role]
    assert mooncake['class_bases']['MooncakeKVSender'] == ['CommonKVSender']
    assert mooncake['class_bases']['MooncakeKVReceiver'] == ['CommonKVReceiver']
    assert 'CommonKVManager' in mooncake['class_bases']['MooncakeKVManager']
    assert all(not ascend['members']['AscendKV'+r] for r in ('Sender','Receiver'))
    path = {
        'enqueue': [
            find_order(mooncake, 'MooncakeKVSender.send', 'super().send('),
            find_order(common, 'CommonKVSender.send', 'self.kv_mgr.add_transfer_request('),
            find_order(mooncake, 'MooncakeKVManager.add_transfer_request', 'self.transfer_queues[shard_idx].put(')],
        'background_transfer_and_status': [
            find_order(mooncake, 'MooncakeKVManager.__init__', 'self.transfer_worker,'),
            find_order(mooncake, 'MooncakeKVManager.transfer_worker', 'kv_chunk = queue.get()',
                       'ret = self.send_kvcache(', 'self.conclude_transfer('),
            find_order(common, 'CommonKVManager.conclude_transfer', 'self.update_status(bootstrap_room, status)',
                       'self.send_kv_status_message(')],
        'decode_state_and_ready_queue': [
            find_order(mooncake, 'MooncakeKVManager.start_decode_thread', 'parsed = self.parse_kv_status_message(msg)',
                       'self.apply_prefill_status(', 'target=decode_thread'),
            find_order(common, 'CommonKVManager.apply_prefill_status',
                       'self.update_status(bootstrap_room, KVPoll.Success)'),
            find_order(common, 'CommonKVReceiver._poll', 'status = self.kv_mgr.check_status(self.bootstrap_room)'),
            find_order(decode_scheduler, 'DecodeTransferQueue.pop_transferred_with_outcomes',
                       'polls = self._poll_with_metadata_gate()', 'elif poll == KVPoll.Success:',
                       'self._commit_transfer_to_req(decode_req)', 'transferred_reqs.append(decode_req.req)'),
            find_order(decode_scheduler, 'SchedulerDisaggregationDecodeMixin.process_decode_queue',
                        'self.disagg_decode_transfer_queue.pop_transferred()', 'self.waiting_queue.extend(transferred_reqs)')],
        'prefill_scheduler_cleanup_and_histogram': [
            find_order(prefill_scheduler, 'SchedulerDisaggregationPrefillMixin.process_disagg_prefill_transfer_queue',
                       'polls = poll_and_all_reduce_attn_cp_tp_group(', 'elif poll == KVPoll.Success:',
                       'req.time_stats.set_prefill_kv_transfer_finish_time()',
                       'req.time_stats.compute_and_observe_kv_transfer_metrics(', 'self.output_streamer.stream_output(')],
    }
    return {'source_sha256_by_module': {name: f['sha256'] for name, f in files.items()},
            'startup_resolved': decode['startup'], 'source_sites':path,
            'scope': 'The background worker can publish transfer success to Decode and local state. '
                     'Decode uses its local status and metadata checks to make the request eligible. '
                     'Prefill scheduler polling observes local completion and feeds Prefill cleanup, '
                     'metrics and its own output stream. Client response routing requires a separate join.'}


def halfway_histogram(earlier, later):
    for row in (earlier, later):
        assert row['epoch'] == 1317 and row['runner_PID'] == 3090471 and row['current_phase'] == 'cold'
        assert row['baseline_sha256'] == BASELINE_SHA
    assert earlier['run_id'] == later['run_id']
    h = [histogram(row['roles']['prefill'], 'kv_transfer_latency_ms') for row in (earlier,later)]
    assert [item['count_delta'] for item in h] == [4,16]
    assert all(item['delta_cumulative_buckets']['5000.0'] == 0 for item in h)
    additional = (h[1]['sum_delta']-h[0]['sum_delta'])/1000
    assert additional > 0
    return {'source_before_first_cold_sha256': BASELINE_SHA, 'earlier_sample_D_UTC': earlier['at_UTC'],
            'later_sample_D_UTC':later['at_UTC'], 'prefill_rank':0,
            'first_4_sender_observations_mean_s': h[0]['mean_in_metric_units']/1000,
            'first_16_sender_observations_mean_s': h[1]['mean_in_metric_units']/1000,
            'observations_5_through_16_mean_s': additional/12,
            'first_16_above_max_finite_histogram_bound_s': 5,
            'scope': 'Histogram spans last-chunk queue entry through Prefill scheduler completion observation. '
                     'The actual worker, transfer and notification intervals require timestamped events.'}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('prefill_source','decode_source','first_four','first_sixteen'):
        cli.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args = cli.parse_args()
    p,p_observer = read_receipt(args.prefill_source,PREFILL,SOURCES)
    d,d_observer = read_receipt(args.decode_source,DECODE,SOURCES)
    first,metric_observer = read_receipt(args.first_four,METRICS_SHA,METRICS)
    later,second_observer = read_receipt(args.first_sixteen,HALFWAY,METRICS)
    assert p_observer == d_observer and metric_observer == second_observer
    report = {'epoch':1317,'runtime_source_boundaries':status_path(p,d),
             'actual_cold_prefill_metrics':halfway_histogram(first,later),
             'formal_result_accepted':False,
             'private_receipts_sha256': {'prefill':PREFILL,'decode':DECODE,'early':METRICS_SHA,'halfway':HALFWAY},
             'analysis_source_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
