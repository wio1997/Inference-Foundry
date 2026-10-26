"""Local independent failure diagnosis; never creates a performance certificate."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CLIENT = '20261017T2000-sglang-formal-C64-L04-v2'
BEFORE = '20261017T1992-sglang-L04-C32-native-v2'
AFTER = '20261017T2002-sglang-L04-C64-native-v2'
STDERR = '20261017T2014-sglang-C64-archived-stderr-review-v1'
SOURCE = '20261017T2018-sglang-C64-native-cache-source-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native(run, node):
    response = json.loads((ROOT / 'runs' / run / f'node-{node}.json').read_text())
    assert response['rc'] == 0
    return json.loads(response['stdout'])


def meta_counters(document):
    files = [file for snapshot in document['native_SDK_small_file_snapshots'] for file in snapshot.get('files', []) if file['path'].endswith('/mmc-meta-audit.log')]
    assert len(files) == 1
    file = files[0]
    assert hashlib.sha256(file['text'].encode()).hexdigest() == file['sha256']
    lines = [line for line in file['text'].splitlines() if ' keys=' in line and ' evict=' in line]
    assert lines
    last = lines[-1]
    fields = {name: int(value) for name, value in re.findall(r'\b(keys|evict|alloc_req|alloc_success|alloc_fail|get_req|get_success|get_fail|get_not_found|batch_get_req|batch_get_fail|remove_req)=(\d+)', last)}
    assert set(fields) == {'keys', 'evict', 'alloc_req', 'alloc_success', 'alloc_fail', 'get_req', 'get_success', 'get_fail', 'get_not_found', 'batch_get_req', 'batch_get_fail', 'remove_req'}
    used, capacity = re.search(r'\bdram_used=(\d+)/(\d+)', last).groups()
    fields.update(dram_used=int(used), dram_capacity=int(capacity))
    return {'counters': fields, 'source_sha256': file['sha256']}


def main():
    report_path = ROOT / 'runs' / CLIENT / 'formal-result.remote.json'
    assert sha(report_path) == '381deb1c28dd386b2390a775d7341fd84f791b4978baba922e49dfc4d7dbf0ec'
    report = json.loads(report_path.read_text())
    assert report['baseline_accepted'] is False and len(report['formal']['rows']) == 256
    failed = {row['request_id'] for row in report['formal']['rows'] if not row['result']['logical_success']}
    assert len(failed) == 46
    P, D, prefetch = set(), set(), set()
    rank_counts = Counter()
    waiting_seconds, bootstrap_seconds = [], []
    native_stderr = {}
    for node in (34, 35, 36, 41):
        output = native(STDERR, node)
        text = output['archived_stderr_text']
        assert output['utf8_lossless'] and hashlib.sha256(text.encode()).hexdigest() == output['raw_sha256']
        assert len(text.encode()) == output['raw_bytes']
        count = Counter()
        for line in text.splitlines():
            if 'server_args=' in line:
                continue
            match = re.search(r"(?:req.rid|decode_req.req.rid)='([^']+)'", line)
            if match and 'Prefill bootstrap failed' in line:
                P.add(match[1].removesuffix('.P'))
                count['P_bootstrap_failure_rank_lines'] += 1
            if match and 'Decode transfer failed' in line:
                D.add(match[1].removesuffix('.D'))
                rank_counts[match[1].removesuffix('.D')] += 1
                count['D_transfer_failure_rank_lines'] += 1
            timeout = re.search(r'timed out after ([0-9.]+)s in (KVPoll\.\w+)', line)
            if timeout:
                assert timeout[2] in ('KVPoll.WaitingForInput', 'KVPoll.Bootstrapping')
                values = waiting_seconds if timeout[2] == 'KVPoll.WaitingForInput' else bootstrap_seconds
                values.append(float(timeout[1]))
            match = re.search(r'Prefetch operation (\S+) failed to retrieve page ', line)
            if match:
                prefetch.add(match[1].removesuffix('.P'))
                count['prefetch_missing_page_rank_lines'] += 1
        native_stderr[str(node)] = {'raw_bytes': output['raw_bytes'], 'raw_sha256': output['raw_sha256'], 'compressed_archive_sha256': output['source_archive_sha256'], 'counts': dict(count)}
    assert D == failed and P <= failed and len(P) == 19
    assert len(waiting_seconds) == 92 and set(rank_counts.values()) == {2}
    assert len(prefetch) == 5
    assert all(300 <= value < 302 for value in waiting_seconds + bootstrap_seconds)
    before, after = {}, {}
    audit_delta, SDMA = {}, {}
    for node in (34, 35, 36, 41):
        before[node], after[node] = native(BEFORE, node), native(AFTER, node)
        for output in (before[node], after[node]):
            assert all(log['full_archive_complete'] for log in output['logs'].values())
        counts = []
        for output in (before[node], after[node]):
            counts.append(Counter(row['event'] for file in output['persistent_audit'].values() for row in file['rows']))
        delta = {event: counts[1][event] - counts[0][event] for event in counts[0] | counts[1]}
        assert all(value >= 0 for value in delta.values())
        assert not any('error' in event.lower() for event in counts[1])
        audit_delta[str(node)] = delta
        old, new = (output['logs']['stdout']['native_sdma'] for output in (before[node], after[node]))
        SDMA[str(node)] = {'copy_descriptor_bytes_delta': new['copy_descriptor_bytes'] - old['copy_descriptor_bytes'], 'copy_calls_delta': new['copy_calls'] - old['copy_calls'], 'cumulative_native_status_values': new['copy_status_values'], 'physical_link_bytes': None}
        assert new['copy_status_values'] == ([0] if node in (34, 35) else [])
    old_meta, new_meta = meta_counters(before[34]), meta_counters(after[34])
    assert old_meta['counters']['dram_capacity'] == new_meta['counters']['dram_capacity'] == 568 * 1024**3
    changes = {name: new_meta['counters'][name] - old_meta['counters'][name] for name in old_meta['counters'] if name not in ('keys', 'dram_used', 'dram_capacity')}
    assert changes['evict'] == 159537 and changes['get_not_found'] == 7816 and changes['alloc_fail'] == changes['get_fail'] == 0
    sources = native(SOURCE, 34)
    source_hashes = {}
    for file in sources['files']:
        assert hashlib.sha256(file['source'].encode()).hexdigest() == file['sha256']
        source_hashes[file['relative']] = file['sha256']
    pipeline = next(file['source'] for file in sources['files'] if file['relative'].endswith('buffer_mode/pipeline.py'))
    assert 'self._cache.storage_existence_cache.covers_all(' in pipeline
    output = {'project_id': 'glm53', 'run_id': CLIENT, 'measurement_valid': False, 'performance_certificate_created': False, 'analyzer_sha256': sha(Path(__file__)), 'raw_client_report_sha256': sha(report_path), 'all_native_captures_complete': True, 'failed_requests': 46, 'D_native_300s_waiting_timeouts_join_all_failed_client_requests': True, 'D_failure_rank_lines': 92, 'D_ranks_per_failed_request': 2, 'P_native_300s_bootstrap_timeout_distinct_requests': len(P), 'P_timeout_ids_are_subset_of_D_and_client_failure_ids': True, 'native_prefetch_missing_page_distinct_requests': len(prefetch), 'native_stderr': native_stderr, 'Meta_same_service_period_before': old_meta, 'Meta_same_service_period_after': new_meta, 'Meta_period_deltas': changes, 'native_SDMA_and_typed_audit_interval': {'typed_event_deltas': audit_delta, 'native_copy_deltas': SDMA, 'scope': 'Warm plus formal interval; submitted/completed sends and native statuses are evidence of successful actual calls, not proof that all logical requests were completed or that every cached prefix survived.'}, 'actual_SGLang_source_hashes': source_hashes, 'source_hypothesis': 'Buffer-mode confirmed local storage dedup beliefs can outlive independent external eviction. Skipped duplicate backups do not check external existence or propagate HBM-use recency through the native backend. PublicSDK Meta existence lookups promote LRU; actual isolated native CPU counterfactual is pending.', 'causal_limits': ['Eviction/missing keys and timeouts are both proved; the full causal chain and sufficient fix are not proved.', 'Do not attribute C64 failure to L04 without an unoptimized matched C64 counterfactual.', 'Do not waive the failed request/cache/count/SDMA gates or extend client/engine timeouts to manufacture a passing measurement.', 'The before/after Meta interval includes64prefix warmup and formal traffic;159537evictions are not all formal prefix evictions.', 'No cross-host wall-clock ordering assumption. Exact request IDs joined privately; no prompt text/output/IDs are published.'], 'source_evidence_files': {str(path.relative_to(ROOT)): sha(path) for path in [ROOT / 'runs' / STDERR / 'receipt.json', ROOT / 'runs' / SOURCE / 'receipt.json', ROOT / 'results/sglang-L04-C64-v2-failed-attempt-v1.json']}}
    path = ROOT / 'results/sglang-C64-native-failure-diagnosis-v1.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'result': str(path.relative_to(ROOT)), 'sha256': sha(path), 'failed_joined': 46, 'P_timeout_joined': 19, 'prefetch_missing_operations': 5, 'Meta_deltas': changes}))


if __name__ == '__main__':
    main()
