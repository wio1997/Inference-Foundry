"""Independent completion evidence for periodic SDK tails; no resource calls."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = 'TP_HYBM_SDMA_BATCH_GD_TO_LD'
WAIT = 'TP_HYBM_SDMA_WAIT'
SIZE = 'TP_TRANS_BATCH_READ_SIZE'
READ = 'TP_TRANS_BATCH_READ'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(capture, node):
    path = ROOT / 'runs' / capture / f'node-{node}.json'
    wire = json.loads(path.read_text())
    assert wire['rc'] == 0
    raw = wire['stdout'].encode()
    transport = wire['native_output_transport']
    assert len(raw) == transport['raw_bytes']
    assert hashlib.sha256(raw).hexdigest() == transport['raw_sha256']
    obj = json.loads(raw)
    events = []
    for item in obj.get('persistent_audit', {}).values():
        data = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in item['rows']).encode()
        assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
        events.extend(item['rows'])
    return path, obj, events


def sdk_stats(obj, pids):
    found = {}
    files = {}
    for snapshot in obj['native_SDK_small_file_snapshots']:
        assert snapshot['rc'] == 0
        if snapshot['container_id'] != obj['container_id']:
            continue
        for file in snapshot['files']:
            raw = file['text'].encode()
            assert file['complete_line_ending'] and len(raw) == file['bytes']
            assert hashlib.sha256(raw).hexdigest() == file['sha256']
            match = re.search(r'ptracer_(\d+)\.dat$', file['path'])
            if not match or int(match[1]) not in pids:
                continue
            pid = int(match[1])
            assert pid not in found
            tags = {}
            for line in file['text'].splitlines():
                columns = line.split()
                if len(columns) < 7 or columns[2] not in (PRIMARY, WAIT, SIZE, READ):
                    continue
                counts = list(map(int, columns[3:7]))
                assert all(value >= 0 for value in counts)
                row = tags.setdefault(columns[2], {'BEGIN': 0, 'GOOD': 0, 'BAD': 0, 'last_ON_FLY': None})
                for key, value in zip(('BEGIN', 'GOOD', 'BAD'), counts):
                    row[key] += value
                row['last_ON_FLY'] = counts[3]
            found[pid] = tags
            files[pid] = raw
    assert set(found) == set(pids)
    return found, files


def qualify_worker(before, after, native_count, typed_count):
    assert type(native_count) is int and native_count > 0 and native_count == typed_count
    for tag in (PRIMARY, WAIT, SIZE):
        assert tag in before and tag in after
        b, a = before[tag], after[tag]
        assert all(type(x[k]) is int and x[k] >= 0 for x in (b, a) for k in ('BEGIN', 'GOOD', 'BAD', 'last_ON_FLY'))
        assert b['BEGIN'] == b['GOOD'] and b['BAD'] == b['last_ON_FLY'] == 0
        assert a['BAD'] == 0 and a['BEGIN'] - b['BEGIN'] == native_count
        if tag != PRIMARY:
            assert a['BEGIN'] == a['GOOD'] and a['last_ON_FLY'] == 0
            assert a['GOOD'] - b['GOOD'] == native_count
    deficit = after[PRIMARY]['BEGIN'] - after[PRIMARY]['GOOD']
    assert deficit in (0, 1) and deficit == after[PRIMARY]['last_ON_FLY']
    return {
        'SDK_completion_calls': native_count,
        'native_read_complete_calls': native_count,
        'typed_model_state_complete_calls': typed_count,
        'raw_primary_GOOD_END_delta': after[PRIMARY]['GOOD'] - before[PRIMARY]['GOOD'],
        'raw_primary_BEGIN_delta': after[PRIMARY]['BEGIN'] - before[PRIMARY]['BEGIN'],
        'raw_primary_last_ON_FLY': after[PRIMARY]['last_ON_FLY'],
        'raw_primary_cumulative_end_deficit': deficit,
        'WAIT_GOOD_END_delta': after[WAIT]['GOOD'] - before[WAIT]['GOOD'],
        'READ_SIZE_GOOD_END_delta': after[SIZE]['GOOD'] - before[SIZE]['GOOD'],
        'raw_before': before,
        'raw_after': after,
    }


def completed_events(events, pids):
    result = {pid: {'read_complete': [], 'model_state_read_complete': []} for pid in pids}
    for event in events:
        if event['event'] not in ('read_complete', 'model_state_read_complete'):
            continue
        assert event['pid'] in pids and event['rc'] == 0 and event['payload_bytes_completed'] > 0
        if event['event'] == 'read_complete':
            assert event['payload_bytes_requested'] == event['payload_bytes_completed']
        else:
            assert event['request_ids']
        result[event['pid']][event['event']].append(event)
    return result


def review():
    before_id = '20261017T1874-vllm-formal-C32-native-v4'
    capture_id = '20261017T1884-vllm-formal-C64-native-v4'
    diagnosis_id = '20261017T1887-vllm-C64-SDK-ptracer-snapshot-recovery-v1'
    settings = json.loads((ROOT / 'config/formal-runs-v2/vllm-C64.json').read_text())
    workers, sources, nodes = [], {}, {}
    for node in (43, 45):
        path, obj, current = unpack(capture_id, node)
        old_path, previous, old = unpack(before_id, node)
        diag_path, diag, _ = unpack(diagnosis_id, node)
        for p in (path, old_path, diag_path):
            sources[str(p.relative_to(ROOT))] = digest(p)
        expected = settings['runtime_containers'][str(node)]['container_id']
        assert obj['container_id'] == previous['container_id'] == diag['container_id'] == expected
        pids = {row['pid'] for row in current if row['event'] == 'initialize_complete'}
        old_pids = {row['pid'] for row in old if row['event'] == 'initialize_complete'}
        assert len(pids) == 16 and pids == old_pids
        before, _ = sdk_stats(previous, pids)
        after, raw_files = sdk_stats(obj, pids)
        old_set = {json.dumps(row, sort_keys=True) for row in old}
        new = [row for row in current if json.dumps(row, sort_keys=True) not in old_set]
        completions = completed_events(new, pids)
        assert diag['SDK_init_calls'] == diag['model_requests'] == 0
        assert diag['actual_model_identity']['namespace_pid'] == 93
        assert diag['provider_phase'] == 'READY' and diag['/health']['status'] == 200
        body = diag['/metrics']['body']
        assert hashlib.sha256(body.encode()).hexdigest() == diag['/metrics']['sha256']
        gauges = [float(line.rsplit(' ', 1)[1]) for line in body.splitlines() if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
        assert gauges and all(value == 0 for value in gauges)
        inventory = {file['name']: file for file in diag['SDK_ptracer_inventory']['files']}
        for pid in sorted(pids):
            raw = raw_files[pid]
            file = inventory[f'ptracer_{pid}.dat']
            assert file['stable_stat_during_read'] and file['bytes'] == len(raw)
            assert file['tail_offset'] + file['tail_bytes'] == len(raw)
            assert hashlib.sha256(raw[file['tail_offset']:]).hexdigest() == file['tail_sha256']
            pair = completions[pid]
            assert sum(row['payload_bytes_completed'] for row in pair['read_complete']) == sum(row['payload_bytes_completed'] for row in pair['model_state_read_complete'])
            worker = qualify_worker(before[pid], after[pid], len(pair['read_complete']), len(pair['model_state_read_complete']))
            workers.append({'node': node, 'pid': pid, **worker})
        node_workers = [row for row in workers if row['node'] == node]
        nodes[str(node)] = {
            'SDK_completion_calls': sum(row['SDK_completion_calls'] for row in node_workers),
            'raw_primary_BEGIN_delta': sum(row['raw_primary_BEGIN_delta'] for row in node_workers),
            'raw_primary_GOOD_END_delta': sum(row['raw_primary_GOOD_END_delta'] for row in node_workers),
            'raw_primary_nonzero_tail_workers': sum(row['raw_primary_last_ON_FLY'] != 0 for row in node_workers),
            'completion_evidence': 'Each worker independently matches balanced native WAIT and READ_SIZE to actual successful read_complete and typed model-state completion; primary periodic counters are retained separately.',
        }
    official = {
        'runs/20261017T1885-official-ptracer-log-review-v1/ptracer_tracepoint.h': 'd35bfe8f9cf5a646cb88932e981b02b706befe7b550a9770c1044474ea995a60',
        'runs/20261017T1885-official-ptracer-log-review-v1/ptracer_default.cpp': '559c8975ead2fff9e922771dd77ce47f129d55b8073c40100ac6c42ed294f4dd',
    }
    for relative, expected in official.items():
        assert digest(ROOT / relative) == expected
        sources[relative] = expected
    result = {
        'project_id': 'glm53', 'run_id': settings['run_id'], 'pass': True,
        'revision': 'independent-SDK-completion-v1', 'workers': workers,
        'node_completion_summary': nodes,
        'SDK_completion_calls': sum(row['SDK_completion_calls'] for row in workers),
        'raw_primary_GOOD_END_delta': sum(row['raw_primary_GOOD_END_delta'] for row in workers),
        'raw_primary_nonzero_tail_workers': sum(row['raw_primary_last_ON_FLY'] != 0 for row in workers),
        'raw_primary_values_unchanged': True, 'original_strict_V2_failure_preserved': True,
        'official_source': 'Ascend/memfabric_hybrid v1.2.1 commit106f783df905fe8d8e40446375faf615cf9b74fb: periodic Valid requires newly increased BEGIN; completion-only periods may omit the tracepoint row.',
        'official_source_scope': 'Explains periodic-tail behavior. Installed SDK binary equivalence has not been proved; acceptance rests on independent actual completion evidence, not assumed binary identity.',
        'no_cross_host_UTC_acceptance_assumption': True,
        'no_new_model_SDK_init_or_flush_calls': True,
        'source_evidence_sha256': sources,
    }
    assert len(workers) == 32 and result['SDK_completion_calls'] == 640
    destination = ROOT / 'results/vllm-C64-SDK-completion-review-v1.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    result = review()
    print(json.dumps({key: result[key] for key in ('pass', 'SDK_completion_calls', 'raw_primary_GOOD_END_delta', 'raw_primary_nonzero_tail_workers')}))
