"""Read complete temporary AISBench rows without accepting a formal result."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time


MAX_TEMP_BYTES = 64 * 1024 * 1024


def snapshot_prefix(path):
    with path.open('rb') as stream:
        stream.seek(0, 2)
        boundary = stream.tell()
        if boundary > MAX_TEMP_BYTES:
            raise ValueError('Temporary output exceeds the bounded observer read size')
        stream.seek(0)
        raw = stream.read(boundary)
    if len(raw) != boundary:
        raise ValueError('Temporary output shrank while reading')
    newline = raw.rfind(b'\n') + 1
    return raw[:newline], boundary - newline


def read_status(path, expected):
    for attempt in range(3):
        raw = path.read_bytes()
        if len(raw) > 16384:
            raise ValueError('Unexpected AISBench status size')
        try:
            records = json.loads(raw)
        except json.JSONDecodeError:
            records = []
        if records == []:
            if attempt < 2:
                time.sleep(0.05)
            continue
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError('Expected one AISBench task record in the status list')
        record = records[0]
        total, completed = record['total_count'], record['finish_count']
        assert type(total) is int and type(completed) is int
        assert 0 <= completed <= total == expected
        return {'total_count': total, 'finish_count': completed,
                'status': record['status'], 'sha256': sha256(raw).hexdigest()}
    return None


def audit(dataset_raw, expected_sha, temp, status, input_tokens, output_tokens):
    if sha256(dataset_raw).hexdigest() != expected_sha:
        raise ValueError('Frozen input SHA256 mismatch')
    dataset = [json.loads(raw) for raw in dataset_raw.splitlines() if raw.strip()]
    before = read_status(status, len(dataset))
    complete_bytes, ignored_tail_bytes = snapshot_prefix(temp)
    outcomes = Counter()
    seen_data, seen_ids = set(), set()
    for raw in complete_bytes.splitlines():
        row = json.loads(raw)
        data_id = row['data_id']
        assert type(data_id) is int and 0 <= data_id < len(dataset)
        assert data_id not in seen_data and row['id'] not in seen_ids
        assert row['input'] == dataset[data_id]['question']
        seen_data.add(data_id)
        seen_ids.add(row['id'])
        if row['success'] is True:
            assert (row['input_tokens'], row['output_tokens']) == (input_tokens, output_tokens)
            outcomes['recorded_success_with_exact_usage'] += 1
        elif row['success'] is False:
            outcomes['recorded_failure'] += 1
        else:
            raise ValueError('Unexpected temporary row success state')
    after = read_status(status, len(dataset))
    if before is not None and after is not None:
        assert before['finish_count'] <= after['finish_count']
    return {
        'dataset_sha256': expected_sha, 'configured_requests': len(dataset),
        'verified_complete_temp_rows': len(seen_data),
        'temp_rows_by_outcome': dict(outcomes),
        'complete_temp_prefix_bytes': len(complete_bytes), 'complete_temp_prefix_sha256': sha256(complete_bytes).hexdigest(),
        'ignored_nonterminated_temp_bytes': ignored_tail_bytes,
        'status_before_and_after': [before, after],
        'all_complete_temp_rows_match_frozen_inputs': True,
        'formal_terminal_acceptance': False,
        'scope': 'Live bounded read. Temporary JSONL, status and terminal prediction/performance files have distinct roles.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--dataset-sha256', required=True)
    parser.add_argument('--temp', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--input-tokens', type=int, required=True)
    parser.add_argument('--output-tokens', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset.read_bytes(), args.dataset_sha256,
                           args.temp, args.status, args.input_tokens, args.output_tokens), sort_keys=True))


if __name__ == '__main__':
    main()
