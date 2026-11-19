"""Audit the fixed SGLang chunk compute budget against original cold logs."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import textwrap
from types import SimpleNamespace


SOURCE_SHA = '033e6514847afbf0bf104c8b729c3171a6e3e68f18cfa617126a2baec57455d1'
REPORT_SHA = '748002b9b4cb13f017080e320e19c5b20cbbbc706b745c25f3dcdc209a17ba11'
POLICY = 'managers/schedule_policy.py'
SCHEDULER = 'managers/scheduler.py'
FIELDS = ('new-seq', 'new-token', 'cached-token', 'queue-req')
PREFILL_BATCH = re.compile(r'^\[([\d-]+ [\d:]+) TP0 EP0\] Prefill batch,')


def read_receipt(path, expected, marker):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == expected
    payload = json.loads(raw)
    assert payload['outcome'] == 'finished' and payload['exitstatus'] == 0
    line, = (line.removeprefix(marker + ' ') for line in payload['output'].splitlines()
             if line.startswith(marker + ' '))
    record = json.loads(line)
    assert record['epoch'] == 1317 and record['role'] == 'prefill'
    return record


def exercise_budget(policy):
    status = SimpleNamespace(NO_TOKEN='NO_TOKEN', OTHER='OTHER', CONTINUE='CONTINUE')
    namespace = {'AddReqResult': status, 'is_cp_sharded_padded': lambda: False}
    texts = [policy['functions'][fn][0]['text'] for fn in ('_update_prefill_budget', 'budget_state')]
    code = '\n\n'.join(textwrap.dedent(text) for text in texts)
    exec(compile(code, '<fixed-prefill-budget>', 'exec'), namespace)

    cases = ((8192, True, True, 'OTHER', 0), (4096, False, True, 'CONTINUE', 4096),
             (8192, False, False, 'NO_TOKEN', 0), (4097, False, True, 'CONTINUE', 3968))
    results = []
    for extend, ongoing, memory_available, expected, wanted_remaining in cases:
        adder = SimpleNamespace(
            ceil_paged_tokens=lambda n: (n + 127) // 128 * 128,
            exact_chunk_fill=False, dllm_config=None, rem_chunk_tokens=8192,
            rem_input_tokens=16384, logical_page_size=128, per_req_token_overhead=128,
            log_hit_tokens=0, log_input_tokens=0, log_replay_tokens=0,
            memory_budget=SimpleNamespace(reserve=lambda *args, **kwargs: None,
                                          has_capacity=lambda: memory_available,
                                          state_slots_exhausted=lambda: False),
        )
        admission_prefix = 8192 if ongoing else 0
        namespace['_update_prefill_budget'](adder, admission_prefix, extend, 0, False,
                                            is_chunked_continuation=ongoing)
        assert adder.rem_input_tokens > 0 and adder.rem_chunk_tokens == wanted_remaining
        assert adder.log_hit_tokens == 0 and adder.log_input_tokens == extend
        decision = namespace['budget_state'](adder)
        assert decision == expected
        results.append({'admitted_extend_rows': extend, 'chunked_continuation': ongoing,
                        'remaining_chunk_rows': adder.rem_chunk_tokens,
                        'remaining_prefill_rows': adder.rem_input_tokens,
                        'allocator_has_capacity_fixture': memory_available, 'budget_state': decision})
    return results


def analyze(source, reports):
    settings = source['startup_settings']
    assert settings['chunked_prefill_size'] == ['8192'] and settings['max_prefill_tokens'] == ['16384']
    assert settings['max_running_requests'] == ['16'] and settings['prefill_max_requests'] == ['None']
    assert settings['enable_dynamic_chunking'] == settings['enable_mixed_chunk'] == ['False']
    assert settings['page_size'] == ['128'] and settings['schedule_policy'] == ["'fcfs'"]
    policy, scheduler = (source['source_summary'][name] for name in (POLICY, SCHEDULER))
    schedule = scheduler['functions']['_get_new_batch_prefill_raw'][0]['text']
    assert schedule.index('adder.add_chunked_req(') < schedule.index('adder.add_one_req(')
    assert 'get_schedule().prefill_max_requests' in schedule
    add = policy['functions']['add_one_req'][0]['text']
    assert 'self.prefill_max_requests) is not None' in add
    assert 'self.rem_chunk_tokens -= compute_charge' in policy['functions']['_update_prefill_budget'][0]['text']

    histograms = {key: Counter() for key in FIELDS}
    stamps = []
    for row in reports['reports']['engine']['selected_lines']:
        match = PREFILL_BATCH.match(row)
        assert match
        stamps.append(match[1])
        fields = {k: int(v) for k, v in re.findall(r'#([a-z-]+): (\d+)', row)}
        for key in FIELDS:
            histograms[key][fields[key]] += 1
    assert len(stamps) == 36 and stamps == sorted(set(stamps))
    assert histograms['new-seq'] == {1: 36}
    assert histograms['new-token'] == {8192: 36} and histograms['cached-token'] == {0: 36}
    assert min(histograms['queue-req']) > 0
    return {
        'epoch': 1317, 'rank': 0, 'startup_settings': settings,
        'actual_cold_log_report_count': len(stamps), 'first_and_last_prefill_host_log_time': [stamps[0], stamps[-1]],
        'observed_report_field_histograms': {key: dict(values) for key, values in histograms.items()},
        'fixed_source_budget_CPU_cases': exercise_budget(policy),
        'source_artifact_sha256': {POLICY: policy['sha256'], SCHEDULER: scheduler['sha256']},
        'source_receipt_sha256': SOURCE_SHA, 'log_receipt_sha256': REPORT_SHA,
        'scope': ('One fully charged 8192-row extend exhausts the compute chunk budget even with allocator '
                  'capacity. Native memory fit remains an independent admission requirement. '
                  'The observed cold log window is not a complete run or a chunk-size A/B comparison.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_receipt', type=Path)
    parser.add_argument('prefill_log_receipt', type=Path)
    args = parser.parse_args()
    source = read_receipt(args.source_receipt, SOURCE_SHA, 'G1472_PREFILL_ADMISSION_SOURCES')
    reports = read_receipt(args.prefill_log_receipt, REPORT_SHA, 'G1468_COLD_LOGS')
    print(json.dumps(analyze(source, reports), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
