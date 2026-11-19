#!/usr/bin/env python3
"""Check sanitized incident receipt invariants, not remote runtime or request behavior."""
import copy
import json
import re
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'evidence/cold_cwd_incident_035.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def validate_incident(doc):
    cold, claim = doc['cold'], doc['claim']
    require(claim['state'] == 'possibly_submitted_even_if_controller_interrupted' and claim['replay_allowed'] is False,
            'old claim must remain possibly submitted and not replayed')
    require(cold['exit_code'] == 124 and cold['hot_submitted'] is False and cold['performance_validated'] is False,
            'invalid cold terminal interpretation')
    require(cold['detail_count_at_terminal_review'] == 0 and cold['failure_is_executor_cwd'] is True
            and cold['model_health_proven'] is False, 'invalid failure scope')
    def timestamp(text):
        value = datetime.fromisoformat(text)
        require(value.tzinfo is not None, 'timezone missing')
        return value
    start = timestamp(cold['controller_started_at_utc'])
    claimed = timestamp(claim['claimed_at_utc'])
    terminal = timestamp(cold['terminal_at_utc'])
    require(start < claimed < terminal, 'invalid same-controller chronology')
    source = doc['source_order']
    require(source['static_order_only'] is True and source['config_dump_line'] == source['traceback_dump_line']
            and source['config_dump_line'] < source['subprocess_launch_line'], 'invalid static execution order')
    router = doc['router_observation']
    require(router['matched_method'] == 'POST' and router['matched_api'] == 'completions endpoint'
            and router['matched_post_lines'] == 0 and router['coverage_all_network_paths'] is False
            and router['cross_host_clocks_calibrated'] is False, 'router scope overstated')
    require(timestamp(router['at_or_after_utc']) <= timestamp(router['last_selected_log_utc'])
            <= timestamp(router['observed_at_utc']), 'invalid router-local chronology')
    smoke = doc['writable_cwd_smoke']
    require(smoke['config_dump_exit_code'] == 0 and smoke['cwd_in_unique_writable_mount'] is True
            and smoke['full_aisbench_or_inference_verified'] is False
            and 'manual transcription' in smoke['type'], 'smoke scope overstated')
    require(doc['candidate_fix']['live_end_to_end_accepted'] is False, 'candidate not yet accepted')
    for name, value in doc['evidence_sha256'].items():
        require(valid_sha(value), 'invalid evidence SHA: ' + name)
    for group, names in ((claim, ('config_sha256', 'dataset_sha256', 'old_first_ledger_sha256')),
                         (source, ('aisbench_local_py_sha256',)),
                         (smoke, ('probe_source_sha256', 'synthetic_config_sha256')),
                         (doc['candidate_fix'], ('local_fixed_runner_sha256', 'prior_staged_runner_sha256', 'prior_staged_controller_sha256'))):
        for name in names:
            require(valid_sha(group[name]), 'invalid source SHA: ' + name)
    return True


class IncidentTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(RECEIPT.read_text(encoding='utf-8'))

    def test_actual_sanitized_receipt(self):
        self.assertTrue(validate_incident(self.data))
        self.assertEqual(self.data['source_order']['config_dump_line'], 249)
        self.assertEqual(self.data['source_order']['subprocess_launch_line'], 260)
        self.assertEqual(self.data['writable_cwd_smoke']['synthetic_config_bytes'], 21)

    def test_unsafe_or_invalid_interpretations_are_rejected(self):
        for case in ('replay', 'claim_state', 'accepted', 'hot', 'coverage', 'smoke', 'hash', 'order'):
            data = copy.deepcopy(self.data)
            if case == 'replay': data['claim']['replay_allowed'] = True
            elif case == 'claim_state': data['claim']['state'] = 'not_submitted'
            elif case == 'accepted': data['cold']['performance_validated'] = True
            elif case == 'hot': data['cold']['hot_submitted'] = True
            elif case == 'coverage': data['router_observation']['coverage_all_network_paths'] = True
            elif case == 'smoke': data['writable_cwd_smoke']['full_aisbench_or_inference_verified'] = True
            elif case == 'hash': data['evidence_sha256']['cold_cli_log'] = 'invalid'
            else: data['source_order']['config_dump_line'] = 261
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_incident(data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
