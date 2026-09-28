#!/usr/bin/env python3
"""Reduce admitted Run597 fixed-algorithm W0 into current execution row classes.

This is a workload cardinality ledger, not compulsory work or a time bound.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
ARM = ROOT / 'evidence/20260928_loop081_bound/run597/live/b'
OUT = ROOT / 'evidence/20260928_loop081_bound/run598'
DRAFT_SOURCE = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py')
DRAFT_SOURCE_SHA = 'e9163996db794c5fba777ea55a5374283780a2fd3404778f5369caea76f9344f'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if (OUT / 'summary.json').exists():
        raise ValueError('Run598 output already exists')
    admission_path = ARM / 'basis_admission.json'
    cleanup_path = ARM.parent / 'cleanup_status.txt'
    admission = json.loads(admission_path.read_text())
    cleanup = dict(line.split('=', 1) for line in cleanup_path.read_text().splitlines())
    assert admission['status'] == 'diagnostic_compact_basis_all8_admitted'
    assert set(cleanup) == {'run_exit', 'stop_exit', 'stop_verify_exit',
        'restore_exit', 'source_sha_exit', 'source_compare_exit',
        'script_sha_exit', 'script_compare_exit', 'final_exit'}
    assert all(value == '0' for value in cleanup.values()), cleanup
    assert sha(DRAFT_SOURCE) == DRAFT_SOURCE_SHA
    draft_source = DRAFT_SOURCE.read_text()
    assert draft_source.count('self.num_query_per_req = self.num_speculative_tokens') == 1
    assert draft_source.count('self.num_query_per_req = 1 + self.num_speculative_tokens') == 1
    assert draft_source.count('num_query_total = batch_size * self.num_query_per_req') == 1
    assert draft_source.count('self._dflash_num_context = int(cad.query_start_loc_cpu[batch_size])') == 1
    source = {str(admission_path.relative_to(ROOT)): sha(admission_path),
              str(cleanup_path.relative_to(ROOT)): sha(cleanup_path),
              str(DRAFT_SOURCE): DRAFT_SOURCE_SHA}
    cohorts = []
    for cohort in range(5, 9):
        path = ARM / 'basis' / f'rank0_cohort{cohort}.json'
        record = json.loads(path.read_text())
        source[str(path.relative_to(ROOT))] = sha(path)
        assert source[str(path.relative_to(ROOT))] == admission['source_sha256'][str(path.relative_to(ROOT))]
        cycles = record['cycles']
        assert len(record['count_history']) == len(record['token_history']) == cycles
        active = sum(count > 0 for row in record['count_history'] for count in row)
        staged = sum(record['staged_output_counts'])
        assert staged == sum(count for row in record['count_history'] for count in row)
        assert active * 8 == record['basis_check']['active_target8_rows']
        assert cycles * 96 == record['basis_check']['current_physical_target_rows']
        witnesses = record['draft_model_witnesses']
        assert witnesses and set(record['basis_check']['target_checkpoint_cycles']) == {w['cycle'] for w in witnesses}
        query_widths = sorted({w['num_query_per_req'] for w in witnesses})
        assert all(w['sample_from_anchor'] and w['num_context'] == 96 for w in witnesses)
        if query_widths != [7]:
            draft_submitted = None
            draft_active = None
            draft_terminal = None
        else:
            # Every-cycle width relies on a source-static fixed Q=7 setting.
            # Sparse live witnesses test that setting, but do not time the path.
            draft_submitted = cycles * 12 * 7
            draft_active = active * 7
            draft_terminal = 12 * 7
        cohorts.append(dict(cohort=cohort, cycles=cycles,
            target8_issued_rows=cycles * 96,
            target8_active_class_rows=active * 8,
            target8_parked_class_rows=cycles * 96 - active * 8,
            accepted_staged_tokens=staged,
            active_context_padding_rows=active * 8 - staged,
            draft_query_width_live_sparse=query_widths,
            draft_query_issued_rows_conditional=draft_submitted,
            draft_query_active_class_rows_conditional=draft_active,
            draft_query_terminal_issued_rows_conditional=draft_terminal,
            draft_context_combine_issued_rows_conditional=cycles * 96,
            draft_context_kv_three_layer_issued_rows_conditional=cycles * 96 * 3,
            host_park_events=len(record['host_park_events']),
            scheduled_target_cycles=sum(bool(x['scheduled_target']) for x in record['branch_history']),
            sparse_witness_cycles=[w['cycle'] for w in witnesses]))
    keys = ('cycles', 'target8_issued_rows', 'target8_active_class_rows',
            'target8_parked_class_rows', 'accepted_staged_tokens',
            'active_context_padding_rows', 'host_park_events',
            'scheduled_target_cycles')
    totals = {key: sum(row[key] for row in cohorts) for key in keys}
    for key in ('draft_query_issued_rows_conditional',
                'draft_query_active_class_rows_conditional',
                'draft_query_terminal_issued_rows_conditional',
                'draft_context_combine_issued_rows_conditional',
                'draft_context_kv_three_layer_issued_rows_conditional'):
        values = [row[key] for row in cohorts]
        totals[key] = sum(values) if all(value is not None for value in values) else None
    assert totals['target8_active_class_rows'] == admission['measured_rank0_totals']['active_target8_rows']
    assert totals['target8_issued_rows'] == admission['measured_rank0_totals']['current_physical_target_rows']
    result = dict(status='diagnostic_W0_current_execution_class_ledger',
        scope='new Run597 measured W0; source-static Draft Q=7 conditional on sparse live witnesses',
        limitation=('Issued and active-class rows are current execution occurrences, '
                    'not unique fresh semantic work, compulsory operations, HBM bytes or time bounds.'),
        cohorts=cohorts, totals=totals, source_sha256=source,
        formal_Run99_same_state=False, compulsory_work_complete=False,
        hardware_resource_bound_s=None, scheduling_execution_bound_s=None,
        product_e2e_bound_tps=None, formal_current_tps=571.681)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(totals, sort_keys=True))


if __name__ == '__main__':
    main()
