#!/usr/bin/env python3
"""Read-only reuse audit for existing complete diagnostic Product W0 ledgers.

This audits work and output identity only. It deliberately makes no timing
transfer, fixed-cost, observer-effect or scheduling-savings claim.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
OUT = ROOT / 'evidence/20260928_loop081_bound/run649/product_ledger_reuse.json'

def load(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(run):
    base = ROOT / f'evidence/20260928_loop081_bound/run{run}'
    live = base / 'live/b'
    summary_path = base / ('summary.json' if run == 602 else 'product_summary.json')
    summary = load(summary_path)
    product_admission = load(live / 'product_admission.json')
    basis_admission = load(live / 'basis_admission.json')
    client_admission = load(live / 'client_admission.json')
    assert summary['status'] == 'diagnostic_accounting_pass'
    assert summary['run_ts'] == product_admission['run_ts']
    assert summary['product_admission_sha256'] == sha(live / 'product_admission.json')
    assert summary['totals']['client_total_usage'] == 49152
    assert summary['totals']['api_total_ids'] == 49152
    assert summary['totals']['scheduler_pre_actual_ids'] + summary['totals']['bulk_accepted'] == 49152
    assert product_admission['status'] == 'product_identity_pass'
    assert client_admission['status'] == 'client_two_phase_admitted'
    assert client_admission['measured_summary']['success'] == 48
    assert client_admission['measured_summary']['fail'] == 0
    assert client_admission['measured_summary']['concurrency'] == 12
    assert client_admission['measured_summary']['max_tokens'] == 1024
    assert all(int(summary['cleanup'][name]) == 0 for name in summary['cleanup'])
    cohort_rows = []
    total_cycles = 0
    rank0_runtime_s = 0.0
    for cohort in range(5, 9):
        ranks = [load(live / f'basis/rank{rank}_cohort{cohort}.json') for rank in range(8)]
        cycle_set = {x['cycles'] for x in ranks}
        assert len(cycle_set) == 1, (run, cohort, cycle_set)
        cycles = cycle_set.pop()
        for rank, x in enumerate(ranks):
            assert x['rank'] == rank and x['cohort'] == cohort
            assert x['run_ts'] == summary['run_ts']
            assert all(len(x[key]) == cycles for key in
                       ('count_history', 'token_history', 'draft_history', 'branch_history'))
        cohort_summary = next(x for x in summary['cohorts'] if x['cohort'] == cohort)
        counts = cohort_summary['counts']
        assert counts['api_total_ids'] == counts['client_total_usage'] == 12288
        assert counts['scheduler_pre_actual_ids'] + counts['bulk_accepted'] == 12288
        total_cycles += cycles
        rank0_runtime_s += load(live / f'runtime/rank0_cohort{cohort}.json')['wall_seconds']
        cohort_rows.append({'cohort': cohort, 'cycles': cycles,
                            'actual_prebulk_ids': counts['scheduler_pre_actual_ids'],
                            'accepted_bulk_ids': counts['bulk_accepted'],
                            'final_api_ids': counts['api_total_ids']})
    return {
        'run': run,
        'w0': summary['run_ts'],
        'source': str(summary_path.relative_to(ROOT)),
        'source_sha256': sha(summary_path),
        'basis_admission_sha256': sha(live / 'basis_admission.json'),
        'product_admission_sha256': sha(live / 'product_admission.json'),
        'client_admission_sha256': sha(live / 'client_admission.json'),
        'basis_admission_status': basis_admission.get('status'),
        'diagnostic_product_wall_s': client_admission['measured_summary']['duration_s'],
        'diagnostic_tps': client_admission['measured_summary']['output_tps_diagnostic_only'],
        'rank0_runtime_scope_sum_s': rank0_runtime_s,
        'arithmetic_product_minus_rank0_runtime_s':
            client_admission['measured_summary']['duration_s'] - rank0_runtime_s,
        'total_cycles': total_cycles,
        'output_ids': 49152,
        'cohorts': cohort_rows,
        'limitation': 'Separate observer-perturbed W0; no formal Run99 timing/cost transfer or schedule gain.'
    }

def main():
    rows = [audit(run) for run in (602, 606)]
    assert rows[0]['w0'] != rows[1]['w0']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        'status': 'complete_product_work_and_output_ledger_reused',
        'cases': rows,
        'new_live_needed_for': ['selected value-ready/resource/queue edges',
                                'shape-conditioned primitive cost transfer',
                                'low-observer timing bridge'],
        'not_needed_again_for': ['final Scheduler/API IDs',
                                 'complete Runtime cycle/count trajectories',
                                 'client measured output count and diagnostic wall'],
        'framework_only_product_tps_bound': None,
    }, indent=2) + '\n')
    print(OUT)

if __name__ == '__main__':
    main()
