#!/usr/bin/env python3
"""V3.37: fixed-DSpark7 ordinary dispatch and independent dual-Bound audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_36 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints


ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'evidence/20260928_loop080_bound/run580/bound_calibration_v3_36.json': 'd3a5c5ad9c2f7cca7ae9864627118c77b1fc120134fa748b2f9d83fb933e4f71',
    'evidence/20260928_loop081_bound/run606/summary.json': 'c6e37798dc686d7bd7d6279813ac943398a282959e40e5e7aa4b102fb0d3de11',
    'evidence/20260928_loop081_bound/run606/product_summary.json': '0de47dc0821cfd86b4ff71ed8513ef8b74af06eeb7ae8730b504ddcd820c8029',
    'evidence/20260928_loop081_bound/run606/astra_post_review.md': 'ceca4b6fa50e6b5aeb07da4d4a81db5afe74e0e3a0c68dc3da3af8d6921809d7',
    'evidence/20260928_loop081_bound/run607/astra_bound_review.md': '65b1215d5b9ebf394cc02881521daafe094b76ae0bb95eb44c836c33a461c53e',
    'evidence/20260928_loop081_bound/run607/astra_bound_review_inputs.json': '2c8d29631e3da0fea60618e47224d6730f087beb31c653d418e7423049424626',
    'evidence/20260928_loop081_bound/run605/astra_post_review.md': '5b4ef97d116bae8a0f2bce27f020903e69b9494c63727b8adde44e77e9bf495e',
    'scripts/loop081_dispatch_reduce_run606.py': '680c959b10afe96a3ad64df252db96609a4fa934562ecf0779bf4daeeb194d3a',
    'evidence/20260928_loop081_bound/run606/live/b/basis_admission.json': '5bbbbb3492d75c7c37679d911d1986174ddf8d3ebb599e982a90b894847defe7',
    'evidence/20260928_loop081_bound/run606/live/b/product_admission.json': '1d0a9bbfe941080a9b49b1b3f8b566460d94fa84664c6691ff67ae46bacd834f',
    'evidence/20260928_loop081_bound/run606/live/b/dispatch_admission.json': '721ab3206d1edaa437b8e630fed9e700c637f3e2ae778e50d408862bf2378a6d',
}
RUN605_RESULT = 'tasks/deepseek-extreme-p0/loops/loop-081/runs/mixed_32k_1024_c12/run605/result.json'
RUN606_RESULT = 'tasks/deepseek-extreme-p0/loops/loop-081/runs/mixed_32k_1024_c12/run606/result.json'
TASK_RESULT_PINS = {
    RUN605_RESULT: 'ceb51df6c99759dfa9ce6710460fccaa72cc142e3080679d576561d54e9bd31c',
    RUN606_RESULT: 'c14d12e768d1597277bdcb69781430006178a5cbfc0493269627f823ef0991d7',
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def check_raw(rows):
    for name, expected in rows.items():
        path = Path(name)
        path = path if path.is_absolute() else ROOT / path
        need(sha(path) == expected, f'raw SHA drift: {name}')


def require_v337_null_endpoints(model):
    """Fail closed on every new numeric Bound endpoint in this revision."""
    require_null_endpoints(model)
    bound = model['bound_semantics_v3_37']
    fields = (
        (bound['conditional_engineering_model'], 'numeric_endpoint_tps'),
        (bound['strict_resource_certificate'], 'latency_floor_s'),
        (bound['strict_scheduling_certificate'], 'latency_floor_s'),
        (bound, 'strict_product_e2e_tps_ceiling'),
        (bound, 'numeric_current_to_credible_limit_gap'),
    )
    for owner, key in fields:
        need(owner[key] is None, f'uncertified V3.37 endpoint: {key}')


def build():
    for name, expected in PINS.items():
        need(sha(ROOT / name) == expected, f'input SHA drift: {name}')
    for name, expected in TASK_RESULT_PINS.items():
        need(sha(ROOT / name) == expected, f'TaskCtl result SHA drift: {name}')
    prior = build_prior()
    prior_path = ROOT / next(iter(PINS))
    need(json.dumps(prior, ensure_ascii=False, indent=2) + '\n' == prior_path.read_text(),
         'V3.36 generator/output drift')
    need(prior['model_revision'] == 'V3.36' and
         prior['current']['accepted_formal_tps'] == 571.681,
         'formal Current/model revision drift')
    need(load(ROOT / RUN605_RESULT)['status'] == 'invalid' and
         load(ROOT / RUN606_RESULT)['status'] == 'pass',
         'Run605/606 admission class drift')

    scope = 'evidence/20260928_loop081_bound/run606/'
    summary = load(ROOT / (scope + 'summary.json'))
    product = load(ROOT / (scope + 'product_summary.json'))
    arm = ROOT / (scope + 'live/b')
    basis_admission = load(arm / 'basis_admission.json')
    product_admission = load(arm / 'product_admission.json')
    dispatch_admission = load(arm / 'dispatch_admission.json')
    need(all(row['run_ts'] == 'LOOP081-RUN606-B' for row in
             (basis_admission, product_admission, dispatch_admission, product)) and
         summary['status'] == 'scoped_run606_reduction_pass',
         'run/admission identity drift')
    need(basis_admission['status'] == 'diagnostic_compact_basis_all8_admitted' and
         product_admission['status'] == 'product_identity_pass' and
         dispatch_admission['status'] == 'scoped_dispatch_identity_pass',
         'admission status drift')
    need(len(basis_admission['source_sha256']) == 97 and
         len(product_admission['raw_sha256']) == 281 and
         len(dispatch_admission['raw_sha256']) == 64,
         'raw provenance cardinality drift')
    need(product['product_admission_sha256'] ==
         dispatch_admission['product_admission_sha256'] ==
         PINS[scope + 'live/b/product_admission.json'],
         'Product admission SHA join drift')
    for rows in (basis_admission['source_sha256'], product_admission['raw_sha256'],
                 dispatch_admission['raw_sha256']):
        check_raw(rows)
    need(summary['admission_sha256'] == {
        'basis': PINS[scope + 'live/b/basis_admission.json'],
        'product': PINS[scope + 'live/b/product_admission.json'],
        'dispatch': PINS[scope + 'live/b/dispatch_admission.json']},
        'summary/admission SHA join drift')
    need(set(summary['cleanup']) == {
            'run_exit', 'stop_exit', 'stop_verify_exit', 'restore_exit',
            'source_sha_exit', 'source_compare_exit', 'script_sha_exit',
            'script_compare_exit', 'final_exit'} and
         set(summary['cleanup'].values()) == {'0'} and
         summary['rank0_totals']['ordinary_pairs'] == 40 and
         summary['rank0_totals']['context_rows'] == 5675 and
         summary['rank0_totals']['query_rows'] == 1813 and
         summary['rank0_totals']['existing_full_replay_syncs'] == 16,
         'Run606 totals/cleanup drift')
    need(len(summary['cohorts']) == 4 and
         all(len(c['allrank']) == 8 for c in summary['cohorts']) and
         sum(c['allrank'][0]['modes'].get('NONE', 0) for c in summary['cohorts']) == 24 and
         sum(c['allrank'][0]['modes'].get('FULL', 0) for c in summary['cohorts']) == 16,
         'Run606 all8 branch drift')
    need(product['status'] == 'diagnostic_accounting_pass' and
         product['totals']['scheduler_pre_actual_ids'] == 564 and
         product['totals']['bulk_accepted'] == 48588 and
         product['totals']['client_total_usage'] == 49152 and
         product['totals']['scheduler_pre_placeholders'] == 768,
         'same-W0 Product accounting drift')
    need(basis_admission['measured_rank0_totals']['cycles'] == 1214 and
         basis_admission['measured_rank0_totals']['active_target8_rows'] == 101848 and
         basis_admission['measured_rank0_totals']['current_physical_target_rows'] == 116544,
         'same-W0 Runtime occurrence drift')

    # This is an observed issued-row census, not a freshness or W-minus proof.
    ordinary = {'actual': 0, 'decode': 0, 'prefill_or_extend': 0, 'padded': 0}
    for cohort in range(5, 9):
        worker = load(arm / 'product' / f'rank0_cohort{cohort}.json')
        for forward in worker['frontier']['forward']:
            rows = forward['attention']
            need(len(rows) == 8 and all(row == rows[0] for row in rows),
                 'ordinary metadata group drift')
            actual = rows[0]['num_actual_tokens']
            decode = rows[0]['num_decode_tokens']
            need(0 <= decode <= actual <= forward['batch_num_tokens'],
                 'ordinary row shape drift')
            ordinary['actual'] += actual
            ordinary['decode'] += decode
            ordinary['prefill_or_extend'] += actual - decode
            ordinary['padded'] += forward['batch_num_tokens']
    need(ordinary == {'actual': 5675, 'decode': 1688,
                      'prefill_or_extend': 3987, 'padded': 5760},
         'ordinary issued-row census drift')
    need(ordinary['actual'] == summary['rank0_totals']['context_rows'],
         'Target/context current row relation drift')

    prior['model_revision'] = 'V3.37'
    prior['bound_semantics_v3_37'] = {
        'active_objective': 'minimum Product execution time for unchanged DSpark7 algorithm, acceptance/cycle trajectory, output and model semantics',
        'measurement_class': 'same-W0 all8 scoped ordinary dispatch topology plus independent dual-Bound review',
        'current_formal_tps': 571.681,
        'run605': 'observer ownership invalid; no Draft/context zero-work inference or timing comparison',
        'run606': {
            'diagnostic_W0_only': True,
            'runtime_cycles': 1214,
            'runtime_target8_active_class_rows': 101848,
            'runtime_target8_physical_rows': 116544,
            'ordinary_pairs_per_rank': 40,
            'ordinary_target_issued_rows_per_rank': ordinary,
            'ordinary_graph_none_fallthrough': 24,
            'ordinary_graph_full_replay_with_existing_sync': 16,
            'ordinary_dspark_query_rows_per_rank': 1813,
            'ordinary_context_method_rows_per_rank': 5675,
            'actual_prebulk_ids': 564,
            'accepted_runtime_bulk_ids': 48588,
            'product_output_ids': 49152,
            'confidence': 'high for admitted current identity and issued geometry; low for compulsory work/traffic or unperturbed timing',
        },
        'conditional_engineering_model': {
            'status': 'open_calibration_track_not_a_strict_ceiling',
            'attained_mixed_service_prior': 'Run579/580 whole-chain GMM+HCCL concurrent slower than serial in their independent-ready fixture',
            'next_nodes': ['full Runtime Target/aux/logits', 'accept/state',
                           'Draft context per-layer KV', 'Draft query/head/commit',
                           'collective producer/consumer and Host submission'],
            'numeric_endpoint_tps': None,
            'unknowns': ['same-W0 node readiness and native identity',
                         'legal interbranch overlap and mixed resource competition',
                         'observer OFF/ON timing transfer'],
        },
        'strict_resource_certificate': {
            'required_form': 'required_q(W0)>=W_q_minus and service_q(I)<=C_q_plus*duration(I)+B_q',
            'W_minus_complete': False,
            'exact_board_C_plus_B_complete': False,
            'latency_floor_s': None,
            'confidence': 'high that present attained service and current counters cannot substitute for either missing proof',
        },
        'strict_scheduling_certificate': {
            'all8_legal_dependency_dag_complete': False,
            'necessary_node_duration_floors_complete': False,
            'latency_floor_s': None,
            'context_marker_age_is_slack': False,
        },
        'strict_product_e2e_tps_ceiling': None,
        'numeric_current_to_credible_limit_gap': None,
        'highest_information_next_experiment': 'same-W0 all8 two adjacent Runtime cycles with Target aux/logits, acceptance/state, context per-layer KV generations, actual Draft query/commit and next Target consumer; structural admission before observer-controlled timing',
        'parallel_resource_work': 'declare required fresh semantic evaluation/initial availability class and obtain authoritative exact-board compute/HBM/HCCL C_plus,B with source-to-booted-image/freshness proof',
        'independent_review': 'Run607 Astra High SHA-pinned dual-Bound review',
    }
    prior['next_measurement']['priority'] = prior['bound_semantics_v3_37']['highest_information_next_experiment']
    prior['input_paths'] += list(PINS) + [RUN605_RESULT, RUN606_RESULT]
    require_v337_null_endpoints(prior)
    need(not any(prior['proof_dag']['certified'].values()),
         'uncertified strict proof promoted')
    return prior


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'ok', 'revision': 'V3.37',
                      'finite_strict_endpoints': 0,
                      'ordinary_issued_rows': result['bound_semantics_v3_37']['run606']['ordinary_target_issued_rows_per_rank']}))


if __name__ == '__main__':
    main()
