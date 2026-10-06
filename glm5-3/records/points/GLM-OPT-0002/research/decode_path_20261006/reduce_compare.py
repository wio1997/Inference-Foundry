"""Validate and summarize actual Run251 evidence; never starts a request."""
import ast
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def target(path, name):
    functions = [x for x in ast.walk(ast.parse(path.read_text())) if isinstance(x, ast.FunctionDef) and x.name == name]
    if name == 'finalize':
        tree = ast.parse(path.read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'PrepareAndFinalizeWithAll2All')
        functions = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name]
    assert len(functions) == 1
    functions[0].name = 'finalize'
    return ast.dump(functions[0], include_attributes=False)


if __name__ == '__main__':
    point = Path(__file__).resolve().parents[2]
    run = point / 'runs/GLM-RUN-0251'
    plan = read(run / 'functional_plan.json')
    for name, field in [('prepare_finalize_original.py', 'original_sha256'),
                        ('prepare_finalize_candidate.py', 'candidate_sha256'),
                        ('prepare_finalize_shim.py', 'shim_sha256')]:
        assert hashlib.sha256((run/name).read_bytes()).hexdigest() == plan[field]
    assert target(run/'prepare_finalize_candidate.py', 'finalize') == target(run/'prepare_finalize_shim.py', '_glm53_direct_finalize')
    byte_fields = ['full_output_bytes_equal', 'input_bytes_preserved', 'independent_expected_bytes',
                   'unpad_bytes_equal', 'retained_output_lifetime', 'all_ranks_passed']
    cases = []
    for rank in range(16):
        obj = read(run/'micro_checks'/('rank%d.json' % rank))
        assert obj['rank'] == rank and len(obj['cases']) == 20
        assert all(case[field] is True for case in obj['cases'] for field in byte_fields)
        cases.extend(obj['cases'])
    actual_padding_nan = {}
    for mode in ['B1', 'B2']:
        rows = read(run/(mode+'_witness.json'))
        assert {x['rank'] for x in rows} == set(range(16)) and len(rows) == 16
        assert all(x[field] is True for x in rows for field in ['actual_hidden_bitwise_equal', 'input_bytes_equal', 'all_ranks_bytes_equal'])
        actual_padding_nan[mode] = [{'rank': x['rank'], 'NaNs': x['input_nan_count']} for x in rows if x['input_nan_count']]
    results = read(run/'comparison_results.json')
    assert len(results) == 12 and all(x['profiler_active'] is False for x in results)
    compact = read(run/'matched_event_timeline.json')
    for result in results:
        event_path = run/(result['label']+'_events.json')
        if event_path.exists():
            assert hashlib.sha256(event_path.read_bytes()).hexdigest() == compact[result['label']]['original_events_sha256']
            events = read(event_path)
        else:
            # Git carries timing/chunk fields; full raw stays at the indexed
            # server path. Wire response itself is retained separately as SSE.
            row = compact[result['label']]
            assert row['DONE'] is True
            events = [dict(arrived_ns=c['arrived_ns'], value=dict(choices=[dict(token_ids=c['token_ids'])])) for c in row['chunks']]
            events.append(dict(value=None))
        ids, counts, arrivals = [], [], []
        assert events[-1]['value'] is None
        for event in events:
            for choice in (event['value'] or {}).get('choices', []):
                tokens = choice.get('token_ids') or choice.get('delta', {}).get('token_ids') or []
                if tokens:
                    ids.extend(tokens)
                    counts.append(len(tokens))
                    arrivals.append(event['arrived_ns'])
        assert ids == result['token_ids'] and counts == result['chunks']
        assert abs((arrivals[-1]-arrivals[0])/1e6/(len(ids)-1)-result['TPOT_ms']) < 1e-9
        assert any('hits_total' in k and v == result['prompt_tokens'] for k,v in result['external_KV_delta'].items())
    short = [x for x in results if '_measure' in x['label']]
    assert len(short) == 8 and all(x['prompt_tokens'] == 2334 and x['completion_tokens'] == 8 for x in short)
    assert all(x['token_ids'] == short[0]['token_ids'] and x['chunks'] == [1,2,2,2,1] for x in short)
    complete = [x for x in results if '_complete' in x['label']]
    assert len(complete) == 4 and all(x['semantic_accepted'] and x['finish_reason'] == 'stop' and x['final_content'] == '2' for x in complete)
    assert all(x['token_ids'] == [154842,17,154827] and x['chunks'] == [1,1,1] for x in complete)
    pairs = []
    for a, b in [('A1', 'B1'), ('A2', 'B2')]:
        av = [x['TPOT_ms'] for x in short if x['label'].startswith(a+'_')]
        bv = [x['TPOT_ms'] for x in short if x['label'].startswith(b+'_')]
        ac = next(x for x in complete if x['label'] == a+'_complete')
        bc = next(x for x in complete if x['label'] == b+'_complete')
        pairs.append(dict(baseline=a, patched=b, short_A_ms=av, short_B_ms=bv,
                          short_A_median_ms=statistics.median(av), short_B_median_ms=statistics.median(bv),
                          short_TPOT_reduction_percent=100*(1-statistics.median(bv)/statistics.median(av)),
                          complete_A_PD_wall_s=ac['PD_wall_s'], complete_B_PD_wall_s=bc['PD_wall_s'],
                          complete_wall_reduction_percent=100*(1-bc['PD_wall_s']/ac['PD_wall_s']),
                          complete_A_P_wall_s=ac['P_wall_s'], complete_B_P_wall_s=bc['P_wall_s'],
                          complete_A_D_wall_s=ac['D_wall_s'], complete_B_D_wall_s=bc['D_wall_s'],
                          complete_D_wall_reduction_percent=100*(1-bc['D_wall_s']/ac['D_wall_s'])))
    summary = dict(run='GLM-RUN-0251', correctness=dict(native_HCCL_cases=320, ranks=16,
                   full_output_bytes=True, input_preserved=True, independent_rank_order=True,
                   unpad=True, retained_output_lifetime=True, real_model_B_warm_checks=32,
                   actual_padding_NaNs=actual_padding_nan, candidate_AST_equals_executed_shim=True),
                   comparison=pairs, final_state=read(run/'restoration.json'),
                   verdict='INCONCLUSIVE for formal PERF_KEEP; positive scoped code signal, complete E2E first pair near noise',
                   Current=None, formal_SLA=False, full_API_contract=False)
    (run/'execution_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
