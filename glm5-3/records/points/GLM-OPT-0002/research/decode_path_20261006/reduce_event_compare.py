"""Validate Run253 original evidence and reduce paired output timing offline.

Does not start a request. Raw arrivals are client166 monotonic nanoseconds;
no cross-host timestamp subtraction or profiler-ON/OFF rescaling is used.
"""
import ast
import hashlib
import json
import statistics
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def first_helper(p):
    return ast.dump(next(n for n in ast.parse(p.read_text()).body
                         if isinstance(n, ast.FunctionDef) and n.name == 'maybe_record_moe_event'),
                    include_attributes=False)


if __name__ == '__main__':
    research = Path(__file__).resolve().parent
    run = research.parents[1] / 'runs/GLM-RUN-0253'
    plan = read(run / 'functional_plan.json')
    for name, identities in plan['source_files'].items():
        for kind in ('original', 'candidate', 'shim'):
            assert sha(run/kind/name) == identities[kind+'_sha256']
    assert first_helper(run/'candidate/utils.py') == first_helper(run/'shim/utils.py')
    assert all(read(research/'moe_event_cpu_correctness.json')[f] for f in
               ('passed', 'policy_false_no_stream_or_event_call', 'policy_true_original_record_wait',
                'none_consumer_safe', 'true_consumer_missing_event_rejected'))
    native = read(run/'micro_execution.json')
    assert native['passed'] and native['inner_exit_code'] == 0 and not native['timed_out']
    fields = ('stock_candidate_bytes_equal', 'independent_expected_bytes',
              'input_bytes_preserved', 'retained_output_lifetime', 'event_policy', 'all_ranks_passed')
    for rank in range(16):
        row = read(run/'micro_checks'/('rank%d.json' % rank))
        assert row['rank'] == rank and row['passed'] and len(row['cases']) == 11
        assert all(c[f] for c in row['cases'][:10] for f in fields)
        assert row['cases'][10]['overlap_true_cross_stream'] and row['cases'][10]['all_ranks_passed']
    owned = set(read(run/'guards_after.json')['167']['worker_namespace_pids'].values())
    for mode, name in enumerate(('A1', 'B1', 'A2', 'B2')):
        rows = read(run/(name+'_witness.json'))
        assert len(rows) == 16 and {r['rank'] for r in rows} == set(range(16))
        assert {r['pid'] for r in rows} == owned
        assert all(r['mode'] == mode and not r['configured_overlap'] and
                   r['returned_none'] == bool(mode % 2) and
                   r['candidate_utils_sha256'] == plan['source_files']['utils.py']['candidate_sha256'] for r in rows)
    results = read(run/'comparison_results.json')
    assert len(results) == 12 and all(not x['profiler_active'] for x in results)
    timeline = {}
    for result in results:
        label = result['label']
        events = read(run/(label+'_events.json'))
        assert events[-1]['value'] is None
        wire = []
        for line in (run/(label+'_D.sse')).read_text().splitlines():
            if line.startswith('data: '):
                value = line[6:]
                wire.append(None if value == '[DONE]' else json.loads(value))
        assert wire == [e['value'] for e in events]
        ids, chunks, content, finish = [], [], '', None
        for event in events:
            for c in (event['value'] or {}).get('choices', []):
                content += c.get('delta', {}).get('content') or ''
                if c.get('finish_reason') is not None:
                    finish = c['finish_reason']
                tokens = c.get('token_ids') or c.get('delta', {}).get('token_ids') or []
                if tokens:
                    ids.extend(tokens)
                    chunks.append(dict(arrived_ns=event['arrived_ns'], token_ids=tokens))
        assert ids == result['token_ids'] and [len(c['token_ids']) for c in chunks] == result['chunks']
        assert content == result['final_content'] and finish == result['finish_reason']
        span_ms = (chunks[-1]['arrived_ns'] - chunks[0]['arrived_ns']) / 1e6
        assert abs(span_ms/(len(ids)-1)-result['TPOT_ms']) < 1e-9
        assert abs(result['P_wall_s']+result['D_wall_s']-result['PD_wall_s']) < 1e-9
        assert any('hits_total' in k and v == result['prompt_tokens'] for k, v in result['external_KV_delta'].items())
        timeline[label] = dict(original_events_sha256=sha(run/(label+'_events.json')),
                               wire_sha256=sha(run/(label+'_D.sse')), chunks=chunks, DONE=True,
                               finish_reason=finish, first_last_span_ms=span_ms)
    short = [r for r in results if '_measure' in r['label']]
    complete = [r for r in results if r['complete']]
    assert len(short) == 8 and all(r['prompt_tokens'] == 2334 and r['completion_tokens'] == 8 for r in short)
    assert all(r['token_ids'] == [785,1196,374,10156,264,3405,304,8452] and r['chunks'] == [1,2,2,2,1] for r in short)
    assert len(complete) == 4 and all(r['semantic_accepted'] and r['prompt_tokens'] == 58 and
                                    r['completion_tokens'] == 23 and r['finish_reason'] == 'stop' and
                                    r['final_content'] == 'P 负责处理输入并生成 KV；D 复用 KV，逐步生成输出。' for r in complete)
    assert all(r['token_ids'] == complete[0]['token_ids'] and r['chunks'] == complete[0]['chunks'] for r in complete)
    bodies = []
    for r in complete:
        body = read(run/(r['label']+'_request.json'))
        assert body.pop('cache_salt') == plan['run_id']+'-'+r['label']
        assert body['thinking_token_budget'] == 0 and body['max_tokens'] == 96
        assert not body.get('ignore_eos', False) and 'min_tokens' not in body
        bodies.append(body)
    assert all(b == bodies[0] for b in bodies)
    pairs = []
    for a, b in (('A1','B1'), ('A2','B2')):
        av = [r['TPOT_ms'] for r in short if r['label'].startswith(a+'_')]
        bv = [r['TPOT_ms'] for r in short if r['label'].startswith(b+'_')]
        ac = next(r for r in complete if r['label'] == a+'_complete')
        bc = next(r for r in complete if r['label'] == b+'_complete')
        deltas = {key:1000*(ac[key]-bc[key]) for key in ('P_wall_s','D_wall_s','PD_wall_s')}
        pairs.append(dict(baseline=a, patched=b, short_A_ms=av, short_B_ms=bv,
                          short_A_median_ms=statistics.median(av), short_B_median_ms=statistics.median(bv),
                          short_TPOT_reduction_percent=100*(1-statistics.median(bv)/statistics.median(av)),
                          complete_A={k:ac[k] for k in ('P_wall_s','D_wall_s','PD_wall_s','TPOT_ms')},
                          complete_B={k:bc[k] for k in ('P_wall_s','D_wall_s','PD_wall_s','TPOT_ms')},
                          complete_deltas_ms=deltas,
                          complete_PD_wall_reduction_percent=100*(1-bc['PD_wall_s']/ac['PD_wall_s']),
                          complete_D_wall_reduction_percent=100*(1-bc['D_wall_s']/ac['D_wall_s']),
                          complete_generation_TPOT_reduction_percent=100*(1-bc['TPOT_ms']/ac['TPOT_ms']),
                          complete_generation_span_reduction_ms=22*(ac['TPOT_ms']-bc['TPOT_ms'])))
    def phase_drift(letter, field):
        vals = [r[field] for r in complete if r['label'].startswith(letter)]
        return abs(vals[1]-vals[0])
    summary = dict(run='GLM-RUN-0253', correctness=dict(native_cases=176, ranks=16,
                   actual_imports=True, independent_expected_bytes=True, lifetime=True,
                   true_aux_stream_dependency=True, model_witnesses=64,
                   exact_short_ids_and_chunks=True, exact_complete_23_ids_content_chunks_EOS=True,
                   full_external_KV_hits=True, candidate_helper_AST_equals_executed_shim=True),
                   comparison=pairs,
                   observed_phase_drift=dict(baseline_complete_PD_ms=1000*phase_drift('A','PD_wall_s'),
                                            baseline_complete_P_ms=1000*phase_drift('A','P_wall_s'),
                                            baseline_complete_D_ms=1000*phase_drift('A','D_wall_s'),
                                            baseline_complete_TPOT_ms=phase_drift('A','TPOT_ms'),
                                            patched_complete_D_ms=1000*phase_drift('B','D_wall_s'),
                                            patched_complete_TPOT_ms=phase_drift('B','TPOT_ms')),
                   attribution='Repeated D generation benefit exceeds observed phase drift; do not credit P-wall deltas to the D code.',
                   verdict='Positive scoped code evidence; INCONCLUSIVE/PARKED for formal PERF_KEEP/full-workload acceptance',
                   limits=['One bounded complete natural-EOS fixture, not standard dynamic workload/capacity/SLA acceptance.',
                           'MTP emits two-token chunks; average TPOT is not individual token ITL or SLA percentile.',
                           'A/B uses a shared in-memory selector and None-safe consumers in both modes; pure patch has no selector.',
                           'Observed phase drift is a descriptive small-sample bound, not a statistical confidence interval.'],
                   final_state=read(run/'restoration.json'), Current=None, formal_SLA=False, full_API_contract=False)
    (run/'matched_event_timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
    (run/'execution_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
