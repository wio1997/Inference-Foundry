"""Saved-profile evidence for three BF16 epilogues and one INT32 control."""
from collections import defaultdict, Counter
from statistics import median
from finalize_origin_coverage_v9 import O, read, write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import fullkey


def stats(values):
    return dict(n=len(values), minimum=min(values), median=median(values), maximum=max(values)) if values else None


def main():
    output = O/'quant-vector-owner-profile-v82.json'
    assert not output.exists()
    cohorts = {(2304,5120,'DT_BF16'): 'shared_Down', (1280,32768,'DT_BF16'): 'Q_B',
               (1280,4096,'DT_BF16'): 'index_Q', (5120,4608,'INT32'): 'shared_GateUp_control'}
    selected = []
    for row in read(O/'A3-typed-ingress-regime-models-v80.json')['regimes']:
        key = (row['K'],row['N'],row['signature']['outputDataTypes_text'])
        if key not in cohorts or row['signature']['inputDataTypes_text'].split(';')[:2] != ['INT8','INT8']: continue
        selected.append(dict(regime_id=row['regime_id'], cohort=cohorts[key], module_candidates=row['module_candidates'],
            operator=row['operator'], role=row['role'], origin=row['origin'], context=row['context'],
            executions=row['executions'], M=row['M'], K=row['K'], N=row['N'], signature=row['signature']))
    assert len(selected) == 31 and sum(r['executions'] for r in selected) == 14143
    lookup = {fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']): r for r in selected}
    assignments = defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
        assignments[a[0]][a[1]] = a
    observations = defaultdict(list); seen = set(); missing = Counter()
    metrics = ['aic_total_cycles','aic_mac_ratio','aiv_total_cycles','aiv_vec_ratio','aiv_vec_time','aiv_mte2_ratio','aiv_mte3_ratio']
    for db in range(32):
        native = read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json')
        defs = {r['globalTaskId']:r for r in native['COMPUTE_definitions']}
        tasks = {r['task_record_rowid']:r for r in native['TASK_records']}
        current = {}
        for tid,a in assignments[db].items():
            task = tasks[tid]; definition = defs[task['globalTaskId']]
            key = fullkey(a[3],definition['opType_text'],a[5],a[2],a[6],definition)
            if key in lookup:
                assert (db,tid) not in seen
                seen.add((db,tid));current[tid] = lookup[key]
        matched = set()
        pmu = read(CACHE/f'analysis/native-PMU-complete-v18/db-{db:02d}.json')
        for group in pmu['groups']:
            tids = group['native_TASK_candidates']
            if len(tids) != 1 or tids[0] not in current: continue
            tid = tids[0];assert tid not in matched;matched.add(tid)
            row = current[tid];task = tasks[tid];definition = defs[task['globalTaskId']]
            values = {r['metric_name']:r['value'] for r in group['metric_rows']}
            observations[row['regime_id']].append(dict(database_index=db,TASK_rowid=tid,
                native_TASK_duration_ns=task['endNs']-task['startNs'], blockNum=definition['blockNum'],
                mixBlockNum=definition['mixBlockNum'], **{m:values[m] for m in metrics}))
        for tid in current:
            if tid not in matched:missing[current[tid]['regime_id']] += 1
        if db % 8 == 7: print(dict(scanned_DB=db+1,selected_tasks=len(seen)),flush=True)
    assert len(seen) == 14143 and not sum(missing.values())
    for row in selected:
        obs = observations[row['regime_id']]
        assert len(obs) == row['executions']
        row.update(native_PMU_records=len(obs),positive_AIV_total=sum(x['aiv_total_cycles'] > 0 for x in obs),
            positive_AIV_vector=sum(x['aiv_vec_time'] > 0 for x in obs),
            metrics={m:stats([x[m] for x in obs]) for m in metrics},records=obs)
    summary = {}
    for cohort in cohorts.values():
        rows = [r for r in selected if r['cohort'] == cohort]
        summary[cohort] = dict(regimes=len(rows),executions=sum(r['executions'] for r in rows),
            positive_AIV_total=sum(r['positive_AIV_total'] for r in rows),
            positive_AIV_vector=sum(r['positive_AIV_vector'] for r in rows),
            native_mix_descriptors=sorted({r['signature']['mixBlockNum'] for r in rows}))
    write(output,dict(regimes=selected,summary=summary,regime_count=31,executions=14143,
        BF16_output_regimes=22,BF16_output_executions=9671,INT32_control_regimes=9,INT32_control_executions=4472,
        missing_PMU_records=0,native_loaded_source_or_final_cast_owner_attested=False,
        interpretation='Mixed descriptors and positive AIV work can support an epilogue-owner hypothesis. They cannot identify the final cast instruction, branch, tiling key, binary version, value readers or buffer aliases.',
        task_carriers_are_not_additional_operators=True,no_new_device_work=True,
        global_849_missing_PMU_16_phase_3136_ambiguity_preserved=True,overall_complete=False))
    print(summary)


if __name__ == '__main__': main()
