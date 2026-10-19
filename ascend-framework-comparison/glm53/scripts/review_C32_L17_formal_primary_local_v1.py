"""Primary independent acceptance; reads saved evidence only, no runtime calls."""
import collections
import hashlib
import json
import math
import re
from decimal import Decimal
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v4 as c

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {}


def read(path):
    raw = (ROOT / path).read_bytes()
    SOURCES[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def same(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (a, b)


def counters(doc, pids):
    out = {}
    for snap in doc['native_SDK_small_file_snapshots']:
        assert snap['rc'] == 0
        if snap['container_id'] != doc['container_id']:
            continue
        for f in snap['files']:
            assert f['complete_line_ending']
            assert hashlib.sha256(f['text'].encode()).hexdigest() == f['sha256']
            m = re.search(r'ptracer_(\d+)\.dat$', f['path'])
            if not m or int(m[1]) not in pids:
                continue
            pid = int(m[1]); tags = {}
            for line in f['text'].splitlines():
                p = line.split()
                if len(p) <= 11 or p[2] not in {'TP_MMC_PY_BATCH_GET', 'TP_MMC_LOCAL_BATCH_GET', 'TP_MMC_LOCAL_BATCH_GET_SIZE', 'TP_HYBM_SDMA_BATCH_LD_TO_GD'}:
                    continue
                v = tags.setdefault(p[2], [0, 0, 0, None, Decimal(0)])
                for i in range(3):
                    v[i] += int(p[3+i])
                v[3] = int(p[6]); total = Decimal(p[11])
                assert total.is_finite() and total >= 0
                v[4] += total
            for v in tags.values():
                assert v[0] == v[1] and v[2] == 0 and v[3] == 0
            out[pid] = tags
    assert set(out) == set(pids)
    return out


def main():
    native = read(c.FORMAL_NATIVE_AUDIT)
    measurement = read(c.FORMAL_MEASUREMENT)
    client = read(c.FORMAL_CLIENT_AUDIT)
    report_path = f'runs/{c.FORMAL_RUN}/formal-result.remote.json'
    raw = read(report_path)
    review = read(c.FORMAL_REVIEW)
    baseline_acceptance = read('results/C32-L17-native-pre-formal-acceptance-v2.json')
    assert baseline_acceptance['decision'] == 'ACCEPTED_FRESH_PRE_FORMAL_BASELINE_ONLY'
    assert native['pass'] and measurement['measurement_valid'] and client['client_audit_pass']
    assert raw['phase'] == 'COMPLETE' and raw['baseline_accepted'] is True
    for doc in [native, measurement, client]:
        for p, expected in doc['source_evidence_sha256'].items():
            actual = hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            assert actual == expected, p
            SOURCES[p] = actual
    assert measurement['raw_formal_report_sha256'] == SOURCES[report_path]
    assert measurement['native_postrun_evidence_sha256'] == SOURCES[c.FORMAL_NATIVE_AUDIT]
    sessions = {}; writes = []; readers = []; sdk_workers = []; eager_workers = []
    for node in c.ALL_NODES:
        docs = []
        for rid in [c.FORMAL_BASELINE_CAPTURE, c.FORMAL_CAPTURE]:
            p = f'runs/{rid}/node-{node}.json'; read(p)
            _, _, o = c.load_native_capture(p)
            docs.append(o)
        before, after = docs
        assert before['container_id'] == after['container_id'] == native['runtime_CIDs'][str(node)]
        assert after['provider_phase'] == 'READY' and after['/health']['status'] == 200
        assert before['profiler_artifacts'] == after['profiler_artifacts']
        assert after['supervisor']['phase'] == 'MODEL_RUNNING'
        assert 'model_exit_code' not in after['supervisor']
        old, new = [c.persistent_events(o) for o in docs]
        init = {x['pid']: x for x in new if x['event'] == 'initialize_complete'}
        assert len(init) == 16
        assert all(x['rc'] == 0 and x['protocol'] == 'SDMA' and x['protocol_value'] == 1 for x in init.values())
        assert {x['namespace_pid'] for x in after['owned_npu_process_rows']} == set(init) | {after['supervisor']['provider_pid'], after['supervisor']['model_pid']}
        sessions.update({x['session']: (node, x['device_id']) for x in init.values()})
        for key, item in before['persistent_audit'].items():
            assert after['persistent_audit'][key]['rows'][:len(item['rows'])] == item['rows']
        for key, item in before['alignment_audit'].items():
            assert after['alignment_audit'][key]['rows'][:len(item['rows'])] == item['rows']
        c.alignment_events(before); c.alignment_events(after)
        old_set = {json.dumps(x, sort_keys=True) for x in old}
        delta = [x for x in new if json.dumps(x, sort_keys=True) not in old_set]
        assert not any(x['event'] in {'write_error','model_state_send_error','forbidden_tcp_model_payload'} for x in delta)
        for x in delta:
            if x['event'] == 'write_complete':
                assert node in c.P_NODES and x['pid'] in init and x['rc'] == 0
                assert x['payload_bytes_completed'] == x['payload_bytes_requested'] > 0
                assert x['context']['kind'] in {'send_kvcache','send_aux'}
                writes.append(dict(x, node=node))
        b, a = [counters(o, set(init)) for o in docs]
        sdk_workers.extend((node, pid) for pid in a)
        if node in c.P_NODES:
            for pid in init:
                name = 'TP_MMC_PY_BATCH_GET'
                good = a[pid][name][1] - b[pid].get(name,[0,0])[1]
                assert good > 0
                readers.append({'node':node,'pid':pid,'device_id':init[pid]['device_id'],'new_good_batch_get':good})
            descriptor_delta = sum(x['native_sdma']['copy_descriptor_bytes'] for x in after['logs'].values()) - sum(x['native_sdma']['copy_descriptor_bytes'] for x in before['logs'].values())
            assert descriptor_delta == sum(x['payload_bytes_completed'] for x in writes if x['node'] == node)
            assert len(after['native_optional_extension_worker_maps']) == 16
        else:
            assert after['native_optional_extension_worker_maps'] == []
            eager = after['D_eager_input_audit']
            assert not eager['file_errors']
            for f in eager['files']:
                records = [x['row'] for x in sorted(eager['records'],key=lambda x:x['line']) if x['path'] == f['path']]
                data = ''.join(json.dumps(x, sort_keys=True, separators=(',',':'), ensure_ascii=True)+'\n' for x in records).encode()
                assert len(data) == f['bytes'] and hashlib.sha256(data).hexdigest() == f['sha256']
            for pid in init:
                records = [x['row'] for x in eager['records'] if x['row']['pid'] == pid]
                selected = [x for x in records if x['is_draft_worker'] and x['forward_mode'] == 'DECODE' and x['batch_size'] == 2]
                assert len(selected) == 1
                assert selected[0]['native_no_copy_get'] is True and selected[0]['clone_distinct'] is True
                assert selected[0]['field_verdicts'] == {'identity':109,'immutable_equal':0}
                eager_workers.append((node,pid))
        for o in docs:
            assert hashlib.sha256(o['/metrics']['body'].encode()).hexdigest() == o['/metrics']['sha256']
        assert all(float(line.rsplit(' ',1)[1]) == 0 for line in after['/metrics']['body'].splitlines() if line.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{')))
        assert [x for log in after['logs'].values() for x in log['error_lines']] == [x for log in before['logs'].values() for x in log['error_lines']]
    assert len(readers) == 32 and len(sdk_workers) == 64 and len(eager_workers) == 32
    destinations = {sessions[x['remote_session']] for x in writes}
    assert destinations == {(n,d) for n in c.D_NODES for d in range(16)}
    descriptor_bytes = sum(x['payload_bytes_completed'] for x in writes)
    assert len(writes) == native['native_PD_write_batches'] == 3456
    assert descriptor_bytes == native['native_PD_descriptor_bytes'] == 2808364105728
    assert native['physical_wire_bytes'] is None and native['native_PD_request_ID_join'] is None
    cpu_wire = read(c.INDEPENDENT_METRICS_NODE); assert cpu_wire['rc'] == 0
    cpu = json.loads(cpu_wire['stdout']); archive = cpu['independent_formal_metric_archive_audit']
    assert archive['archive_sha256'] == measurement['D_metric_archive_sha256']
    assert cpu['formal_report_identity']['sha256'] == SOURCES[report_path]
    assert archive['all_native_bodies_SHA_and_DP_gauges_recomputed'] and not archive['errors']
    assert archive['sample_pairs'] == archive['complete_pairs'] == 248 and archive['native_HTTP_rows'] == 496
    graph = cpu['formal_native_target_graph_counters']
    assert set(graph['delta_by_global_EP_rank']) == {str(i) for i in range(32)}
    assert all(v > 0 for v in graph['delta_by_global_EP_rank'].values())
    rows = raw['formal']['rows']; assert len(rows) == 128 and len(raw['warmup']) == 32
    metrics = collections.defaultdict(list)
    for x in rows:
        y=x['result']; valid=y['native']['extra_perf_data']['glm53_valid_output_chunk_times']
        assert y['logical_success'] and y['transport_success'] and y['count_pass']
        assert y['native']['input_tokens'] == 80000 and y['native']['output_tokens'] == 600
        metrics['ttft_seconds'].append(valid[0]-x['outer_dispatch_perf_counter_ns']/1e9)
        metrics['tpot_seconds'].append((valid[-1]-valid[0])/599)
        metrics['e2e_seconds'].append((x['outer_terminal_perf_counter_ns']-x['outer_dispatch_perf_counter_ns'])/1e9)
    clock=(max(x['outer_terminal_perf_counter_ns'] for x in rows)-min(x['outer_dispatch_perf_counter_ns'] for x in rows))/1e9
    same(clock,measurement['full_clock_seconds']);same(76800/clock,measurement['output_tokens_per_second'])
    for name, values in metrics.items():
        values.sort()
        for label,p in [('P50',.5),('P90',.9),('P99',.99)]:
            index=(len(values)-1)*p; lo=math.floor(index); hi=math.ceil(index)
            q=values[lo]+(values[hi]-values[lo])*(index-lo)
            same(q,measurement['quantiles'][name][label])
    assert measurement['aggregate_SLA_pass'] is False
    reference_path=review['evaluation_policy']['reference']; reference=read(reference_path)
    vllm_path='results/vllm-formal-C32-v2-measurement-v1.json'; vllm=read(vllm_path)
    tps_gain=measurement['output_tokens_per_second']/reference['output_tokens_per_second']-1
    tpot_gain=1-measurement['quantiles']['tpot_seconds']['P50']/reference['quantiles']['tpot_seconds']['P50']
    policy=review['evaluation_policy']; assert policy['minimum_useful_median_TPOT_gain_fraction']==.05
    assert tps_gain>0 and tpot_gain<0
    proof={'project_id':'glm53','decision':'ACCEPTED_INDEPENDENT_FORMAL_NATIVE_AND_TIMING_EVIDENCE','run_id':c.FORMAL_RUN,'measurement_valid':True,'all64_protocol_ownership_SDK_balanced':True,'SDK_TOTAL_column_index':11,'P_actual_new_cache_read_workers':readers,'D_typed_SDMA_destinations':32,'native_write_batches':len(writes),'native_descriptor_bytes':descriptor_bytes,'physical_wire_bytes':None,'native_logical_request_join':None,'all32_eager_bs2_files_rebuilt_and_fields_preserved':True,'graph_ranks_with_positive_within_formal_delta':32,'metric_archive_pairs':248,'metric_archive_HTTP_rows':496,'metric_archive_scope':'Primary reviewed the frozen reread program, actual rc0 wire, identity bindings and returned values; no new remote archive fetch in this local acceptance. Native body hashes/DP counters were independently recomputed by that program.','native_period_scope':'Fresh pre-formal v2 baseline to final postcapture v3, including replacement warmup; excludes original interrupted warmup. Machine audit graph-decision scope wording referring to functional capture is historical wording, not the baseline used.','timing_rederived_from_original_valid_output_chunks':True,'external_model_requests_by_this_review':0,'source_evidence_sha256':dict(SOURCES)}
    proof_path='results/C32-L17-formal-primary-independent-review-v1.json'
    assert not (ROOT/proof_path).exists()
    (ROOT/proof_path).write_text(json.dumps(proof,indent=2)+'\n')
    SOURCES[proof_path]=hashlib.sha256((ROOT/proof_path).read_bytes()).hexdigest()
    decision={'project_id':'glm53','run_id':c.FORMAL_RUN,'decision':'VALID_MEASUREMENT_NO_DECODE_OPTIMIZATION_PROMOTION_TARGET_MISSED','measurement_valid':True,'optimization_accepted':False,'optimization_verdict':'INCONCLUSIVE_FOR_RELIABLE_GAIN_PREDECLARED_TARGET_MISSED','loop_complete':True,'global_optimization_objective_complete':False,'evaluation_policy':policy,'observed_changes_against_L12D':{'TPS_fraction':tps_gain,'median_TPOT_improvement_fraction':tpot_gain,'TTFT_P50_improvement_fraction':1-measurement['quantiles']['ttft_seconds']['P50']/reference['quantiles']['ttft_seconds']['P50']},'vllm_reference_comparison':{'SGLang_TPS_fraction':measurement['output_tokens_per_second']/vllm['output_tokens_per_second']-1,'SGLang_TPOT_P50_fraction':measurement['quantiles']['tpot_seconds']['P50']/vllm['quantiles']['tpot_seconds']['P50']-1},'minimum_useful_TPOT_improvement_met':False,'TPS_no_regression_met':True,'aggregate_SLA_pass':False,'measurement':c.FORMAL_MEASUREMENT,'retained_reference':reference_path,'native_candidate_still_running':True,'rollback_executed':False,'runtime_note':'D36/41 remain the L17 NoCopy candidate; P34/35 and Meta retained. Historical L12D is a reference, not the current D runtime.','interpretation':'One C32 cohort: TPS improved, median TPOT increased. TPS gain alone does not meet the predeclared decode objective. No causal regression, empirical noise estimate, significance or global-optimality claim. NoCopy is not promoted as a successful decode optimization.','actual_new_backend_attempts':320,'qualified_new_backend_inferences':320,'qualified_new_backend_output_tokens':76992,'original_failed_warmup_native_pending':64,'requests_profiles_retries_added_by_acceptance':0,'next_action':'Close L17 evidence and ledger mechanically through ZCode. Further bottleneck work should first resolve existing L15 native AscendCL/API correlation evidence; no new model/profile or rollback admitted by this decision.','source_evidence_sha256':dict(SOURCES)}
    path='results/C32-L17-native-formal-decision-v1.json';assert not (ROOT/path).exists()
    (ROOT/path).write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps({'proof':proof_path,'proof_sha256':SOURCES[proof_path],'decision':path,'decision_sha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),'TPS_fraction':tps_gain,'TPOT_improvement_fraction':tpot_gain,'measurement_valid':True,'optimization_accepted':False}))


if __name__ == '__main__':
    main()
