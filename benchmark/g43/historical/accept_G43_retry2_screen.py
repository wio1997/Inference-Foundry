"""One measured166 G33 screen, matched to both preserved native32K observations."""
from pathlib import Path
import argparse, datetime, hashlib, json
from accept_completed_native_profiles import timestamps
from analyze_aisbench_results import analyze


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('run_id'); args = ap.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id.endswith('_ascops43r2_screen')
    r = Path(__file__).resolve().parent; root = r / 'results/sglang' / args.run_id
    report = json.loads((root / 'completed_phase_verification.json').read_text())
    assert report['completed_phases_verified'] == 2 and report['formal_profiles_verified'] == 1
    measured = analyze(root, 32768); assert measured == report['formal_metrics'][0]
    formal_rows = None; detail_path = None
    for check in report['phase_checks']:
        path = root / check['details']; assert hashlib.sha256(path.read_bytes()).hexdigest() == check['details_sha256']
        rows = [json.loads(s) for s in path.read_text().splitlines() if s]
        count = 16 if check['phase'] == 'warmup' else 32
        assert len(rows) == count and all(x['success'] and x['input_tokens'] == 32768 and x['output_tokens'] == 1024 for x in rows)
        expected = [json.loads(s)['question'] for s in (r / f"datasets/{check['phase']}_input_32768.jsonl").read_text().splitlines() if s]
        assert sorted(x['input'] for x in rows) == sorted(expected)
        if check['phase'] == 'formal': formal_rows, detail_path = rows, path
    timeline = timestamps(detail_path, formal_rows, measured)
    samples = [json.loads(s) for s in (root / 'npu_ownership_samples.jsonl').read_text().splitlines() if s]
    assert samples and all(not x['foreign'] and x['own_npu_process_count'] >= 16 for x in samples)
    gate = json.loads((r / 'service_gate_sglang_ascops43r2.json').read_text())
    assert gate['Engram_gate_runtime'] == 'G33FP32chunk1024' and gate['runtime_payloads_SHA256_verified'] == 62 and gate['trained_Markov_weights_loaded_all16ranks'] is True
    component = json.loads((r / 'sglang_G42_fused_RMS_component_acceptance.json').read_text())
    assert all(component['checks'].values()) and component['cases'] == 93
    assert gate['mHC_statistics_runtime'] == 'G42fusedRMS_FP32projection'
    assert gate['actual_file_bind_mounts_verified']['/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4_mhc.py'] == component['source_sha256']['candidate']
    log = (r / 'sglang_ascops43r2/server.log').read_text()
    assert log.count('A3_V41_HC_ROW_SHARD rank=') >= 16
    assert gate['mHC_row_layout_runtime'] == 'G43targetHC_MoE_rowsTP4_full_attention'
    row_proof = json.loads((r / 'sglang_G43_HC_row_component_acceptance.json').read_text())
    assert len(row_proof['ranks']) == 16
    assert row_proof['helper_sha256'] == hashlib.sha256((r/'compat/runtime/g43/hc_row_shard.py').read_bytes()).hexdigest()
    assert gate['actual_min_full_KV_tokens_per_DP'] >= 1052672 and gate['engram_placement'] == 'cpu'
    source_cfg = json.loads((root / 'server_info_before.json').read_text())
    assert source_cfg['context_length'] >= 263168 and source_cfg['max_running_requests'] == 16
    native = json.loads((r / 'native_vllm_baseline_acceptance.json').read_text())
    original = next(x for x in native['metrics'] if x['input_tokens'] == 32768)
    repeated = json.loads((r / 'native_vllm_32K_repeat_acceptance.json').read_text())['repeat32K_metrics']
    baseline = json.loads((r / 'sglang_G37_all4_acceptance.json').read_text())
    assert baseline['all4_SGLang_comparison_complete'] is True
    corrected = next(x['metrics'] for x in baseline['profiles'] if x['input_tokens'] == 32768)
    g42 = json.loads((r/'sglang_G42_all4_acceptance.json').read_text())
    last = next(x['metrics'] for x in g42['profiles'] if x['input_tokens'] == 32768)
    comparisons = {}
    for name, baseline in [('native_original', original), ('native_same_instance_repeat', repeated), ('G37_corrected_baseline', corrected), ('G42_last_full32K', last)]:
        comparisons[name] = {'output_TPS_ratio': measured['aggregate_output_tokens_per_s'] / baseline['aggregate_output_tokens_per_s'],
                             'latency_ratios': {k: measured[k] / baseline[k] for k in
                                                ['ttft_mean_ms', 'ttft_p99_ms', 'tpot_mean_ms', 'tpot_p99_ms', 'e2el_mean_ms', 'e2el_p99_ms']}}
    result = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': '166_G43_retry2_32K_C16_screen_raw_accepted', 'run_id': args.run_id,
              'hardware': native['hardware'], 'image_id': 'sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517',
              'container_id': gate['container_id'], 'runtime_manifest': gate['runtime_manifest'],
              'metrics': measured, 'SQLite_timestamp_contract': timeline,
              'ownership_samples_without_foreign_NPU_process': len(samples),
              'comparison_to_both_native32K_runs': comparisons,
              'whole_model_quality_or_numeric_equivalence_verified': False,
              'screen_repeatability_verified': False, 'fused_RMS_component_strict_gate_passed': True, 'HC_row_component_strict_gate_passed': True, 'HC_row_component_proof':'sglang_G43_HC_row_component_acceptance.json', 'component_proof': 'sglang_G42_fused_RMS_component_acceptance.json', 'all4_SGLang_comparison_complete': False,
              'scope': 'One32K screen with exact original input/output/C16 and full KV. '
                       'No model quality, repeatability or all4 outperformance claim.'}
    dest = r / 'sglang_G43_retry2_screen_acceptance.json'; assert not dest.exists()
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'TPS': measured['aggregate_output_tokens_per_s'],
                      'TTFT_mean_ms': measured['ttft_mean_ms'], 'TPOT_mean_ms': measured['tpot_mean_ms'],
                      'vs_both_native32K': comparisons}))


if __name__ == '__main__': main()
