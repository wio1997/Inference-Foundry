from pathlib import Path
import hashlib,json,subprocess,sys
r=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(r/'analyze_cumsum_i32_formal.py')],check=True,capture_output=True,text=True)
p=r/'cumsum-i32-formal-verdict-v1.json';x=json.loads(p.read_text());c=r/'cumsum-i32-formal-contract-v1.json';contract=json.loads(c.read_text());quality=json.loads((r/'cumsum-i32-quality-reviewed.json').read_text());control=json.loads((r/'cumsum-i32-quality-control-summary.json').read_text())
assert contract['formal_design']['C4_formal_requests']==384 and quality['proceed_finite_formal'] and control['completed'] and control['numeric_all'] and control['rows']==28 and all(z['original_i32_in_native_set'] and z['current_i32_in_native_set'] for z in control['controls'])
ratios={k:x['C4_pooled']['adaptive'][k]['p95_linear']/x['C4_pooled']['native'][k]['p95_linear'] for k in ['TTFT_ms','last_token_ms','mean_TPOT_ms']}
gates={'C4_aggregate_throughput_improved':x['C4_candidate_over_native']>1,'all3_C4_cycles_improved':all(z['candidate_over_native']>1 for z in x['C4_cycles']),'C4_all_pooled_p95_nonregressing':all(z<=1 for z in ratios.values()),'C1_TTFT_and_TPOT_nonregressing':all(x['C1_equal_case_ratios'][k]<=1 for k in ['TTFT_ms','mean_TPOT_ms']),'cancel_and_peer_reuse_complete':x['cancel']['completed'] and x['cancel']['client_close_requests']==10 and x['cancel']['full_reuse_and_peer_responses']==16}
assert gates['cancel_and_peer_reuse_complete']
x['decision']='ACCEPT_FINITE_JOINT_POINT_CUMSUM_I32' if all(gates.values()) else 'MIXED_DEFAULT_OFF_CUMSUM_I32_NO_JOINT_GAIN' if x['C4_candidate_over_native']>1 else 'REJECT_DEFAULT_OFF_CUMSUM_I32'
x.update(frozen_contract_sha256=hashlib.sha256(c.read_bytes()).hexdigest(),predeclared_joint_gates=gates,C4_p95_candidate_over_native=ratios,finite_quality_review=quality,C1_native_counter_accounting=json.loads((r/'cumsum-i32-C1-counter-summary-v1.json').read_text()),adoption_scope='Only this finite same-load point when every predeclared gate passes. Default helper env remains off; no generic quality/capacity, global optimum, single-cause or cross-window combined-gain claim.')
(r/'cumsum-i32-reviewed-verdict-v1.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps({k:x[k] for k in ['decision','predeclared_joint_gates','C1_equal_case_ratios','C4_candidate_over_native','C4_cycles','C4_p95_candidate_over_native','cancel']},indent=2))
