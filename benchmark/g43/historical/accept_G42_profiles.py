"""Accept only completed G35 phases using original raw inputs and SQLite times."""
from pathlib import Path
import argparse,datetime,hashlib,json
from accept_completed_native_profiles import timestamps
from analyze_aisbench_results import analyze

ap=argparse.ArgumentParser();ap.add_argument('run_id');a=ap.parse_args()
assert Path(a.run_id).name==a.run_id and a.run_id.endswith('_ascops42_full')
r=Path(__file__).resolve().parent;root=r/'results/sglang'/a.run_id
report=json.loads((root/'completed_phase_verification.json').read_text())
assert report['engine']=='sglang' and report['formal_profiles_verified']>0
gate=json.loads((r/'service_gate_sglang_ascops42.json').read_text())
assert gate['runtime_payloads_SHA256_verified']==60 and gate['trained_Markov_weights_loaded_all16ranks'] is True and gate['Engram_gate_runtime']=='G33FP32chunk1024'
assert gate['actual_min_full_KV_tokens_per_DP']>=1052672 and gate['engram_placement']=='cpu'
component=json.loads((r/'sglang_G42_fused_RMS_component_acceptance.json').read_text())
assert component['cases']==93 and all(component['checks'].values())
assert gate['mHC_statistics_runtime']=='G42fusedRMS_FP32projection'
assert gate['actual_file_bind_mounts_verified']['/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4_mhc.py']==component['source_sha256']['candidate']
cfg=json.loads((root/'server_info_before.json').read_text())
assert cfg['moe_a2a_backend']==cfg['speculative_moe_a2a_backend']=='ascend_ops'
assert cfg['max_running_requests']==16 and cfg['context_length']>=263168
base=json.loads((r/'native_vllm_baseline_acceptance.json').read_text())
native={x['input_tokens']:x for x in base['metrics']}
repeat=json.loads((r/'native_vllm_32K_repeat_acceptance.json').read_text())['repeat32K_metrics']
accepted=[]
for metric in report['formal_metrics']:
 length=metric['input_tokens'];assert metric==analyze(root,length)
 phases=[x for x in report['phase_checks'] if x['input_tokens']==length]
 assert {x['phase'] for x in phases}=={'warmup','formal'}
 detail=None;formal=None
 for check in phases:
  p=root/check['details'];assert hashlib.sha256(p.read_bytes()).hexdigest()==check['details_sha256']
  rows=[json.loads(s) for s in p.read_text().splitlines() if s]
  count=16 if check['phase']=='warmup' else 32
  assert len(rows)==count and all(x['success'] and x['input_tokens']==length and x['output_tokens']==1024 for x in rows)
  expected=[json.loads(s)['question'] for s in (r/f"datasets/{check['phase']}_input_{length}.jsonl").read_text().splitlines() if s]
  assert sorted(x['input'] for x in rows)==sorted(expected)
  if check['phase']=='formal':detail,formal=p,rows
 timeline=timestamps(detail,formal,metric)
 compare={'native_original_TPS_ratio':metric['aggregate_output_tokens_per_s']/native[length]['aggregate_output_tokens_per_s'],
          'latency_ratios':{k:metric[k]/native[length][k] for k in ['ttft_mean_ms','ttft_p99_ms','tpot_mean_ms','tpot_p99_ms','e2el_mean_ms','e2el_p99_ms']}}
 if length==32768:compare['native_same_instance_repeat_TPS_ratio']=metric['aggregate_output_tokens_per_s']/repeat['aggregate_output_tokens_per_s']
 accepted.append({'input_tokens':length,'metrics':metric,'SQLite_timestamp_contract':timeline,'comparison':compare})
samples=[json.loads(s) for s in (root/'npu_ownership_samples.jsonl').read_text().splitlines() if s]
assert samples and all(not x['foreign'] and x['own_npu_process_count']>=16 for x in samples)
all4=report['all_four_profiles_complete'];assert all4==(len(accepted)==4)
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'166_G42_all4_raw_accepted' if all4 else '166_G42_partial_raw_accepted','run_id':a.run_id,
        'hardware':base['hardware'],'image_id':gate.get('image_id','sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517'),
        'container_id':gate['container_id'],'runtime_manifest':gate['runtime_manifest'],'profiles':accepted,'all4_SGLang_comparison_complete':all4,
        'observed_all4_primary_metric_higher_than_native':all4 and all(x['comparison']['native_original_TPS_ratio']>1 for x in accepted),
        'repeatability_verified':False,'whole_model_quality_or_numeric_equivalence_verified':False,'outperformance_goal_achieved':False,
        'scope':'Exact original input/output/C16, success/SHA and actual SQLite timestamps. Higher observed TPS alone does not establish repeatable outperformance or quality.'}
dest=r/('sglang_G42_all4_acceptance.json' if all4 else 'sglang_G42_partial_acceptance.json')
if all4:assert not dest.exists()
temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(dest)
print(json.dumps({'status':result['status'],'profiles':[{'input_tokens':x['input_tokens'],'TPS':x['metrics']['aggregate_output_tokens_per_s'],'comparison':x['comparison']} for x in accepted]}))
