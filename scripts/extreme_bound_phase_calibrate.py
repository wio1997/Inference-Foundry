#!/usr/bin/env python3
"""Calibrate cohort phase fractions and trajectory coupling for the V0 replay."""
import argparse,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
 x=load('evidence/20260926_loop059_boundary/run239/phase_analysis.json')
 waves=x['waves'];assert len(waves)==4 and x['bench_summary']['success']==48
 labels=list(waves[0]['phases'])
 totals={k:sum(w['phases'][k] for w in waves) for k in labels}
 current=x['bench_summary']['duration_s']
 cycles=sum(w['cycles'] for w in waves)
 latest_decode=sum(max(w['decode_rank_wall_s']) for w in waves)
 cycle_s=latest_decode/cycles
 # Marginal response at the observed Run239 trajectory, never a feasible bound.
 def counterfactual(prefill_save_s_per_cohort=0,target_save_ms_per_cycle=0,added_cycles_per_cohort=0):
  added=4*added_cycles_per_cohort
  duration=current-4*prefill_save_s_per_cohort-(cycles+added)*target_save_ms_per_cycle/1000+added*cycle_s
  return dict(duration_s=duration,tps=49152/duration,added_cycle_cost_s=added*cycle_s)
 r188=load('evidence/20260925_loop048_prefill/run188/analysis.json')
 p=r188['phase_summary'];control=(p['A']['client_envelope_s']+p['A2']['client_envelope_s'])/2
 gross_prefill=(p['A']['max_rank_prefill_forward_wall_sum_s']+p['A2']['max_rank_prefill_forward_wall_sum_s'])/2-p['B']['max_rank_prefill_forward_wall_sum_s']
 extra_runtime=p['B']['latest_rank_runtime_wall_s']-(p['A']['latest_rank_runtime_wall_s']+p['A2']['latest_rank_runtime_wall_s'])/2
 r188_check=dict(control_mean_client_s=control,B_client_s=p['B']['client_envelope_s'],observed_delta_s=p['B']['client_envelope_s']-control,
                 gross_prefill_forward_reduction_s=gross_prefill,decode_cycle_delta=p['B']['decode_cycles']-p['A']['decode_cycles'],
                 runtime_wall_increase_s=extra_runtime,naive_prefill_minus_runtime_delta_s=-gross_prefill+extra_runtime,
                 limitation='Different prefill shapes/routes/acceptance and asynchronous overlap; naive sum is not a causal prediction.')
 out=dict(run239=dict(duration_s=current,tps=x['bench_summary']['output_tps'],cycles=cycles,
                      phase_totals_s=totals,phase_fraction_of_client={k:v/current for k,v in totals.items()},
                      latest_rank_decode_wall_s=latest_decode,empirical_decode_wall_per_cycle_ms=cycle_s*1000,
                      wave_prefill_s=[w['phases']['first_execute_to_handoff_s'] for w in waves],
                      wave_serve_s=[w['phases']['serve_start_to_serve_end_s'] for w in waves]),
          marginal_sensitivity=dict(prefill_save_0_1s_per_cohort=counterfactual(.1),
                                    target_save_1ms_per_cycle=counterfactual(0,1),
                                    prefill_save_0_5s_per_cohort_fixed_cycles=counterfactual(.5),
                                    prefill_save_0_5s_per_cohort_plus_8cycles=counterfactual(.5,0,8)),
          run188_admission_coupling_check=r188_check,
          limits=['One instrumented Run239 pass calibrates phase proportions, not a stable distribution or removable work.',
                  'Empirical decode wall/cycle is a trajectory-specific marginal approximation; changed routes, cache state and concurrency can invalidate it.',
                  'Run188 gross prefill reduction is not exposed E2E savings; additional cycles and overlap almost cancel it.',
                  'Product communication bound remains unknown after Run240 path mismatch.'])
 Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:out[k] for k in ('run239','marginal_sensitivity','run188_admission_coupling_check')},indent=2))
if __name__=='__main__':main()
