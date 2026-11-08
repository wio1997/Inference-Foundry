#!/usr/bin/env python3
"""Portable conditional performance models; only the supplied numeric inputs.

No model tensors, NPU operations or private archive paths are needed. Results
are sensitivities, not new kernel measurements or confidence intervals.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def stats(x):
    return dict(n=len(x),min=min(x),median=statistics.median(x),max=max(x)) if x else dict(n=0)


def gain(up,tau,rho,h):
    # rho is concurrent/original throughput; h excludes this throughput loss.
    return sum(u-max(0.,u/rho-tau)-h for u in up)/1000


def root_tau(up,rho,h,gate_ms):
    if (sum(up)-len(up)*h)/1000<gate_ms:return None
    lo,hi=0.,max(up)/rho
    for _ in range(70):
        mid=(lo+hi)/2
        if gain(up,mid,rho,h)>=gate_ms:hi=mid
        else:lo=mid
    return hi


def run(data):
    windows=data['K03_windows']; ttft=data['reference']['old_TTFT_ms'];batch=data['reference']['old_batch_ms']
    gate=ttft*.01
    assert len(windows)==112 and all(len(w['shared_up_us'])==40 for w in windows)
    k3grid=[]
    for tau in [10.,15.,20.,23.,25.]:
        for rho in [1.,.9,.8]:
            for h in [0.,2.,5.,8.]:
                for count in [20,30,40]:
                    deltas=[gain(w['shared_up_us'][:count],tau,rho,h) for w in windows]
                    percentages=[d/w['period_ms']*100 for d,w in zip(deltas,windows)]
                    k3grid.append(dict(tau_us=tau,rho=rho,h_us=h,eligible_layers=count,
                        saved_period_ms=stats(deltas),percent_of_same_period=stats(percentages),
                        windows_at_1pct=sum(d>=w['period_ms']*.01 for d,w in zip(deltas,windows))))
    thresholds=[]
    for rho in [1.,.9,.8]:
        for h in [0.,2.,5.]:
            roots=[root_tau(w['shared_up_us'],rho,h,w['period_ms']*.01) for w in windows]
            thresholds.append(dict(rho=rho,h_us=h,required_tau_us=stats([x for x in roots if x is not None])))
    caps=[]
    for count in [20,30,40]:
        values=[(sum(sorted(w['shared_up_us'],reverse=True)[:count])-count*5)/1000 for w in windows]
        caps.append(dict(eligible_layers=count,h_us=5,selection='optimistic largest Up durations; tau unlimited',
            saved_period_ms=stats(values),percent_of_same_period=stats([x/w['period_ms']*100 for x,w in zip(values,windows)]),
            windows_at_1pct=sum(x>=w['period_ms']*.01 for x,w in zip(values,windows))))
    required_coverage=[next((n for n in range(1,41) if gain(w['shared_up_us'][:n],20,1,5)>=w['period_ms']*.01),None) for w in windows]
    required_rho=[]
    for w in windows:
        up=w['shared_up_us'];assert min(up)>20
        # Exact here because tau20 is below each u/rho for rho<=1.
        required_rho.append(sum(up)/(sum(up)+40*(20-5)-w['period_ms']*.01*1000))
    ordinary=[]
    for req in data.get('ordinary_requests',[]):
        rows=[]
        for ref in req['references']:
            n=max(ref['graph_steps'])
            rows.append(dict(generation=ref['generation'],reference_response_ms=ref['elapsed_ms'],optimistic_graph_count=n,
                required_saved_ms_per_graph_for_response_1pct=ref['elapsed_ms']*.01/n,
                hypothetical_decode_saved_ms_at_point6=.6*n,
                hypothetical_response_percent_at_point6=.6*n/ref['elapsed_ms']*100))
        ordinary.append(dict(name=req['name'],references=rows))
    k3=dict(formula='sum[u_i-max(0,u_i/rho-tau_i)-h_i]/1000 ms; min(u_i,tau_i)-h_i at rho=1',
        definitions=dict(rho='concurrent/original shared-Up throughput',h='net extra critical overhead EXCLUDING Up throughput loss already in rho',
            tau='usable interval after all required routed Cube writes retire, before Vector critical tail finishes',
            eligibility='scenario uses first N layer positions, not measured tail availability; cap uses best N as optimistic screen'),
        grid=k3grid,tail_thresholds=thresholds,full_hiding_caps=caps,
        required_rho_at_tau20_h5=dict(required_ratio=stats(required_rho),windows_feasible_rho_at_most_one=sum(x<=1 for x in required_rho)),
        required_layer_count_at_tau20_rho1_h5=dict(feasible_count=stats([x for x in required_coverage if x is not None]),
            impossible_with_40=sum(x is None for x in required_coverage),
            arithmetic_count_including_counts_above40=stats([math.ceil(w['period_ms']*.01*1000/15) for w in windows])),
        old_batch_critical_step_gate_at_point6=math.ceil(batch*.01/.6),
        old_batch_count_scenarios=[dict(critical_steps=n,saved_ms=.6*n,reference_batch_pct=.6*n/batch*100) for n in [200,223,250,310]],
        ordinary_decode_sensitivities=ordinary,
        ordinary_limit='Assumes the point6ms saving transfers to every counted target graph and is fully critical. No prefill credit or DP summation. This transfer is not observed; graph counts/response endpoints are older cohorts.')
    carriers=[]
    for p in data['K01_carriers']:
        old=p['original_AG_us'];alpha=p['alpha_reference_us'];narrow=alpha+(old-alpha)*.1;dual=alpha+(old-alpha)*.2
        comp=38*(p['original_global_BF16_projection_us']-p['new_local_projection_model_us'])/1000
        comm=(36*(old-narrow)+(old-dual))/1000
        carriers.append(dict(dp=p['dp'],carrier=p['carrier'],global_rows=p['global_rows'],local_rows=p['local_rows'],
            original_AG_us=old,narrow_AG_model_us=narrow,dual_AG_model_us=dual,
            original_BF16_projection_us=p['original_global_BF16_projection_us'],local_projection_model_us=p['new_local_projection_model_us'],
            compute_delta_ms=comp,transfer_delta_ms=comm,total_delta_ms=comp+comm))
    compute=sum(x['compute_delta_ms'] for x in carriers);transfer=sum(x['transfer_delta_ms'] for x in carriers);gross=compute+transfer
    k1grid=[]
    for fc,fm in [(v,v) for v in [.25,.5,.75,1.]]+[(0.,.5),(.5,0.),(.5,.25)]:
        for extra in [0.,5.,10.,15.]:
            saved=fc*compute+fm*transfer-extra
            k1grid.append(dict(compute_exposure=fc,transfer_exposure=fm,net_extra_critical_ms=extra,saved_envelope_ms=saved,
                fraction_of_old_TTFT_scale_pct=saved/ttft*100,fraction_of_old_batch_scale_pct=saved/batch*100,at_1pct_TTFT_scale=saved>=gate))
    k1=dict(formula='f_compute*compute_delta + f_transfer*transfer_delta - c_extra',carriers=carriers,
        compute_delta_ms=compute,transfer_delta_ms=transfer,gross_delta_ms=gross,scenarios=k1grid,
        net_uniform_fraction_needed_before_extra=gate/gross,
        cost_room_ms_at_half_exposure=gross*.5-gate,
        cost_room_us_per_148_communication_sites_at_half_exposure=(gross*.5-gate)*1000/148,
        interpretation='Owner placement saves duplicate projections and wide AG. Fusion-only increment over an equivalent multi-kernel owner implementation is unmeasured and must not receive this same budget twice.',
        extra_cost_definition='c is unpriced net critical packing/deadline/tiling/contending overhead above modelled local GEMM and narrow-AG replacement; do not subtract those replacement costs twice.')
    bases=[];k2grid=[]
    for name,s in data['K02_direct_scenarios'].items():
        vals=s['results']['TTFT_P90'];ref=vals[1]
        bases.append(dict(name=name,rates=s['rates'],reference_TTFT_saved_ms=vals,reference_middle_ms=ref,
            extra_net_penalty_room_for_1pct_ms=ref-gate,reference_batch_saved_ms=s['results']['makespan']))
        for penalty in [0.,.5,1.,2.,3.]:
            saved=ref-penalty
            k2grid.append(dict(base=name,net_incremental_OA_penalty_ms=penalty,conditional_saved_ms=saved,
                fraction_of_old_TTFT_scale_pct=saved/ttft*100,at_1pct_TTFT_scale=saved>=gate))
    k2=dict(formula='S_joint = S_direct_reference - delta_OA_net_critical; delta includes avoided old transport/layout, new work and interference exactly once',
        direct_reference_bases=bases,scenarios=k2grid,output_bytes_per_query=dict(before=65536,after=16384),
        owner_weight_extra_MiB=dict(all40=2240,compressed38_only=2128),
        retained_OA_gate=data['K02_OA_gate'],K256_rate_thresholds=data['K02_rate_thresholds'],
        interpretation='No aligned joint direct/OA implementation timing exists. This is an incremental penalty budget, not a claimed direct+OA point forecast. The OA gate from a separate cohort is not added to the direct reference.')
    return dict(status='CONDITIONAL_OFFLINE_PERFORMANCE_ANALYSIS',schema_version=1,scope=data['scope'],new_NPU_measurements=0,
        reference=data['reference'],K03=k3,K01=k1,K02=k2,
        limits=data['limits']+['Scenario grids are not device parameter searches or performance trials.','Percentages are elapsed-time reduction ratios, not reciprocal throughput speedups.','No new end-to-end P90 was recomputed; no independent gains are added.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run(json.loads(a.inputs.read_text()));r['inputs_sha256']=hashlib.sha256(a.inputs.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    case=lambda tau,rho,h,n:next(x for x in r['K03']['grid'] if (x['tau_us'],x['rho'],x['h_us'],x['eligible_layers'])==(tau,rho,h,n))
    print(json.dumps(dict(K03=[case(20,rho,5,40) for rho in [1.,.9,.8]],K03_rho=r['K03']['required_rho_at_tau20_h5'],
      K03_coverage=r['K03']['required_layer_count_at_tau20_rho1_h5'],K03_caps=r['K03']['full_hiding_caps'],
      K03_tau=[x for x in r['K03']['tail_thresholds'] if x['h_us']==5],
      K01={k:r['K01'][k] for k in ['compute_delta_ms','transfer_delta_ms','gross_delta_ms','cost_room_ms_at_half_exposure']},
      K02_bases=r['K02']['direct_reference_bases']),ensure_ascii=False))


if __name__=='__main__':main()
