#!/usr/bin/env python3
"""Compare matched two-repeat official PD runs and retain protected task failures."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--baseline-prefix',default='official-baseline-v2');parser.add_argument('--candidate-prefix',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    def load(prefix):
        rows=[]
        for i in (1,2):
            paths=list(args.evidence.glob(f'node32-{prefix}-r{i}-*/metrics.json'));assert len(paths)==1,(prefix,i,paths)
            p=paths[0];d=json.loads(p.read_text());rows.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':d})
        return rows
    A=load(args.baseline_prefix);B=load(args.candidate_prefix)
    assert all(x['data']['completed'] and x['data']['passed'] for x in A),'Baseline must be complete and valid'
    report={'baseline_sources':[{k:x[k] for k in ('path','sha256')} for x in A],
        'candidate_sources':[{k:x[k] for k in ('path','sha256')} for x in B],'cases':[],
        'scope':'Same stock proxy/workloads/precision/hosts; two repeats each. C4 is six synchronized request waves, not a sustained capacity or production SLO claim. Run percentile means are not pooled percentiles.',
        'candidate_complete':all(x['data']['completed'] and x['data']['passed'] for x in B)}
    if not report['candidate_complete']:
        report.update(decision='reject_or_fix_specific_candidate_failure_continue_optimization',failures=[x['data'].get('failure') for x in B])
    else:
        keyedA=[{c['case']:c for c in x['data']['cases']} for x in A];keyedB=[{c['case']:c for c in x['data']['cases']} for x in B]
        assert all(set(x)==set(keyedA[0]) for x in keyedA+keyedB)
        noise=[];ratios=[];spec={}
        for name in keyedA[0]:
            a=[x[name] for x in keyedA];b=[x[name] for x in keyedB]
            av=sum(x['effective_output_tok_s'] for x in a)/2;bv=sum(x['effective_output_tok_s'] for x in b)/2
            ratio=bv/av;ratios.append(ratio);rr=abs(a[1]['effective_output_tok_s']/a[0]['effective_output_tok_s']-1);noise.append(rr)
            row={'case':name,'baseline_mean_tok_s':av,'candidate_mean_tok_s':bv,'throughput_ratio':ratio,'baseline_repeat_relative_difference':rr,
                'baseline_requests':sum(x['requests'] for x in a),'candidate_requests':sum(x['requests'] for x in b)}
            for kind in ('ttft_ms','tpot_ms'):
                for percentile in ('p50','p95'):
                    va=sum(x[kind][percentile] for x in a)/2;vb=sum(x[kind][percentile] for x in b)/2
                    row[kind+'_'+percentile]={'baseline_mean_run_value':va,'candidate_mean_run_value':vb,'ratio':vb/va}
            report['cases'].append(row)
            for c in b:
                for k,v in c.get('spec_metrics_delta',{}).items():spec[k]=spec.get(k,0)+v
        tasksA=[{c['expression']:c for c in x['data']['task_checks']} for x in A]
        tasksB=[{c['expression']:c for c in x['data']['task_checks']} for x in B]
        protected={expr for expr in tasksA[0] if all(x[expr]['numeric_pass'] for x in tasksA)}
        regressions=[{'expression':expr,'candidate_texts':[x[expr]['response']['choices'][0]['text'] for x in tasksB]}
                     for expr in sorted(protected) if not all(x[expr]['numeric_pass'] for x in tasksB)]
        report.update(baseline_task_scores=[x['data']['task_numeric_passes'] for x in A],candidate_task_scores=[x['data']['task_numeric_passes'] for x in B],
            protected_baseline_tasks=sorted(protected),protected_task_regressions=regressions,
            baseline_known_failing_tasks=[expr for expr in tasksA[0] if expr not in protected],
            throughput_geomean_ratio=math.exp(sum(math.log(x) for x in ratios)/len(ratios)),
            max_baseline_repeat_relative_difference=max(noise),candidate_spec_metrics_delta=spec)
        threshold=1+max(.05,2*max(noise));report['improvement_threshold']=threshold
        report['TTFT_over_5pct_regressions']=[r['case'] for r in report['cases'] if r['ttft_ms_p50']['ratio']>1.05]
        report['TPOT_over_5pct_regressions']=[r['case'] for r in report['cases'] if r['tpot_ms_p50']['ratio']>1.05]
        if regressions:report['decision']='reject_protected_task_regression_continue_next_candidate'
        elif report['throughput_geomean_ratio']<=threshold:report['decision']='no_gain_beyond_repeat_noise_continue_next_candidate'
        elif report['TTFT_over_5pct_regressions'] or report['TPOT_over_5pct_regressions']:report['decision']='measured_gain_with_latency_tradeoff_requires_followup_not_unqualified_adoption'
        else:report['decision']='measured_candidate_gain_continue_path_and_protected_validation'
        report['full_model_accuracy_accepted']=False
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
