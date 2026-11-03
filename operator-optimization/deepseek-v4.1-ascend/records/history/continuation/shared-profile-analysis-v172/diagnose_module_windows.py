"""One CPU pass over cached factorized executions; no new device workload."""
import collections
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path

A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis')
O=A/'joint172'
# Reuse reviewed definitions, without re-running attribution or opening source.
exec((O/'classify_and_rank_modules.py').read_text().split('started=time.time();')[0])
DIM={x['profile']:x['maps'] for x in json.loads((O/'BLOCK-DIMENSION-DECODE.json').read_text())['profiles']}
started=time.time()
stage_modules={};stage_optional={};stage_profiles=collections.defaultdict(list)
stage_counts=collections.defaultdict(collections.Counter)
details={};work={};frame_reasons=collections.Counter();unowned_by_op=collections.Counter()
stage_intervals={};pins=[]
for p in sorted((O/'profiles').glob('*.json')):
    d=json.loads(p.read_text());ap=O/'assignments'/(d['profile']+'.json')
    ad=json.loads(ap.read_text());assign={x[0]:x for x in ad['rows']}
    # Two separately exported formal windows; names encode export-start cluster.
    stamp=re.search(r'_(20261102\d{9})_ascend_pt$',d['profile']).group(1)
    stage=stamp[:14]
    if stage not in stage_modules:
        stage_modules[stage]={m:stats() for m in DEVICE_MODULES}
        stage_optional[stage]={m:{e:[] for e in ENGINES} for m in DEVICE_MODULES}
        stage_intervals[stage]={}
    stage_profiles[stage].append(d['profile']);all_intervals=[]
    for nr in d['native_rows']:
        row=assign[nr[0]];owner,cands,reason,dummy=row[1:]
        r=d['regimes'][nr[9]];metrics=dict(zip(d['metric_names'],nr[11])) if nr[11] is not None else {}
        stage_counts[stage]['native']+=1;stage_counts[stage]['PMU']+=int(bool(metrics))
        stage_counts[stage]['certain' if owner else 'ambiguous']+=1
        stage_counts[stage]['runtime_dummy_connected']+=int(dummy)
        all_intervals.append((nr[2],nr[3]))
        if owner:
            update(stage_modules[stage][owner],nr,metrics,reason)
            if owner in ['M12','M23','M16']:
                key=(stage,owner,r['opType'],r['taskType'],DIM[d['profile']].get('blockDim',{}).get(str(r.get('blockDim'))))
                if key not in details:details[key]={'stats':stats(),'durations_ns':[],'shapes':collections.Counter(),'native_names':collections.Counter()}
                x=details[key];update(x['stats'],nr,metrics,reason);x['durations_ns'].append(nr[3]-nr[2]);x['shapes'][r['inputShapes']]+=1;x['native_names'][r['name']]+=1
            w=matrix_work(r)
            if w:
                key=(stage,owner,w['mode'])
                if key not in work:work[key]={'executions':0,'conditional_MAC':0,'matched_kernel_sum_ns':0,'regimes':collections.Counter()}
                x=work[key];x['executions']+=1;x['conditional_MAC']+=w['MAC'];x['matched_kernel_sum_ns']+=nr[3]-nr[2];x['regimes'][r['opType']+' | '+r['inputShapes']+' -> '+r['outputShapes']]+=1
        else:
            unowned_by_op[(stage,r['opType'],r['taskType'])]+=1
            for e,(cyc,rat) in ENGINES.items():
                c=metrics.get(cyc);v=metrics.get(rat)
                if c is not None and c>0 and v is not None and 0<=v<=1:
                    for m in cands:stage_optional[stage][m][e].append((c,v))
        if reason.startswith('MARKOV') and len(nr[10])==1:
            cpu=d['CPU_rows'][str(nr[10][0])]
            fs=[d['source_frame_strings'][str(x)] for x in d['callchains'].get(str(cpu['callchain_id']),[])]
            for f in fs:
                if '_run_merged_draft' in f:frame_reasons[(f,r['opType'])]+=1;break
    stage_intervals[stage][d['profile']]={'all_COMPUTE_interval_union_ns':union(all_intervals),'first_native_start_ns':min(a for a,b in all_intervals),'last_native_end_ns':max(b for a,b in all_intervals)}
    pins.append({'input':str(p),'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'assignment':str(ap),'assignment_sha256':hashlib.sha256(ap.read_bytes()).hexdigest()})

groups={}
for stage,mods in stage_modules.items():
    for m,s in mods.items():
        finish(s)
        s['ownership_bounds']={e:membership_bounds(s['ratios'][rat]['numerator'],s['ratios'][rat]['denominator'],stage_optional[stage][m][e]) for e,(_,rat) in ENGINES.items()}
    ranks={}
    for e,(_,rat) in ENGINES.items():
        rs=[{'module':m,**s['ratios'][rat],'ownership_bounds':s['ownership_bounds'][e]} for m,s in mods.items() if s['ratios'][rat]['denominator']>0 and (e!='AIC_MAC' or m in MATRIX)]
        ranks[e]=sorted(rs,key=lambda x:x['fraction'])
    ranks['AIC_MAC_minimum_stable_under_recorded_ownership_ambiguity']=bool(ranks['AIC_MAC']) and all(ranks['AIC_MAC'][0]['ownership_bounds']['upper']<x['ownership_bounds']['lower'] for x in ranks['AIC_MAC'][1:])
    groups[stage]={'profiles':stage_profiles[stage],'counts':dict(stage_counts[stage]),'modules':mods,'rankings':ranks,'native_window_observations':stage_intervals[stage]}

ds=[]
for key,x in details.items():
    t=sorted(x.pop('durations_ns'));finish(x['stats']);x['shapes']=dict(x['shapes']);x['native_names']=dict(x['native_names'])
    ds.append({'window_cluster':key[0],'module':key[1],'opType':key[2],'taskType':key[3],'blockDim_decoded_export_label':key[4],**x,'duration_ns':{'min':t[0],'p50':statistics.median(t),'p90_order':t[min(len(t)-1,math.ceil(.9*len(t))-1)],'max':t[-1]}})
ws=[]
rates={'INT8':8192,'BF16':4096,'FP16':4096}
for k,x in work.items():
    x['regimes']=dict(x['regimes']);r=rates.get(k[2])
    c=x['conditional_MAC']/(24*r) if r else None
    ns=c/1.8 if c is not None else None
    ws.append({'window_cluster':k[0],'module':k[1],'mode':k[2],**x,'conditional_device_equivalent_cycle_floor':c,'rated1800MHz_reference_floor_ns':ns,'reference_floor_over_matched_kernel_sum':ns/x['matched_kernel_sum_ns'] if ns is not None else None,'status':'CONDITIONAL_DIRECT_DENSE_SHAPE_DOMAIN','guards':['A operand interpreted as untransposed batch,M,K; output batch,M,N','Classical direct-dense algorithm with all MACs necessary; no sparse/alternative algorithm claim','Reported dtype matches actual Cube mode; FLOAT mode unresolved','Rate per current arch2201 core,24 Cube cores;1800MHz rated reference is not physical fmax','Same selected execution population in numerator and matched cumulative kernel time; not request walltime or fused module bound']})
out={'status':'TWO_WINDOW_REPORTED_ACTIVITY_AND_CAUSAL_EVIDENCE_REVIEW','groups':groups,'selected_module_operator_diagnostics':ds,'conditional_matrix_work_comparisons':ws,'Markov_exact_callsite_inventory':[{'frame':k[0],'opType':k[1],'executions':v} for k,v in frame_reasons.items()],'ambiguous_ownership_op_counts':[{'window':k[0],'op':k[1],'resource':k[2],'executions':v} for k,v in unowned_by_op.items()],'inputs':pins,'elapsed_seconds':time.time()-started,'limits':['Two existing v164 CPU+PMU formal exports only; no v168 receiver counts joined','Reported pipe activity fractions are core-cycle weighted, not chip utilization/MFU or speedup','Ambiguity bounds include recorded positive-cycle rows only; missing counters and unrecorded work not bounded','COMPUTE interval union is observed kernel residency, not full-device busy time','All exported blockDim labels decode to N/A; raw STRING_IDS keys cannot be used as launched block/core counts','Existing recorder may perturb execution; no profiler-free module ranking proved'],'new_device_work':False,'overall_complete':False}
(O/'MODULE-WINDOW-DIAGNOSIS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'elapsed_seconds':out['elapsed_seconds'],'windows':{s:{'counts':g['counts'],'AIC_minimum':g['rankings']['AIC_MAC'][0],'AIC_minimum_robust':g['rankings']['AIC_MAC_minimum_stable_under_recorded_ownership_ambiguity']} for s,g in groups.items()}},ensure_ascii=False))
