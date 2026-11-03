"""Profile-first operator chain cost/traffic model; no model/framework execution."""
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter,defaultdict
import json,statistics,hashlib
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE


def stats(xs):
    a=sorted(xs);return dict(count=len(a),sum_ns=sum(a),minimum_ns=a[0],median_ns=statistics.median(a),p90_ns=a[int(.9*(len(a)-1))],maximum_ns=a[-1])


def dims(text):
    return [[int(x)for x in s.split(',')if x]for s in text.strip('"').split(';')]


def union(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        if end is None or a>=end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


def main():
    assert not(O/'operator-fusion-profile-v55.json').exists()
    graph=read(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json')
    bydb=defaultdict(list)
    for g in graph['graph_executions']:bydb[g['database_index']].append(g)
    op_durations=defaultdict(list);regimes=defaultdict(list);chains=[];norms=[];pmu_selected=0
    for dbi in range(32):
        native=read(CACHE/f'analysis/native-execution-identity-v14/db-{dbi:02d}.json')
        defs={r['globalTaskId']:r for r in native['COMPUTE_definitions']}
        tasks={r['task_record_rowid']:r for r in native['TASK_records']}
        select=set()
        for t in tasks.values():
            d=defs.get(t['globalTaskId'])
            if not d:continue
            op=d['opType_text'];dur=t['endNs']-t['startNs'];assert dur>=0
            op_durations[op].append(dur)
            if op=='RmsNormCast':
                norms.append(dict(database_index=dbi,task_rowid=t['task_record_rowid'],duration_ns=dur,blockNum=d['blockNum'],inputShapes=d['inputShapes_text'],inputDtypes=d['inputDataTypes_text'],outputDtypes=d['outputDataTypes_text']))
        for g in bydb[dbi]:
            for block in g['blocks']:
                selected={}
                for tid in block['task_record_rowids']:
                    t=tasks[tid];d=defs[t['globalTaskId']];op=d['opType_text']
                    if op in ['RmsNormCast','MatMulV3','MoeGatingTopKHash']:
                        if op=='MatMulV3' and d['outputShapes_text']!='"3,384"':continue
                        if op=='MatMulV3':assert d['inputShapes_text']=='"3,5120;384,5120"' and d['inputDataTypes_text']=='FLOAT;FLOAT'
                        assert op not in selected
                        selected[op]=dict(task_rowid=tid,globalTaskId=t['globalTaskId'],startNs=t['startNs'],endNs=t['endNs'],duration_ns=t['endNs']-t['startNs'],blockNum=d['blockNum'],mixBlockNum=d['mixBlockNum'],inputShapes=d['inputShapes_text'],inputDtypes=d['inputDataTypes_text'],outputShapes=d['outputShapes_text'],outputDtypes=d['outputDataTypes_text'])
                        select.add(tid)
                assert len(selected)==3
                for op,s in selected.items():regimes[(op,s['inputShapes'],s['inputDtypes'],s['blockNum'])].append(s['duration_ns'])
                ints=[(s['startNs'],s['endNs'])for s in selected.values()]
                chains.append(dict(database_index=dbi,graph_ordinal=g['graph_ordinal'],block_ordinal=block['block_ordinal'],
                    exact_marker_count=g['exact_marker_count'],prepare_root_keys=g['prepare_root_keys'],phase_records=g['phase_records'],
                    kernels=selected,sum_kernel_ns=sum(s['duration_ns']for s in selected.values()),union_kernel_ns=union(ints),
                    observed_span_ns=max(b for a,b in ints)-min(a for a,b in ints)))
        pmu=read(CACHE/f'analysis/native-PMU-complete-v18/db-{dbi:02d}.json')
        metrics={}
        for group in pmu['groups']:
            ids=group['native_TASK_candidates']
            if len(ids)==1 and ids[0]in select:
                assert ids[0]not in metrics
                metrics[ids[0]]={r['metric_name']:r['value']for r in group['metric_rows']};pmu_selected+=1
        for row in chains:
            if row['database_index']==dbi:
                for s in row['kernels'].values():s['PMU']=metrics.get(s['task_rowid'])
        print(json.dumps(dict(processed_DB=dbi+1,graph_chains=len(chains))),flush=True)if dbi%8==7 else None
    assert sum(len(v)for v in op_durations.values())==286950 and len(norms)==4472 and len(chains)==3200
    rows=[dict(op_type=k,**stats(v))for k,v in op_durations.items()];rows.sort(key=lambda r:r['sum_ns'],reverse=True)
    prof=dict(generated_utc=datetime.now(timezone.utc).isoformat(),total_ComputeInfo_executions=286950,operator_types=len(rows),
        ranking_by_summed_device_task_duration=rows,norms=norms,graph_chains=chains,
        chain_roles='Exact saved graph block/task identities with one norm/MatMulV3/gating perblock;source-consistent norm→router→gating,not tensor-storage equality from timestamps.',
        measured_scope='32DB summeddevice time,including phase-unbound/dummy/graph populations;not request latency or criticalpath gain.',
        whole_module_closed=False,overall_complete=False)
    write(O/'operator-fusion-profile-v55.json',prof)
    H=5120;E=384;B=1.6e12;model=[]
    for m in [2,3,800,816,834]:
        norm_bytes=8*m*H+2*H
        reduced=4*m*H+2*H
        router_bytes=4*(m*H+H*E+m*E)
        model.append(dict(M=m,H=H,E=E,norm_scalar_arithmetic_ops=m*(4*H+1),norm_rsqrt_ops=m,
            norm_min_logical_bytes_with_two_outputs=norm_bytes,norm_bytes_without_FP32_output=reduced,
            router_dense_flops=2*m*H*E,router_min_logical_bytes=router_bytes,router_arithmetic_intensity_flop_per_logical_byte=2*m*H*E/router_bytes,
            norm_router_FP32_interface_write_read_bytes=8*m*H,router_topk_logits_write_read_bytes=8*m*E,
            norm_materialized_cold_HBM_reference_us=norm_bytes/B*1e6,
            router_materialized_cold_HBM_reference_us=router_bytes/B*1e6,
            FP32_interface_reference_transfer_time_us=8*m*H/B*1e6,
            logit_interface_reference_transfer_time_us=8*m*E/B*1e6,
            interface_bytes_over_router_weight_bytes=(8*m*H+8*m*E)/(4*H*E)))
    regime_rows=[dict(op_type=k[0],inputShapes=k[1],inputDtypes=k[2],blockNum=k[3],**stats(v))for k,v in regimes.items()]
    valid=[r for r in chains if r['prepare_root_keys']];unbound=[r for r in chains if not r['prepare_root_keys']]
    assert len(valid)==2560 and len(unbound)==640
    write(O/'F04-router-fusion-roofline-v55.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        focus='RmsNormCast→FP32router MatMulV3→MoeGatingTopKHash;existingF04/F20',
        measured_graph_chains=3200,root_bound_chains=2560,phase_unbound_chains=640,selected_PMU_groups=pmu_selected,
        graph_regime_kernel_times=regime_rows,chain_kernel_sum=stats([r['sum_kernel_ns']for r in valid]),
        chain_observed_span=stats([r['observed_span_ns']for r in valid]),
        observed_span_minus_union=stats([r['observed_span_ns']-r['union_kernel_ns']for r in valid]),
        numerical_contract='Existing BF16 output is rounded,thenFP32plane widens those roundedvalues. RetainBF16expert/sharedconsumers,FP32gateweights andsqrtsoftplus/bias/hash/ID/topK semantics.',
        symbolic_bounds=['Normmaterialized scalar algorithm:4MH+M ordinaryscalarops plusMrsqrt;reductionorder/precision matters.',
            'Dense router GEMM conventionalwork 2MHE;this is a stated densealgorithm model,not allalgorithm lowerbound.',
            'T_reference>=max(F/P_dtype,Q_HBM/B_HBM,Q_L2/B_L2,other pipeline/dep bounds). Logical interface bytes are not measured HBM traffic.',
            'WhenFP32norm plane is unused elsewhere,newrouter may widenBF16internally:8MH logical write/read bytes can disappear;BF16consumer plane remains.',
            'MatMul→topK fusion may avoid8ME logitwrite/read bytes ifall other scoreconsumers absent;ignoredthirdcustom-op output is not priced as written.'],
        shape_models=model,
        hardware_reference=dict(source='https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1',publisher='Huawei authors,CloudMatrix384',
            per_die_BF16_FP16_TFLOPS=376,per_die_INT8_TOPS=752,per_die_HBM_TB_per_s=1.6,per_die_Cube=24,per_die_Vector=48,per_die_GB=64,
            current_clock_cached_MHz=1800,reference_clock_unknown=True,
            qualification='Published reference die,not measuredcurrentservercapacity. FP32router peak not supplied by thisBF16table;no FP32MFU calculation orclockscaling invented.'),
        reference_time_qualification='Numerical transfer floors assume every counted logicalbyte crossesHBM and1.6TB/s applies. Cache-resident traffic,duplicate loads,actualSKU/clock andlatency invalidate an unconditional HBMbound. Removed-transfer/B is a traffic valuation,not endtoend gain.',
        verdicts=[dict(id='F04-A',state='ALREADY_FUSED',conclusion='Norm+cast dual output alreadyfused;no newgain.'),
            dict(id='F04-B/F20-NORM',state='CONDITIONAL_LEGAL',conclusion='WidenroundedBF16 inside router;eliminateFP32interface ifall consumers accounted. LargeM muchlargerbyteopportunity;smallM primarilylaunch/reduction/occupancy question.'),
            dict(id='F20-GATE',state='CONDITIONAL_LEGAL',conclusion='RouterGEMM epilogue+nonlinear/bias/hash/topK can avoidscoresurface with equivalentFP32/BF16barriers andexactties;fullrow reduction and newtiling stillneedproof.'),
            dict(id='F04-D',state='CONDITIONAL_SPLIT',conclusion='SmallMnorm uses3blocks;feature-splitting may increaseparallelism but adds crossblock reduction/state. Blockcount is not measuredphysicalutilization.'),
            dict(id='F04-E',state='EXCLUDED',conclusion='DroppingBF16rounding/liveBF16plane,norm-scale cancellation undernonlinearrouter,orcrosslayer same-shape reuse are notunconditional rewrites.')],
        nonadditive_limits=['Exact3200graphblock identities do not prove tensor aliases or actualnewkernelnumerics.',
            'Span gaps include otherID/control/overlapped work;notidle/removablebydefinition.',
            'M800norm andM816routing usepadding domains;not a same-callpair bycount/time. Models forlargeM are independent stateddomains.',
            'SmallM logicalbyte savings cannot be added to removedlaunch times as an observedcriticalpath gain.'],
        implementation=False,native_test=False,measured_speedup=None,whole_module_closed=False,overall_complete=False))
    print(json.dumps(dict(operator_types=len(rows),graph_chains=len(chains),root_bound=len(valid),phase_unbound=len(unbound),shape_models=len(model),overall_complete=False)))


if __name__=='__main__':main()
