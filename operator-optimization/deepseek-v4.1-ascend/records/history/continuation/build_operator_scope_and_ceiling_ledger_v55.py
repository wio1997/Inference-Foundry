"""Operator-only scope and complete observed regime inventory with qualified cost models."""
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math
from finalize_origin_coverage_v9 import N,O,read,write


def shape(text):
    if text in ['N/A','NA','DT_UNDEFINED','']:return None
    try:return [[int(v)for v in s.split(',')if v]for s in text.strip('"').split(';')]
    except ValueError:return None


def main():
    assert not(O/'operator-shape-ceiling-ledger-v55.json').exists()
    stamp=datetime.now(timezone.utc).isoformat()
    cache=read(O/'operator-shape-resource-cache-v55.json')['files'][0]
    raw=Path(cache['local_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==cache['sha256']
    source=json.loads(raw);rows=[];dense=[];bytype=Counter();covered=Counter();below=0
    sizes={'FLOAT':4,'DT_FLOAT':4,'FLOAT32':4,'DT_FLOAT32':4,'DT_BF16':2,'BFLOAT16':2,'BF16':2,
        'FLOAT16':2,'DT_FLOAT16':2,'HALF':2,'INT8':1,'DT_INT8':1,'UINT8':1,'DT_UINT8':1,
        'INT16':2,'DT_INT16':2,'INT32':4,'DT_INT32':4,'UINT32':4,'INT64':8,'DT_INT64':8,'BOOL':1,'DT_BOOL':1}
    def extents(shapes,dtypes):
        if shapes is None or len(shapes)!=len(dtypes):return None
        if any(d not in sizes for d in dtypes):return None
        return sum(math.prod(s)*sizes[d]for s,d in zip(shapes,dtypes))
    for i,r in enumerate(source['shape_role_context_regimes']):
        sig=r['signature'];op=sig['opType_text'];bytype[op]+=1
        ins=shape(sig['inputShapes_text']);outs=shape(sig['outputShapes_text'])
        it=sig['inputDataTypes_text'].split(';');ot=sig['outputDataTypes_text'].split(';')
        row=dict(regime_id='OPR-'+hashlib.sha256(json.dumps([r['module_candidates'],r['role'],r['origin'],r['context'],sig],sort_keys=True).encode()).hexdigest()[:16],
            module_candidates=r['module_candidates'],operator=op,role=r['role'],origin=r['origin'],context=r['context'],
            signature=sig,executions=r['summary']['executions'],missing_PMU=r['summary']['missing_PMU'],
            measured_native_task_duration=r['summary']['native_task_duration_ns'],
            descriptor_input_bytes=extents(ins,it),descriptor_output_bytes=extents(outs,ot),
            descriptor_extent_is_mandatory_traffic=False,
            fusion_analysis_status='REQUIRES_LINK_TO_EXISTING_CANDIDATE_AND_LIVE_CONSUMERS',
            theory_status='NEEDS_OPERATOR_SEMANTIC_WORK_AND_MEMORY_LEVEL_MODEL')
        if op in ['MatMulV2','MatMulV3','QuantBatchMatmulV3'] and ins is not None and outs is not None and len(ins)>=2 and len(outs)==1 and len(ins[0])==len(ins[1])==len(outs[0])==2:
            m,k=ins[0];om,n=outs[0];b=ins[1]
            if om==m and m>0 and n>0 and k>0 and b in [[k,n],[n,k]]:
                flops=2*m*n*k;peak=None;dtype=it[:2]
                if len(dtype)==2 and all(x in ['DT_BF16','BF16','BFLOAT16','FLOAT16','DT_FLOAT16','HALF']for x in dtype):peak=376e12
                if dtype==['INT8','INT8']or dtype==['DT_INT8','DT_INT8']:peak=752e12
                # First two operands and one result only. Quant scales/epilogue can add work/traffic.
                if len(ot)==1 and all(x in sizes for x in dtype+ot):
                    q=m*k*sizes[dtype[0]]+k*n*sizes[dtype[1]]+m*n*sizes[ot[0]]
                    memory=q/1.6e12*1e6;compute=flops/peak*1e6 if peak else None
                    floor=max(memory,compute or 0)
                    med=r['summary']['native_task_duration_ns']['median']/1000
                    model=dict(M=m,N=n,K=k,dtype=';'.join(dtype),conventional_dense_ops_2_per_MAC=flops,
                        one_read_operands_one_write_result_logical_bytes=q,
                        reference_compute_floor_us=compute,reference_cold_HBM_floor_us=memory,
                        reference_roofline_floor_us=floor,observed_median_us=med,
                        observed_to_reference_floor_ratio=med/floor,
                        observed_below_reference_floor=med<floor,
                        assumptions='Dense matrixalgorithm;mandatory operands/result;bytes actuallycrossHBM;reference910Cdie rates apply. Quantmetadata/epilogue,cache,tiling andnative precision not fullymodeled.')
                    row.update(dense_algorithm_model=model,theory_status='CONDITIONAL_DENSE_REFERENCE_ROOFLINE_MODELLED')
                    dense.append(dict(regime_id=row['regime_id'],operator=op,modules='|'.join(r['module_candidates']),origin=r['origin'],context=r['context'],executions=row['executions'],**model))
                    for mod in r['module_candidates']:covered[mod]+=1
                    below+=int(med<floor)
        if (op=='RmsNormCast'and sig['inputShapes_text']=='"3,5120;5120"')or(op=='MatMulV3'and sig['inputShapes_text']=='"3,5120;384,5120"')or(op=='MoeGatingTopKHash'and sig['inputShapes_text'].startswith('"3,384;')):
            row['fusion_analysis_status']='SEE_F04_ROUTER_FUSION_ROOFLINE_V55_AND_PRIOR_F04_F20_CONSUMER_CONTRACTS'
        rows.append(row)
    assert len(rows)==2082 and sum(r['executions']for r in rows)==286950
    assert sum(r['missing_PMU']for r in rows)==849
    write(O/'operator-shape-ceiling-ledger-v55.json',dict(generated_utc=stamp,source_cache=str(O/'operator-shape-resource-cache-v55.json'),
        observed_regimes=rows,regime_count=2082,execution_count=286950,operator_types=len(bytype),
        operator_type_regime_counts=dict(bytype),dense_model_regimes=len(dense),dense_model_module_membership_counts=dict(covered),
        below_cold_reference_floor_regimes=below,
        interpretation='Below a hypothetical cold reference floor means assumptions needreconciliation;it isnot over100percentphysicalefficiency or a failed measurement. Unmodeled regimes remain explicit.',
        all_operator_fusion_decisions_complete=False,all_theoretical_bounds_complete=False,whole_module_closed=False,overall_complete=False))
    if dense:
        with(O/'operator-dense-reference-roofline-v55.csv').open('w',newline='')as f:
            w=csv.DictWriter(f,fieldnames=list(dense[0]));w.writeheader();w.writerows(dense)
    write(O/'OPERATOR-FOCUS-REQUIREMENTS-v55.json',dict(generated_utc=stamp,
        latest_user_steering='代码阅读围绕算子融合和理论上限；目标不是研究框架。',
        objective_unchanged='覆盖指定模型/版本/硬件/负载的整体算子优化机会与关键证据。',
        acceptance_requirements=[
            '覆盖所有实际operator/shape/dtype/phase及影响这些算子的条件分支；对未触发但属于模型范围的分支明确适用条件。',
            '每条融合/拆分/重组给出原子图、消费者、数值/alias/同步/容量约束，结论区分可条件成立、已融合、不可直接改写。',
            '每条关键链核算必要计算和分层访存/通信模型、理论上界或条件下界，绑定同一执行分母比较profile。',
            '跨层/跨卡/跨迭代只围绕operator数据依赖、所有权、复用和临界路径；不独立研究HTTP/API/框架生命周期。',
            '代码仅补profile无法回答而会改变算子合法性或理论上界的具体问题；不把实现/nativebenchmark当条件分析完成前提。',
            '台账成稿不证明所有算子、分支、消费者或上界完成；未分析项保留。'],
        framework_records='已有框架证据保留为背景。M00-M03/M25的HTTP、tokenizer、SSE等全分支穷尽不再作为独立优化研究主线；仅在影响具体算子决策时引用。',
        discontinued_current_expansion=['Stop modelgeneration default/plugin/API/parent audit unless it changes a namedoperatorvariant. No more broad request branch inventory work.'],
        next_operator_priority=['DispatchFFNCombine communication/expert subgraph and smallM versuslargeM decomposition,using existingMC2/EP16evidence.',
            'HcPre/HcPost,norm/quant/RoPE and projection chains;recomputework/traffic afterfusion,not just add originalopcosts.',
            'SparseFlashMla/Indexer/Engram memorypaths andcrosslayerstate/capture lifetime only where needed foroperatorreuse.',
            'Target/draft head and sampling local/partition-aware reductions,keeping mandatory probability/entropy readers.'],
        no_new_implementation_or_device_work=True,overall_complete=False))
    print(json.dumps(dict(regimes=2082,executions=286950,operator_types=len(bytype),conditional_dense_models=len(dense),below_cold_reference_assumptions=below,overall_complete=False)))


if __name__=='__main__':main()
