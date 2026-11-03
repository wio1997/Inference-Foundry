"""Compare consistent roofline domains and exact per-graph merge exposure."""
from collections import defaultdict
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_shared_expert_ceiling_v57 import stat


def main():
    p=O/'shared-expert-shape-ceiling-v57.json';d=read(p)
    for x in d['logical_shape_models']:
        x['ideal_fused_chain_logical_bytes']=x.pop('ideal_interface_eliminated_bytes')
        m=x['M'];H=x['H'];I=x['I'];B=1.6e12;P=752e12
        qs=[3*m*H+4*m,m*H+2*H*I+8*m*I,9*m*I+8*I+8*m,m*I+H*I+2*H+4*m+2*m*H,6*m*H]
        fs=[0,4*m*H*I,0,2*m*H*I,0]
        assert sum(qs)==x['materialized_chain_logical_bytes']
        floors=[max(q/B,f/P)*1e6 for q,f in zip(qs,fs)]
        x['serial_per_stage_reference_floor_us']=sum(floors)
        x['aggregate_materialized_reference_floor_us']=max(x['materialized_cold_reference_io_us'],x['reference_compute_us'])
        x['aggregate_ideal_fused_reference_floor_us']=max(x['ideal_fused_cold_reference_io_us'],x['reference_compute_us'])
        x['reference_model_limit']='Serialstagefloor assumesunchangedserialstages/coldtraffic;aggregatefloor assumesidealresourceoverlap. Ratiooflowerbounds isnotactualspeedup or guaranteedmaximumgain.'
    write(p,d)
    prof=read(O/'MC2-operator-profile-v56.json');bygraph=defaultdict(list)
    for r in prof['root_bound_graph_layers']:bygraph[(r['database_index'],r['graph_ordinal'])].append(r)
    raw=read(N/'original-profile-cache-v53/analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json');graphs=[]
    for g in raw['graph_executions']:
        key=(g['database_index'],g['graph_ordinal'])
        if key not in bygraph:continue
        rows=bygraph[key];assert len(rows)==40
        merge=sum(x['shared_stages'][-1]['duration_ns']for x in rows)
        assert merge<=g['graph_control_pair_ns']
        graphs.append(dict(database_index=key[0],graph_ordinal=key[1],layers=40,merge_task_sum_ns=merge,
            graph_device_control_span_ns=g['graph_control_pair_ns'],observed_exposure_fraction=merge/g['graph_control_pair_ns']))
    assert len(graphs)==64
    p=O/'F23-singleton-merge-specialization-v57.json';s=read(p)
    s['exact_root_graph_exposure']=graphs
    s['merge_time_per_graph_ns']=stat([g['merge_task_sum_ns']for g in graphs])
    s['merge_observed_graph_time_fraction']=stat([g['observed_exposure_fraction']for g in graphs])
    s['exposure_is_achievable_gain_or_hardware_efficiency']=False
    write(p,s)
    print({'M3':next(x for x in d['logical_shape_models']if x['M']==3),
           'M800':next(x for x in d['logical_shape_models']if x['M']==800),
           'merge_graph_fraction':s['merge_observed_graph_time_fraction']})


if __name__=='__main__':main()
