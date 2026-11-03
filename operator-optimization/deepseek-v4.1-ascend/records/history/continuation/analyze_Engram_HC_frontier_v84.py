"""Profile-first Engram projection/hidden join and next HcPre boundary."""
from pathlib import Path
from collections import defaultdict,Counter
from statistics import median
import hashlib
from finalize_origin_coverage_v9 import O,R,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions


def stats(values):
    return dict(n=len(values),minimum=min(values),median=median(values),maximum=max(values))if values else None


def main():
    assert not(O/'Engram-HC-native-frontiers-v84.json').exists()
    model=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py'
    layer=model.parent/'engram/layer.py';common=model.parent/'engram/common.py'
    source=[]
    for path,expected,ranges in [
        (model,'101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',[(837,843),(879,920),(1554,1607)]),
        (layer,'dc9457c88aa3c90133115bc608447e037b0693f8db71bb8da37f1ead44021f80',[(18,56)]),
        (common,'4797b8962264d137cdb142a9988825288283092fb797ddbd334c6ab2223eb58d',[(46,72)])]:
        blob=path.read_bytes();assert hashlib.sha256(blob).hexdigest()==expected
        lines=blob.decode().splitlines()
        source.append(dict(path=str(path),sha256=expected,bytes=len(blob),clauses=[dict(first=a,last=z,source=lines[a-1:z])for a,z in ranges],
            grade='REUSED_PINNED_LOCAL_OPERATOR_CALLER;NOT_NEW_TARGET_SOURCE_OR_LOADED_BINARY_ATTESTATION'))
    config_ref=read(O/'HC-operator-numerical-and-dependency-contract-v59.json')['model_config']
    config_path=Path(config_ref['path']);assert hashlib.sha256(config_path.read_bytes()).hexdigest()==config_ref['sha256']
    config=read(config_path)['text_config'];assert config['engram_layer_ids']==[1,14] and config['hc_mult']==4 and config['hidden_size']==5120
    write(O/'Engram-HC-operator-source-contract-v84.json',dict(source_files=source,config=dict(path=str(config_path),sha256=config_ref['sha256'],
        target_layers=[1,14],draft_Engram_constructed=False,hc=4,H=5120),facts=[
        dict(id='WKV-PURE-ROWS',statement='AscendEngram.forward line41 computes wkv(rows) before the gate. It has no hidden-value input. Rows/weight/quantizer/device/layout/numerical generation still constrain legal hoist/reuse.'),
        dict(id='KEY-RSTD-PURE',statement='Common gate line67 computes rsqrt(mean(key.float()**2)+eps), independent of hidden. Cache this scalar only if original square/mean/add/rsqrt arithmetic and FP32 bits are preserved. Later multiply hidden_rstd by key_rstd in original order.'),
        dict(id='NO-PREWEIGHT-KEY',statement='Original numerator is sum((rotated_hidden*channel_weight)*key), then *combined_rstd*dim^-0.5. Preweighting or normalizing key before this dot reassociates FP32 operations; not an unconditional exact rewrite.'),
        dict(id='MUTATE-BEFORE-HC',statement='Model copies gate result into hidden_states[:n], then invokes layer. Decoder first binds residual=hidden_states and HcPre consumes this post-Engram BF16 value. Both coefficient and collapsed-Y branches require that updated value.'),
        dict(id='NON-ENGRAM-AND-DRAFT',statement='Construction is enabled only at configured target layers1/14 and explicitly excludes draft layers. Guard token_mask.numel() plus lookup-ready/shard/slice precedes the gate; do not hoist into inactive branches or claim all layers have this boundary.'),
        dict(id='OLD-HIDDEN-READERS',statement='Rotation and gate residual use old hidden, while HcPre/current HcPost residual use updated hidden. Direct gate output to the existing slice needs all old readers finished, exact BF16 rounding, row ownership and capture bank/alias/error semantics; no pointer proof is inferred.'),
        dict(id='EARLY-LOOKUP-BASELINE',statement='One hash/DP production already feeds both configured layers. Early WKV is additional pure projection scheduling after lookup/TP/SP readiness, not credit for repeating/deleting already-shared hash or collective work.')],
        numerical_policy='Keep WKV BF16, original FLOAT rotation and separate FP32 multiply/add/reduction order, BF16 gate result, HcPre BF16Y then norm, HF32 coefficient/A2 and current epsilon/rowFactor contract.',
        local_operator_clauses_read=True,new_source_files_obtained=0,new_remote_source_read=False,no_target_execution=True,overall_complete=False))
    assignments=defaultdict(dict)
    for row in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
        assignments[row[0]][row[1]]=row
    bodies=read(O/'state-engram-graph-operator-chains-v62.json')['Engram_bodies']
    assert len(bodies)==160
    by_db=defaultdict(list)
    for body in bodies:by_db[body['database_index']].append(body)
    frontiers=[];seen=set()
    for db,db_bodies in sorted(by_db.items()):
        native=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json')
        definitions={r['globalTaskId']:r for r in native['COMPUTE_definitions']}
        tasks={r['task_record_rowid']:r for r in native['TASK_records']}
        grouped=defaultdict(list)
        for tid,assignment in assignments[db].items():
            if assignment[2]!='GRAPH' or assignment[8] is None or assignment[9] is None:continue
            task=tasks[tid];definition=definitions[task['globalTaskId']]
            grouped[(assignment[8],assignment[9])].append((task,definition,assignment))
        for body in db_bodies:
            graph=body['graph_ordinal'];layer_id=body['layer']
            current=grouped[(graph,layer_id)];previous=grouped[(graph,layer_id-1)]
            pres=sorted([x for x in current if x[1]['opType_text']=='HcPre'],key=lambda x:x[0]['startNs'])
            posts=sorted([x for x in previous if x[1]['opType_text']=='HcPost'],key=lambda x:x[0]['startNs'])
            assert len(pres)==len(posts)==2
            target_pre,predef,preassignment=pres[0];previous_ffn_post,postdef,postassignment=posts[-1]
            tail=tasks[body['stages'][-1]['TASK_rowid']];taildef=definitions[tail['globalTaskId']]
            wkv=tasks[body['WKV']['TASK_rowid']];wkvdef=definitions[wkv['globalTaskId']]
            key=(db,graph,layer_id);assert key not in seen;seen.add(key)
            assert previous_ffn_post['endNs']<=wkv['startNs'] and tail['endNs']<=target_pre['startNs']
            assert taildef['opType_text']=='Cast' and taildef['outputDataTypes_text']=='DT_BF16'
            tailshape=dimensions(taildef['outputShapes_text'])[0];preshape=dimensions(predef['inputShapes_text'])[0]
            assert tailshape==preshape==[3,4,5120]
            assert dimensions(wkvdef['inputShapes_text'])[0]==[3,6144]
            assert dimensions(wkvdef['outputShapes_text'])[0]==[3,25600]
            between=[t for t in native['TASK_records']if t['globalPid']==tail['globalPid'] and t['streamId']==tail['streamId'] and tail['endNs']<=t['startNs']<target_pre['startNs']]
            intervening_compute=[t['task_record_rowid']for t in between if t['globalTaskId'] in definitions]
            assert not intervening_compute
            candidates=[x for x in current if x[1]['opType_text']=='RmsNorm' and dimensions(x[1]['inputShapes_text'])[0]==[3,5120] and target_pre['endNs']<=x[0]['startNs']<pres[1][0]['startNs']]
            assert len(candidates)==1
            norm=candidates[0][0]
            frontiers.append(dict(database_index=db,graph_ordinal=graph,target_layer=layer_id,root_bound=body['root_bound'],
                phase=preassignment[6],M=3,h=4,H=5120,previous_FFN_HcPost_TASK=previous_ffn_post['task_record_rowid'],
                WKV_TASK=wkv['task_record_rowid'],WKV_duration_ns=wkv['endNs']-wkv['startNs'],
                gate_BF16_tail_TASK=tail['task_record_rowid'],first_attn_HcPre_TASK=target_pre['task_record_rowid'],
                input_RMS_TASK=norm['task_record_rowid'],tail_to_HcPre_start_gap_ns=target_pre['startNs']-tail['endNs'],
                intervening_TASK_types=dict(Counter(t['task_type_name']for t in between)),
                intervening_compute_TASKs=intervening_compute,
                previous_HCPost_to_WKV_start_gap_ns=wkv['startNs']-previous_ffn_post['endNs'],
                actual_lookup_ready_time_or_hoist_window_ns=None,
                MEMCPY_address_length_or_defuse_attested=False,
                grade='EXACT_SAVED_GRAPH_ORDINAL+BLOCK_ROLE+SOURCE_STAGE+TYPED_SHAPE;NOT_POINTER_OR_REMOVABLE_TIME_PROOF'))
    wkv_regimes=[r for r in read(O/'A3-typed-ingress-regime-models-v80.json')['regimes']if r['K']==6144 and r['N']==25600]
    assert len(wkv_regimes)==5 and sum(r['executions']for r in wkv_regimes)==208
    write(O/'Engram-HC-native-frontiers-v84.json',dict(frontiers=frontiers,graph_frontier_count=160,
        root_bound_frontiers=sum(f['root_bound']for f in frontiers),phase_unbound_frontiers=sum(not f['root_bound']for f in frontiers),
        target_layers=dict(Counter(str(f['target_layer'])for f in frontiers)),
        tail_to_HcPre_gap_ns=stats([f['tail_to_HcPre_start_gap_ns']for f in frontiers]),
        intermediate_TASK_type_counts=dict(sum((Counter(f['intervening_TASK_types'])for f in frontiers),Counter())),
        WKV_observed_regimes=[{k:r[k]for k in ['regime_id','M','K','N','executions','origin','context','signature','source_native_median_us']}for r in wkv_regimes],
        WKV_all_typed_executions=208,WKV_graph_executions_bound_here=160,WKV_DIRECT_executions_not_forced_into_graph_frontier=48,
        profile_does_not_record_all_aliases_or_lookup_ready_time=True,new_legal_fusion_or_speedup_proved=False,
        no_new_capture_or_device_execution=True,global_uncertainty_849_16_3136_preserved=True,overall_complete=False))
    print(dict(exact_graph_frontiers=len(frontiers),root_bound=sum(f['root_bound']for f in frontiers),
        phase_unbound=sum(not f['root_bound']for f in frontiers),WKV_all_executions=208,
        between_TASKs=dict(sum((Counter(f['intervening_TASK_types'])for f in frontiers),Counter())),
        new_target_source_files=0,loaded_binary_or_aliases_attested=False))


if __name__=='__main__':main()
