"""Check only new rank/projection-cost deltas; reuse old operator proofs."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
from datetime import datetime, timezone
import hashlib, json, re, sqlite3, sys
from finalize_origin_coverage_v9 import N, O, T as TASK_DIR, L as LOOP_DIR, read, write
from analyze_evidence_batches_v155 import pin


def main():
    preflight='--preflight'in sys.argv;count=0
    def check(condition,why):
        nonlocal count
        assert condition,why;count+=1
    for source in read(O/'carrier-resource-inputs-v156.json')['inputs']:check(pin(source['path'])==source,'input pin '+source['path'])
    for source in read(O/('checkpoint-manifest-v155.json'if preflight else'checkpoint-manifest-v156.json'))['artifacts']:check(pin(source['path'])==source,'sealed pin '+source['path'])
    data=read(O/'rank-device-namespace-bindings-v156.json');seen=Counter();groups=set()
    for row in data['rows']:
        db=sqlite3.connect(Path(row['database']['path']).as_uri()+'?mode=ro',uri=True)
        check(db.execute('select rankId,deviceId from RANK_DEVICE_MAP').fetchall()==[(row['global_EP_rank'],row['exported_rank_device_id'])],'rank-device metadata exact')
        check(db.execute('select id from NPU_INFO').fetchall()==[(row['captured_DeviceInfo_id'],)],'exported device metadata')
        raw=db.execute("select value from META_DATA where name='parallel_group_info'").fetchall();check(len(raw)==1,'one parallelgroup map');db.close()
        records=json.loads(raw[0][0]);recorded_tp=[x for x in records.values()if x['group_name']=='tp'];check(len(recorded_tp)==1,'one TPgroup perprofile')
        group=recorded_tp[0];check(group['global_ranks'][group['group_rank']]==row['global_EP_rank'],'TPglobalrank lookup');groups.add(tuple(group['global_ranks']))
        info=read(Path(row['device_info']['path']));profiler=read(Path(row['profiler_info']['path']))
        check(info['DeviceInfo'][0]['id']==row['captured_DeviceInfo_id']and int(info['pid'])==row['container_worker_pid'],'captured info identity')
        check(profiler['rank_id']==row['global_EP_rank'],'profiler rank identity')
        check(row['logged_local_rank']==row['trace_TP_label']==row['global_EP_rank']%8,'logged localrank reset perDP')
        check(not row['physical_chip_or_board_binding_proved'],'no metadata=>physical assumption');seen[row['global_EP_rank']]+=1
    check(seen==Counter({i:2 for i in range(16)})and groups=={tuple(range(8)),tuple(range(8,16))},'all32 metadata joins/twoTPgroups')
    check(data['rank_to_exported_device_metadata_bridge_bound']and not data['exported_device_metadata_to_physical_chip_bridge_bound'],'precise bridge scope')
    source_log=read(O/'rank-device-archived-provenance-v156.json');log_rows={}
    for log in source_log['logs']:
        check(not log.get('selection_capped',False)and not log.get('skipped'),'complete targeted log query')
        for row in log['selected']:
            found=re.search(r'Worker pid=(\d+)\).*world_size=(\d+) rank=(\d+) local_rank=(\d+)',row['text']);check(found is not None,'ranklog parse');pid,world,rank,local=map(int,found.groups());log_rows[rank]=(pid,world,local,row['line'])
    for row in data['rows']:check(log_rows[row['global_EP_rank']]==(row['container_worker_pid'],16,row['logged_local_rank'],row['log_source_line']),'saved log join')
    expected_devices=','.join(map(str,range(16)))
    for script in source_log['scripts']:check(any(x['text']=='export ASCEND_RT_VISIBLE_DEVICES='+expected_devices for x in script['visibility_or_rank_lines']),'archived runtime visible list')
    check(source_log['stopped_container_filtered']['running']is False,'stopped inspect only')
    official=read(O/'device-id-primary-receipt-v156.json');check(official['http_status']==200 and official['not_installed_9_2_attestation'],'primary namespace concept only')
    carrier=read(O/'projected-carrier-weight-amortization-v156.json');check(len(carrier['models'])==20,'current4geometries x5reuse schedules,not candidatepairs')
    prior=read(O/'post-attention-projected-head-carrier-v153.json');geometries={(x['M'],x['Q'],x['T'],x['G']):x for x in prior['geometry']}
    for model in carrier['models']:
        M=model['M'];P=model['P'];R=model['same_layer_same_weight_version_calls'];geometry=geometries[(M,model['Q'],model['T'],model['G'])];check(geometry['P']==P==8,'matching carrier geometry')
        # Independently account for each actual source/recipient commodity in
        # this declared schedule. This is communication costing, not candidate pairing.
        old_group=0;new_group=0;weight_group=0
        for src in range(P):
            for dst in range(P):
                if src==dst:continue
                old_group+=R*M*4096*2;new_group+=R*M*1024*2;weight_group+=4096*1024*2
        old=old_group//P;new=(new_group+weight_group)//P
        check(old==model['old_activation_offrank_delivery_bytes_total'],'old reference deliveries')
        check(new==model['new_activation_plus_weight_delivery_bytes_total'],'new value+weight deliveries')
        check(old-new==model['net_delivery_bytes_saved_total'],'joint delivery difference')
        check(Fraction(new,old)==Fraction(1,4)+Fraction(1024,M*R),'closed ratio formula')
        check(Fraction(model['new_to_old_delivery_ratio']['numerator'],model['new_to_old_delivery_ratio']['denominator'])==Fraction(new,old),'serialized exact ratio')
        check(model['positive_logical_delivery_saving']==(old>new)and model['actual_wire_HBM_or_latency_proved']is False,'cost result scope')
        if R==1:check(new>old,'cold current-M counterexample')
        if R==2:check(new<old,'two-call current-M logical result')
    for seq in carrier['variable_M_sequence_boundaries']:
        sumM=sum(seq['owner_M_sequence']);signed=7*(sumM*8192-sumM*2048-8388608)
        check(seq['sum_M']==sumM and seq['net_offrank_delivery_bytes_saved']==signed,'nonuniform reuse accounting')
        check(seq['positive_saving']==(3*sumM>4096)==(sumM>=1366),'integer strict threshold')
    check(3*1365<4096<3*1366,'exact break-even boundary')
    placement=read(O/'carrier-board-placement-sensitivity-v156.json');archived=read(O/'archived-chip-board-mapping-v155.json')['first_mapping'];chipboard={x['physical_id']:x['npu_board']for x in archived}
    possible_s={8-2*u for u in range(5)};check(possible_s=={p['split_boards']for p in placement['placements']},'algebraic occupancy classes')
    for case in placement['placements']:
        A,B=[g['hypothetical_rank_to_physical_chip']for g in case['groups']];check(len(A)==len(B)==8 and set(A).isdisjoint(B)and set(A)|set(B)==set(range(16)),'conditional bijective placement witness')
        for group in case['groups']:
            chips=group['hypothetical_rank_to_physical_chip'];occupancy=Counter(chipboard[x]for x in chips)
            same=sum(n*(n-1)for n in occupancy.values());cross=8*8-sum(n*n for n in occupancy.values())
            check(same==group['same_board_directed_pairs']==8-case['split_boards'],'same-board pair identity')
            check(cross==group['cross_board_directed_pairs']==48+case['split_boards'],'cross-board pair identity')
            check(group['boards_used']==len(occupancy)==4+case['split_boards']//2,'board count')
            check(case['is_actual_placement']is False,'no chosen actual placement')
    check(placement['not_a_universal_physical_communication_lower_bound']and not placement['actual_rank_to_chip_bridge_proved'],'delivery vs physical-bound limitation')
    summary=read(O/'carrier-resource-summary-v156.json');check(summary['new_pair_clauses']==summary['new_independence_proofs']==summary['new_native_operand_or_phase_promotions']==summary['new_whole_module_closures']==0,'no unproved promotions')
    check(not summary['new_device_work']and not summary['overall_complete'],'goal remains open')
    if not preflight:
        cp=read(O/'REVIEW-CHECKPOINT-v156.json');check(cp['summary']==read(N/'RECOVERY.json')['latest_profile_first_review']['summary'],'recovery agrees')
        check(cp['candidate_ledger'].endswith('candidate-profile-ledger-v154.json')and cp['unchanged_operator_authorities_reused'],'canonical authority reuse')
        check(read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(LOOP_DIR/'state.json')['status']=='EVALUATING','Task/Loop status')
        pointer=read(N/'CURRENT-ANALYSIS-WORKFLOW.json');check(pin(pointer['protocol']['path'])==pointer['protocol']and pin(pointer['work_queue']['path'])==pointer['work_queue']and pointer['effective_checkpoint']==156,'current workflow')
    result=dict(status='PASS',checkpoint=156,preflight=preflight,checks=count,verification_scope=['All32explicit rank/device/group/log metadata joins','Direct per-source/recipient activation+weight delivery accounting for current4geometries and5reuse schedules','Nonuniform reuse break-even and five algebraic board occupancy classes','Input/proof hashes and current authority state;no rerun of unchanged prior operator case suites'],old_unchanged_operator_case_suites_reexecuted=False,new_device_work=False,new_native_or_whole_closures=0,overall_complete=False,observed_utc=datetime.now(timezone.utc).isoformat(),verifier=pin(Path(__file__)))
    if preflight:
        output=O/'carrier-resource-preflight-v156.json';check(not output.exists(),'single preflight receipt');write(output,result)
    else:
        manifest=O/'checkpoint-manifest-v156.json';result.update(sealed_manifest=pin(manifest),sealed_manifest_files=len(read(manifest)['artifacts']),recovery_summary_agrees=True,sealed_files_modified=False);output=O/'carrier-resource-postcheck-result-v156p.json';check(not output.exists(),'single postcheck receipt');write(output,result);write(N/'CURRENT-POSTCHECK.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
