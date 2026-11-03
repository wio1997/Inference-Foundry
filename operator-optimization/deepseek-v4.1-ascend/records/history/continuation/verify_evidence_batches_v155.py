"""Verify the evidence delta and proof reuse, without rerunning old case suites."""
from pathlib import Path
from collections import Counter
import hashlib, json, sys
from finalize_origin_coverage_v9 import N, O, T as TASK_DIR, L as LOOP_DIR, read


def main():
    preflight='--preflight'in sys.argv
    groups=[];checks=0
    def check(ok,label):
        nonlocal checks
        assert ok,label;checks+=1
    def verify_pin(row):
        p=Path(row['path']);b=p.read_bytes();check(len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256'],str(p))
    for row in read(O/'raw-evidence-batch-inputs-v155.json')['inputs']:verify_pin(row)
    if preflight:
        for row in read(O/'checkpoint-manifest-v154.json')['artifacts']:verify_pin(row)
        groups.append('Predecessor154 current shared state and immutable input digests valid before mutation')
    else:
        for row in read(O/'checkpoint-manifest-v155.json')['artifacts']:verify_pin(row)
        cp=read(O/'REVIEW-CHECKPOINT-v155.json')
        check(cp['checkpoint_schema']=='DELTA_CHECKPOINT_V1'and cp['unchanged_operator_authorities_reused'],'delta checkpoint')
        check(cp['summary']==read(N/'RECOVERY.json')['latest_profile_first_review']['summary'],'recovery authority')
        check(cp['candidate_ledger'].endswith('candidate-profile-ledger-v154.json'),'unchanged semantic authority reused')
        check(read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(LOOP_DIR/'state.json')['status']=='EVALUATING','task/loop active')
        groups.append('New sealed manifest, authority reuse and Task ACTIVE/Loop EVALUATING')
    census=read(O/'raw-torch-field-census-v155.json');inv=read(O/'raw-profile-inventory-v155.json');plan={r['relative']:r for r in inv['files']if Path(r['relative']).name=='torch.op_range'}
    check(len(plan)==len(census['files'])==32,'all32 physical op_range files')
    accumulated=Counter();record_tags=Counter();count=0;size=0;rel=[]
    for batch in range(8):
        receipt=read(O/('raw-torch-field-census-batch%02d-v155.json'%batch));check(len(receipt['files'])==len(receipt['plan'])==4,'batch cardinality')
        for row,spec in zip(receipt['files'],receipt['plan']):
            check(row['relative']==spec['relative']and spec==plan[row['relative']],'exact planned raw file')
            check(row['complete']and not row['errors']and row['source_stat_unchanged'],'complete bounded TLV parse')
            check(row['source_bytes']==row['consumed_bytes']==spec['bytes']and row['source_mtime_ns']==spec['mtime_ns'],'whole byte extent')
            check(len(row['sha256'])==64 and int(row['sha256'],16)>=0,'whole stream digest')
            check(row['tag_record_counts'].get('2')==row['records'],'one opName-bearing record per parsed record')
            check(not any(str(t)in row['tag_occurrences']for t in[5,6,7]),'no hidden5/6/7fields in full stream')
            check(not row['duplicate_field_occurrences'],'no duplicate-tag overwrite ambiguity')
            check(row==next(x for x in census['files']if x['relative']==row['relative']),'receipt aggregate faithful')
            accumulated.update(row['tag_occurrences']);record_tags.update(row['tag_record_counts']);count+=row['records'];size+=row['source_bytes'];rel.append(row['relative'])
    check(len(set(rel))==32 and count==census['records']==1448456 and size==census['source_bytes']==3214148270,'unique full census totals')
    check(dict(accumulated)==census['tag_occurrences']and dict(record_tags)==census['tag_record_counts'],'tag aggregate')
    groups.append('32 full archived Torch streams; bounds, source stats, hashes and all-record tag census')
    raw=read(O/'raw-cann-field-census-v155.json');hcount=0;tcount=0;slots=Counter()
    for row in raw['hccl_files']:
        p=Path(row['source']['path']);b=p.read_bytes();check(len(b)%256==0,'HCCL fixed size')
        for offset in range(0,len(b),256):
            check(int.from_bytes(b[offset:offset+2],'little')==0x5a5a,'HCCL magic')
            # Independent offset reads against schema, rather than the analyzer's unpack/index path.
            check(b[offset+96:offset+112]==bytes(16),'HCCL recorded src/dst bothzero')
            hcount+=1
    for row in raw['tensor_files']:
        b=Path(row['source']['path']).read_bytes();check(len(b)%256==0,'tensor fixed size')
        for offset in range(0,len(b),256):
            check(b[offset:offset+2]==b'ZZ','tensor magic')
            num=int.from_bytes(b[offset+32:offset+36],'little');check(0<=num<=5,'tensor slot count')
            tcount+=1;slots['declared_tensor_slots']+=num
            for index in range(num):
                direction=int.from_bytes(b[offset+36+44*index:offset+40+44*index],'little');slots['input_slots'if direction==0 else'output_slots'if direction==1 else'other_direction_slots']+=1
    check(hcount==raw['hccl_counts']['records']==11864,'HCCL all records')
    check(tcount==raw['tensor_counts']['records']==297976,'tensor all records')
    check(all(raw['tensor_counts'][key]==value for key,value in slots.items()),'tensor slot counts')
    check(raw['tensor_schema_has_declared_address_stride_or_workspace_fields']is False and raw['reserved_bytes_not_reinterpreted_as_operands'],'schema-only field interpretation')
    availability=read(O/'raw-field-availability-and-reuse-v155.json')
    check(all(row['repeat_same_scan_for_each_pair']is False for row in availability['routes']if'repeat_same_scan_for_each_pair'in row),'reuse field census')
    check(not availability['native9_1352_closed']and not availability['new_independence_proofs'],'absence not closure')
    groups.append('Independent byte-offset check of all11864HCCL and297976tensor records; no operand/independence promotion')
    hardware=read(O/'archived-hardware-pool-contract-v155.json');check(len(hardware['device_info_files'])==32,'32device info files')
    for row in hardware['device_info_files']:
        verify_pin(row['source']);source=read(Path(row['source']['path']));dev=source['DeviceInfo'][0]
        check(source['cannVersion']=='9.2.0'and(dev['ai_core_num'],dev['aiv_num'])==(24,48),'captured CANN/core metadata')
        check(dev['ctrl_cpu_endian_little']==1,'record endianness')
    boards=read(O/'archived-chip-board-mapping-v155.json');mappings=[]
    for sample in boards['maps']:
        rows=sample['rows'];check(len(rows)==16,'16chip rows per archived sample');mappings.append(sorted((x['npu_board'],x['chip_index'],x['physical_id'],x['bus_id'])for x in rows))
    check(len(mappings)==197 and all(x==mappings[0]for x in mappings),'stable197archived board maps')
    check(len({x[0]for x in mappings[0]})==8 and len({x[2]for x in mappings[0]})==16,'board/chip distinct')
    check(not hardware['reported_clock_is_certified_fmax']and not hardware['runtime_device_to_physical_chip_bridge_proved']and not hardware['physical_MFU_or_lowest_module_established'],'hardware limits explicit')
    groups.append('Archived hardware counts/maps with explicit logical/physical and clock/mode limits')
    protocol=read(O/'coverage-compression-protocol-v155.json');audit=read(O/'coverage-empty-evidence-audit-v155.json');queue=read(O/'coverage-work-queue-v155.json')
    check(protocol['full_goal_scope_preserved']and len(protocol['proof_families'])==7,'complete scope and proof families')
    for family in protocol['proof_families']:
        for p in family['reuse_inputs']:verify_pin(p)
        check(family['family_label_is_not_semantic_equivalence_or_closure'],'proof label scope')
    check(audit['counts']['structured_unknown_rows_misclassified_as_independent']==0 and audit['counts']['explicit_no_independence_from_empty_guards']==595,'empty-field audit')
    check(len(audit['routes'])==595 and all(not r['dispatch_as_individual_pair_task']and not r['proves_independence']for r in audit['routes']),'all595navigation routes no pair tasks')
    check(queue['not_candidate_cartesian_pairs']and queue['legacy_383_rows_are_coverage_index_not_383_tasks']and len(queue['work'])==3,'three question batches')
    navigation=read(O/'source-cut-navigation-v155.json');edges=read(O/'operator-graph-edge-coverage-v124.json')['rows'];seen=[]
    for cut in navigation['cuts']:
        check(not cut['dispatch_as_individual_pair_task']and not cut['proves_independence'],'source grouping no pair task or proof')
        for index in cut['source_edge_rows']:
            row=edges[index];check((row['graph'],row['template'],row['edge']['producer'])==(cut['graph'],cut['template'],cut['producer']),'producer cut key');seen.append(index)
    check(sorted(seen)==list(range(1346))and len(navigation['cuts'])==812,'source edges indexed once')
    check(not navigation['graph_inventory_exhaustiveness_proved']and navigation['new_semantic_proofs']==0,'partial source graph only')
    pointer=read(N/'CURRENT-ANALYSIS-WORKFLOW.json');verify_pin(pointer['protocol']);verify_pin(pointer['work_queue'])
    summary=read(O/'raw-evidence-batch-summary-v155.json');check(summary['new_scoped_pair_clauses']==summary['new_native_operand_or_phase_promotions']==summary['new_independence_proofs']==summary['whole_modules_closed']==0 and summary['overall_complete']is False,'no unjustified progress/closure')
    groups.append('Proof hashes reused; routing compressed without dropping unknown scope or enumerating subsets')
    result=dict(status='PASS',preflight=preflight,verification_groups=groups,checks=checks,old_unchanged_finite_case_suites_reexecuted=False,new_device_work=False,overall_complete=False)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
