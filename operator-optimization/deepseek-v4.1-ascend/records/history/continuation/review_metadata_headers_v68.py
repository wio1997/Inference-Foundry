"""Independent native-field check of C2 header valuation, separate from copy semantics."""
from collections import Counter,defaultdict
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions

def main():
    assert not (O/'metadata-C2-header-native-check-v68.json').exists()
    regs=read(O/'metadata-field-regime-models-v68.json')['regimes']
    fields={'C2_query_lengths','C2_actual_requests','C2_used_prefix','C2_live_used_select','C2_previous_length','C2_first_block_if_used'}
    bydb=defaultdict(dict)
    for r in regs:
        if r['field']in fields:
            for di,tid in r['native_TASK_keys']:bydb[di][tid]=r
    rows=[]
    for di,rr in bydb.items():
        e=read(CACHE/f'analysis/native-execution-identity-v14/db-{di:02d}.json');tasks={t['task_record_rowid']:t for t in e['TASK_records']}
        ordered=sorted(rr,key=lambda tid:(tasks[tid]['startNs'],tid));assert len(ordered)%14==0
        for i in range(0,len(ordered),14):
            keys=ordered[i:i+14];a=[rr[k]for k in keys]
            assert Counter(r['operator']for r in a)=={'Sub':3,'ClipByValueV2':3,'Range':1,'Less':1,'Fill':2,'SelectV2':2,'Cast':1,'Greater':1}
            q=next(r for r in a if r['field']=='C2_query_lengths');b=dimensions(q['signature']['outputShapes_text'])[0][0]
            known=sum(r['known_logical_value_bytes_excluding_unknown_control_scalars']for r in a)
            assert known==116*b+8,(di,b,known)
            rows.append(dict(database_index=di,B=b,TASK_rows=keys,known_14TASK_value_bytes=known,
                additional_five_source_copy_reference_bytes=40*b,source_copy_TRACE_identity_proved=False))
    assert len(rows)==104
    write(O/'metadata-C2-header-native-check-v68.json',dict(header_groups=rows,calls=104,
        verified_typed_core='116B+8knownvaluebytesfrom14nativeTASK;5source-semanticINT32Bplanecopiesadd40Bvaluation,notinventedtraceidentity/physicaltraffic.',
        baseline_header_reference='156B+8',new_direct_header_reference='36B',overall_complete=False))
    print(dict(header_groups=104,native_TASKs=1456,known_core='116B+8',source_copies='40B',declared_source_total='156B+8'))

if __name__=='__main__':main()
