"""Native field-specific SWA/C2/CP metadata models; no opaque planner substitution."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import math,re
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import fullkey,dimensions
from analyze_projection_ceiling_v61 import stat
DT={'INT32':4,'INT64':8,'BOOL':1,'FLOAT':4,'DT_BF16':2}

def line(role):return int(re.search(r'\((\d+)\)',role).group(1))
def family(role):
    if 'a/attention/dsa_v41.py'in role:
        l=line(role)
        if 1008<=l<=1042:return'SWA_FIELDS'
        if 1327<=l<=1368:return'C2_FIELDS'
        if l in [201,202]:return'COMPRESSED_SLOT'
        if l in [905,915,1116]:return'BATCH_AND_COUNTS'
    if 'a/attention/context_parallel/dsa_v41_cp.py'in role:return'CP_LENGTHS'
    return None
FIELDS={201:'physical_compression_predicate',202:'compressed_slot_value',1008:'slot_valid',1011:'skip_slot_valid_zero',
    1013:'actual_end',1014:'actual_token_range',1016:'logical_position_parity',1017:'slot_clamp',
    1018:'page_column_publish',1019:'page_select',1021:'page_divide',1025:'offset_column_publish',1026:'offset_select_modulo',
    1028:'flat_INT64_publish',1041:'cache_length_quotient',1042:'cache_length_residual',
    140:'CP_GPU_length_minus_suffix',141:'CP_GPU_length_clamp',142:'CP_nonempty_query_select',
    1327:'C2_query_lengths',1328:'C2_actual_requests',1329:'C2_used_prefix',1330:'C2_live_used_select',1332:'C2_skip_used_zero',
    1333:'C2_previous_length',1337:'C2_first_block_if_used',1338:'C2_valid_end',1339:'C2_valid_rows',1340:'C2_complete_mask',
    1342:'C2_skip_complete_zero',1345:'C2_source_position_select',1347:'C2_previous_source_position',1348:'C2_unused_position_zero',
    1362:'C2_source_cos_lookup',1368:'C2_source_sin_lookup',905:'inactive_request_seq_tail',915:'batch_positions_INT64',1116:'SWA_nonnegative_index_count'}

def main():
    assert not (O/'metadata-field-regime-models-v68.json').exists()
    cv={r['regime_id']:r for r in read(O/'operator-analysis-coverage-ledger-v67.json')['regimes']}
    rs=[r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']if r['module_candidates']==['M21']and family(r['role'])and cv[r['regime_id']]['state'].startswith('REGIME_')]
    assert len(rs)==586 and sum(r['executions']for r in rs)==67631
    lookup={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in rs}
    selected=defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
        if a[3]==['M21']and family(a[5]):selected[a[0]][a[1]]=a
    ids=defaultdict(list);times=defaultdict(list);missing=Counter();cp=[];cos=[];sin=[];recount=[];length=[];coords=[]
    for di in range(32):
        e=read(CACHE/f'analysis/native-execution-identity-v14/db-{di:02d}.json');tasks={t['task_record_rowid']:t for t in e['TASK_records']};defs={d['globalTaskId']:d for d in e['COMPUTE_definitions']}
        chosen=[]
        for tid,a in selected[di].items():
            t=tasks[tid];d=defs[t['globalTaskId']];r=lookup[fullkey(a[3],d['opType_text'],a[5],a[2],a[6],d)];rid=r['regime_id']
            ids[rid].append([di,tid]);times[rid].append(t['endNs']-t['startNs']);missing[rid]+=not a[11];chosen.append(t)
        def op(t):return defs[t['globalTaskId']]['opType_text']
        def ins(t):return dimensions(defs[t['globalTaskId']]['inputShapes_text'])
        def outs(t):return dimensions(defs[t['globalTaskId']]['outputShapes_text'])
        def ln(t):return line(selected[di][t['task_record_rowid']][5])
        def fam(t):return family(selected[di][t['task_record_rowid']][5])
        def comp(t):return dict(TASK_rowid=t['task_record_rowid'],operator=op(t),field=FIELDS[ln(t)],startNs=t['startNs'],endNs=t['endNs'],streamId=t['streamId'])
        def groups(ts,n,counts,dest,field):
            ts=sorted(ts,key=lambda t:(t['startNs'],t['task_record_rowid']));assert len(ts)%n==0
            for i in range(0,len(ts),n):
                gg=ts[i:i+n];assert Counter(op(t)for t in gg)==counts,(di,field,i,Counter(op(t)for t in gg))
                if field=='CP':rows=outs(next(t for t in gg if op(t)=='Sub'))[0][0]
                elif field=='COUNT':rows=outs(next(t for t in gg if op(t)=='ReduceSum'))[0][0]
                elif field=='LENGTH':rows=outs(next(t for t in gg if op(t)=='FloorDiv'))[0][0]
                elif field=='COORD':rows=ins(next(t for t in gg if op(t)=='ClipByValueV2'))[0][0]
                else:rows=outs(next(t for t in gg if op(t)=='GatherElementsV2'))[0][0]
                dest.append(dict(database_index=di,rows=rows,stages=[comp(t)for t in gg],scope='Typedsource-orderedfieldTASKgroups,notactualtensoralias/value/CPrequest-windowequivalence. ScalarFillsoutsidecoordinatecorekeptseparately.'))
        groups([t for t in chosen if fam(t)=='CP_LENGTHS'],5,{'Sub':1,'ClipByValueV2':1,'Greater':1,'Fill':1,'SelectV2':1},cp,'CP')
        for l,dest,name in [(1362,cos,'COS'),(1368,sin,'SIN')]:
            groups([t for t in chosen if fam(t)=='C2_FIELDS'and ln(t)==l],3,{'BroadcastTo':1,'Cast':1,'GatherElementsV2':1},dest,name)
        groups([t for t in chosen if fam(t)=='BATCH_AND_COUNTS'and ln(t)==1116],4,
            {'Cast':2,'GreaterEqual':1,'ReduceSum':1},recount,'COUNT')
        groups([t for t in chosen if fam(t)=='SWA_FIELDS'and ln(t)in [1041,1042]],5,
            {'Cast':2,'FloorDiv':1,'BroadcastTo':1,'FloorMod':1},length,'LENGTH')
        corelines=[1008,1017,1018,1019,1021,1025,1026,1028]
        groups([t for t in chosen if fam(t)=='SWA_FIELDS'and ln(t)in corelines and op(t)!='Fill'],14,
            {'Cast':4,'GreaterEqual':1,'ClipByValueV2':1,'ViewCopy':2,'FloorDiv':1,'SelectV2':3,'BroadcastTo':1,'FloorMod':1},coords,'COORD')
    assert [len(x)for x in [cp,cos,sin,recount,length,coords]]==[2792,104,104,69,208,2601]
    models=[]
    for r in rs:
        rid=r['regime_id'];assert len(ids[rid])==r['executions']and missing[rid]==r['missing_PMU']
        d=r['signature'];o=r['operator'];a=dimensions(d['inputShapes_text']);z=dimensions(d['outputShapes_text']);it=d['inputDataTypes_text'].split(';');ot=d['outputDataTypes_text'].split(';')
        on=math.prod(z[0])if z and z[0]is not None else None
        ib=[None if s is None or t not in DT else math.prod(s)*DT[t]for s,t in zip(a or [],it)]
        ob=[None if s is None or t not in DT else math.prod(s)*DT[t]for s,t in zip(z or [],ot)]
        fld=FIELDS[line(r['role'])];f=family(r['role']);io=None;work={}
        if o=='GatherElementsV2':
            assert on is not None
            io=on*(DT[it[0]]+DT[it[1]]+DT[ot[0]])
            work=dict(selected_value_elements=on,index_elements=math.prod(a[1]),source_table_descriptor_bytes=DT[it[0]]*math.prod(a[0]),
                source_width=z[0][-1],contract='C2cos/sinselectedrowgather. Whole1048576×64FLOATtableextentnottraffic;expandedINT64index→nativeINT32conversion/errorpolicyretainedinnewrowlookup.')
        elif o=='ViewCopy':
            n=math.prod(a[4]);io=2*DT[it[0]]*n
            work=dict(selected_column_elements=n,base_storage_elements=a[0][0],contract='OnlyNINT32page/offsetcolumnvaluesread/write;16384outputbaseisN2bankcapacitynotfullcopy. Stride2/differentcolumns/prefix-tailobserversretained.')
        elif o=='BroadcastTo':
            io=(ib[0]if ib[0]is not None else 0)+ob[0]
            work=dict(output_elements=on,source_known_bytes=ib[0],contract='C2indexN→N×64physicallymaterializedvsintegerdivisorBOOLscalarbroadcast;source/attrsnormalizationdistinct. Newrowlookup/constantdivisioncanfoldprivateexpansion;allerror/otherreader/aliasgates.')
        elif o=='Cast':
            io=ib[0]+ob[0]if ib and ib[0]is not None and ob[0]is not None else None
            work=dict(elements=on,source_dtype=it[0],output_dtype=ot[0],contract='Native32↔64scalarpromotion/pagefloor/narrowing/flatbankINT64,BOOLpredicatesandexpandedRoPEindicesdifferentfields. Cancelonlywithdomain/range/nativeconversionandallreaderproof.')
        elif o in ['Fill','ZerosLike']:
            io=ob[0];work=dict(value_stores=on,read_input_values=0,contract='ScalarFill1isnotN-rowFill. Validfalse/skipupdates/-1sentinel/inactive-length/sourcepos0havefield-specificconsumers;zero-outputlengthdoesnotdeleteeventfrontiers.')
        elif o=='Range':
            io=ob[0];work=dict(index_values_generated=on,reference_index_adds=on,contract='arange(B)requestlivevsarange(N)tokenvaliddifferentdomains. Keepwidth/start/step/callerbounds;inlineaddressgenerationcanavoidprivateplane.')
        elif o=='ReduceSum':
            io=ib[0]+ob[0];work=dict(reduction_adds=math.prod(a[0])-on,output_count_elements=on,
                contract='CountnonnegativeSWAindicesafterCast/GE/Cast;notrawproducerlengthwithoutclamp(V,0,C)andquery-domainproof. Source0/1inputINT32sumfitscurrentC256,keepnativecountdtype.')
        else:
            assert o in ['Add','Sub','FloorDiv','FloorMod','Greater','GreaterEqual','Less','Equal','LogicalAnd','SelectV2','ClipByValueV2'],o
            io=sum(x for x in ib+ob if x is not None)
            work=dict(pointwise_output_elements=on,integer_divisions=on if o=='FloorDiv'else 0,integer_remainders=on if o=='FloorMod'else 0,
                contract='Perfieldexacttypedinteger/predicate/selectlaw;FloorDivfloor,FloorModremainder,signednegativeinvalidslots,skip/actualend/parityandINT32wrapkept. Scalar/emptydescriptoroperandbytesnotguessed.')
        models.append(dict(regime_id=rid,module_candidates=r['module_candidates'],operator=o,role=r['role'],origin=r['origin'],context=r['context'],
            signature=d,field=fld,family=f,executions=r['executions'],native_TASK_keys=ids[rid],native_TASK_ns=stat(times[rid]),missing_PMU=missing[rid],
            known_logical_value_bytes_excluding_unknown_control_scalars=io,work_model=work,
            consumer_fusion_review_complete_in_this_turn=True,physical_numeric_ceiling_proved=False))
    write(O/'metadata-field-regime-models-v68.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),regimes=models,
        regime_count=586,captured_execution_count=67631,regimes_by_family=dict(Counter(r['family']for r in models)),
        executions_by_family={f:sum(r['executions']for r in models if r['family']==f)for f in sorted(set(r['family']for r in models))},
        scope='SWA/CP/C2/physicalslot/batchfields;opaqueSMLA/QLImetadataoperatorsandDSpark/RoPE/accessorothersremainpending. Sourcecontiguouscopy/publishedwritesoutsidetheComputeInfolayeraccountedseparately,notinventedTASKjoins.',overall_complete=False))
    write(O/'metadata-field-native-chains-v68.json',dict(CP_length_groups=cp,C2_cos_lookup_groups=cos,C2_sin_lookup_groups=sin,
        SWA_recount_groups=recount,C2_length_quotient_residual_groups=length,slot_coordinate_core_groups=coords,
        interpretation='2792CP5TASK/104cos3+104sin3/69recount4/208length5/2601coordinatecore14. Coordinate3scalarFillsareseparateall2880perline,notassumedNlengthorpairedbynearesttime;fullpublishedbanks/readinessknownbysource.',overall_complete=False))
    print(dict(regimes=586,exec=67631,CP_groups=len(cp),cos_sin=[len(cos),len(sin)],SWA_recount=len(recount),lengths=len(length),coordinate_core=len(coords),family_exec={f:sum(r['executions']for r in models if r['family']==f)for f in sorted(set(r['family']for r in models))}))

if __name__=='__main__':main()
