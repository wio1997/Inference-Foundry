"""Saved embedding, HC initialization and input-ID operator identities/work models."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import math,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import fullkey,dimensions
from analyze_projection_ceiling_v61 import stat
DT={'DT_BF16':2,'FLOAT':4,'INT64':8,'INT32':4,'INT8':1,'BOOL':1}

def main():
    assert not (O/'embedding-initialization-regime-models-v67.json').exists()
    old={r['regime_id']:r for r in read(O/'operator-analysis-coverage-ledger-v66.json')['regimes']}
    mods=[['M05'],['M06'],['M19'],['M04'],['M10'],['M05','M21']]
    regs=[r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']if r['module_candidates']in mods and old[r['regime_id']]['state'].startswith('REGIME_')]
    assert len(regs)==236 and sum(r['executions']for r in regs)==7616
    lk={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in regs}
    ass=defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
        if a[3]in mods:ass[a[0]][a[1]]=a
    times=defaultdict(list);ids=defaultdict(list);missing=Counter();embedding=[];target=[];draft=[];sample=[];draft_ids=[];positions=[];unknown=[]
    for di in range(32):
        e=read(CACHE/f'analysis/native-execution-identity-v14/db-{di:02d}.json')
        ts={r['task_record_rowid']:r for r in e['TASK_records']};ds={r['globalTaskId']:r for r in e['COMPUTE_definitions']}
        selected={}
        for tid,a in ass[di].items():
            t=ts[tid];d=ds[t['globalTaskId']];key=fullkey(a[3],d['opType_text'],a[5],a[2],a[6],d)
            if key not in lk:continue
            r=lk[key];rid=r['regime_id'];times[rid].append(t['endNs']-t['startNs']);ids[rid].append([di,tid]);missing[rid]+=not a[11];selected[tid]=a
            if d['opType_text']=='MemSet' or a[3]==['M05','M21']:unknown.append(dict(database_index=di,TASK=t,definition=d,assignment=a))
        def op(t):return ds[t['globalTaskId']]['opType_text']
        def ins(t):return dimensions(ds[t['globalTaskId']]['inputShapes_text'])
        def outs(t):return dimensions(ds[t['globalTaskId']]['outputShapes_text'])
        def comp(t):return dict(TASK_rowid=t['task_record_rowid'],operator=op(t),startNs=t['startNs'],endNs=t['endNs'],streamId=t['streamId'])
        def groups(tasks,size,counts,dest,scope,fields):
            tasks=sorted(tasks,key=lambda t:(t['startNs'],t['task_record_rowid']))
            assert len(tasks)%size==0
            for i in range(0,len(tasks),size):
                g=tasks[i:i+size];assert Counter(op(t)for t in g)==counts,(di,scope,i,Counter(op(t)for t in g))
                dest.append(dict(database_index=di,stages=[comp(t)for t in g],scope=scope,**fields(g)))
        m05=[ts[i]for i,a in selected.items()if a[3]==['M05']and op(ts[i])!='TensorMove']
        groups(m05,14,{'Cast':5,'GreaterEqual':1,'Less':1,'LogicalAnd':1,'Mul':2,'Sub':1,'LogicalNot':1,'GatherV2':1,'MaskedFill':1},embedding,
            '272ordered14TASKembeddinggroups;immutablemaskargument/sourceprogramcontract,notactualtensorpointer/owner-tokenhistogramproof.',
            lambda g:dict(N=outs(next(t for t in g if op(t)=='GatherV2'))[0][0],H=5120,weight_vocab_rows=16160))
        m06=[ts[i]for i,a in selected.items()if a[3]==['M06']]
        groups(m06,5,{'Tile':1,'ZerosLike':1,'Fill':1,'BroadcastTo':1,'ViewCopy':1},target,
            '104targetinitializations:localHC4repeatandonehotFP32premix,notallHCpre/readersalreadyprivate.',
            lambda g:dict(local_rows=outs(next(t for t in g if op(t)=='Tile'))[0][0]))
        m19=[ts[i]for i,a in selected.items()if a[3]==['M19']]
        groups(m19,15,{'Tile':1,'ZerosLike':2,'Cast':3,'PadV3':3,'MemSet':3,'Fill':1,'BroadcastTo':1,'ViewCopy':1},draft,
            '104draftinitializations;3N/AMemSetskeptunpriced. Masknew_zerosleaf149positivelyobservesNonebranch;provided-maskbranchstillretained.',
            lambda g:dict(N=ins(next(t for t in g if op(t)=='Tile'))[0][0],local_rows=outs(next(t for t in g if op(t)=='BroadcastTo'))[0][0]))
        m04=[ts[i]for i,a in selected.items()if a[3]==['M04']]
        groups([t for t in m04 if any('('+str(k)+')'in selected[t['task_record_rowid']][5]for k in [1896,1899])],4,
            {'IndexCheck':1,'Index':1,'LinearIndex':1,'ScatterElementsV2':1},sample,
            '72sampleIDgather→scattergroups. Output8192descriptorisstoragecapacity,semanticwritesKonly;bounds/duplicates/aliasanddraft-after-sampleorderrequired.',
            lambda g:dict(K=ins(next(t for t in g if op(t)=='ScatterElementsV2'))[2][0]))
        groups([t for t in m04 if any('('+str(k)+')'in selected[t['task_record_rowid']][5]for k in [1918,1920,1923])],6,
            {'Slice':1,'Cast':1,'IndexCheck':1,'Index':1,'LinearIndex':1,'ScatterElementsV2':1},draft_ids,
            '72draftIDSlice/Cast/gather/scattergroups. SourceB×5isstridedviewofBmax×6;selectedflatindexboundsB×5,notbasecapacity.',
            lambda g:dict(K=ins(next(t for t in g if op(t)=='ScatterElementsV2'))[2][0],source_BD=math.prod(outs(next(t for t in g if op(t)=='Slice'))[0])))
        groups([t for t in m04 if '(1599)'in selected[t['task_record_rowid']][5]],4,
            {'IndexCheck':1,'Index':1,'Cast':1,'Add':1},positions,
            '88positiongather32→cast64→add64groups. IndexCheckcapacity4/sourceintegersandallpositions/tailreadersrequired;notframeworkworkflowaudit.',
            lambda g:dict(N=outs(next(t for t in g if op(t)=='Add'))[0][0]))
    assert [len(x)for x in [embedding,target,draft,sample,draft_ids,positions]]==[272,104,104,72,72,88]
    rows=[]
    for r in regs:
        rid=r['regime_id'];assert len(times[rid])==r['executions']and missing[rid]==r['missing_PMU']
        d=r['signature'];o=r['operator'];a=dimensions(d['inputShapes_text']);z=dimensions(d['outputShapes_text'])
        it=d['inputDataTypes_text'].split(';');ot=d['outputDataTypes_text'].split(';');on=math.prod(z[0])if z and z[0]is not None else None
        ib=[None if x is None or t not in DT else math.prod(x)*DT[t]for x,t in zip(a or [],it)];ob=[None if x is None or t not in DT else math.prod(x)*DT[t]for x,t in zip(z or [],ot)]
        io=None;complete=True;model={}
        if o=='MemSet':
            complete=False;model=dict(work=None,bytes=None,destination=None,contract='ConstantPadNdMemSetnameandcallerdoNOTrevealoperand/size/workspaceprotocol. Reuse32DBv63schemaevidence:noargs/memorytrace.4regimes312staypending;notwholePadextentbytes.')
        elif r['module_candidates']==['M05','M21']:
            complete=False;io=on;model=dict(BOOL_zero_stores=on,source_field_identity_unresolved=True,
                contract='80graphprefixInplaceZeroBOOL24;24Bstoreworkknown,buttargetfield/readers/alias/mutationgenerationunknown. KeepM05|M21ambiguityandpendingfusiondecision;notall-falsemasksubstitutebyancestry.')
        elif o=='GatherV2':
            n,h=z[0];io=4*n*h+8*n;model=dict(N=n,H=h,BF16_selected_table_read_bytes=2*n*h,BF16_output_bytes=2*n*h,
                table_descriptor_bytes=2*math.prod(a[0]),contract='272vocabGatherusesselectedrows,invalidIDsaddressrow0thenMaskedFill+0. Unique/cacheDandownervalidUunknown;not16160×5120trafficpercall. Standardinput-IDarity/nativecastandotherembeddingplansremain.')
        elif o=='MaskedFill':
            io=2*DT[it[0]]*on+(math.prod(a[1])*DT[it[1]]if a[1]is not None else 0)
            model=dict(output_elements=on,conventional_read_write_valuation_not_compulsory=True,
                contract='EmbeddingrowmaskbroadcastN×1zeroesnonownertokens;input-IDplaceholderreplacesexact-1beforefullclamp. Keepinvalid/+0/NaN/othersignedIDs,allrawIDbankreaders andwriteextent;notentireparentstorage.')
        elif o=='Cast':
            io=on*(DT[it[0]]+DT[ot[0]])if on is not None else None
            model=dict(elements=on,source_scalar_or_empty_descriptor=on is None,contract='Embedding3INT32→64and2BOOL→64casts,IDs64→32andBOOL→INT8→BOOLpaddingseparate. ScalarpaddingTruecastemptyshapeisnotguessed0/1GMbyte;preservecast-width/0-1mask/INT32narrowpolicy.')
        elif o=='Tile':
            n=math.prod(a[0]);io=2*(n+on);model=dict(input_BF16_elements=n,output_BF16_elements=on,
                contract='HC4logicalrepeat,notarithmetic4×projection. TargetlocalNrepeatvsdraftglobalNrepeat-beforeSPdistinct. CommuteSPcopy/zero-padthenrepeatwithsameowners/masks/aliases/capture;laterHCresidualreadersremain.')
        elif o=='ZerosLike':
            io=on*DT[ot[0]];model=dict(stores=on,read_value_elements=0,contract='PremixFP32N×4thencolumn0=1;draftmaskNone→falseprefix;dummyattentionmetadataNone→zerooutput;positionpaddingINT64zero. Differentfields/predicates/readers,notsharedzeroCSE.')
        elif o=='Fill':
            io=on*DT[ot[0]];model=dict(stores=on,contract='Premixscalar1vsquery_start_locunusedtail-1vsseq_lensunusedtail0. Shapeoperandnotvalueinput;scalarGMloadunbound. Preserveeachprefix/tailowner/captureandattentionmetadatareaders.')
        elif o=='BroadcastTo':
            io=(ib[0]or 0)+ob[0];model=dict(output_elements=on,source_unique_value_bytes=ib[0],contract='PremixFP32scalar1→Ncolumnbeforestridedcopy;onehotfullwriteorreadonlyconstantABIcanfold,notgenericbroadcastelimination.')
        elif o=='ViewCopy':
            count=math.prod(a[4]);io=2*DT[it[0]]*count;model=dict(selected_source_elements=count,base_storage_elements=a[0][0],contract='OnlyNFP32column0writesintoN×4premix,notwholebase;restinitialized0. Allinputsreadonly/otherobserversneededforconstantbank/newHCABI.')
        elif o=='PadV3':
            io=ib[0]+ob[0];model=dict(input_value_bytes=ib[0],output_value_bytes=ob[0],
                contract='DraftHC/IDzeropadvsBOOLmaskviaINT8padTrue. Newownerinitializercanavoidglobalpaddingmaterialization;keepallprovidedmaskvalues,tailzeroID/True-mask,unchangedfullinputreaders. SeparateMemSetsunpriced.')
        elif o in ['TensorMove','Slice']:
            io=2*DT[ot[0]]*on;model=dict(selected_elements=on,selected_read_write_value_bytes=io,
                contract='TargetSPclone(localH)vsinputsembedspersistentbankfullN×HcopyvsdraftIDB×5selectedslice. View/producerdirectoutrequiresalias/readiness/capture/lifetimeandallreaderproof,notdeletebyname.')
        elif o=='Index':
            io=2*DT[ot[0]]*on+DT[it[-1]]*math.prod(a[-1]);model=dict(selected_elements=on,
                contract='Selectedcomputed32/requestsample/draft32rows;sourcebounds/validnegativewrapanderrorsretained. Fusedgather-cast-addorselectedstridedIDgathercanavoidprivateoutputs;notfullparenttabletraffic.')
        elif o=='IndexCheck':
            count=math.prod(a[1]);io=8*count;model=dict(index_elements=count,range_comparisons=2*count,
                contract='Advancedindexnegativewrap/bounds/errorsemanticsandcapacitykept. Errorflag/sync/scalarsunpriced;8Njustindexvaluereadreference,notcompulsoryHBM.')
        elif o=='LinearIndex':
            io=12*on;model=dict(elements=on,contract='Scatterint64indices→privateINT32linearaddresses1Dcapacity8192;source_indexINT64ABIunchanged. Fusedstoresneedcapacity/overflow/negative-destinationerrors/order;notallfullbaseindices.')
        elif o=='ScatterElementsV2':
            k=math.prod(a[2]);io=(DT[it[1]]+DT[it[2]]+DT[ot[0]])*k;model=dict(semantic_selected_stores=k,base_capacity_elements=on,
                source_and_index_and_selected_store_value_bytes=io,contract='Inplaceself.input_ids.gpu.scatter_ writesK,8192outputdescriptorisnotfulltraffic. Nativeimplementationmaytouchmore;newdirectgather-storepreserveolduntouchedprefix/tail/duplicateorder/sample-before-draftoverlapandallbankreaders.')
        else:
            assert o in ['GreaterEqual','Less','LogicalAnd','LogicalNot','Mul','Sub','Add','Equal','ClipByValueV2'],o
            io=sum(x for x in ib+ob if x is not None);model=dict(output_elements=on,pointwise_arithmetic_or_predicate_elements=on,
                scalar_or_empty_inputs_unpriced=any(x is None for x in ib),contract='Vocabmaskvalidrange/INT64offset,INT32or64position/countarithmetic,placeholder==-1andclamp_min0areseparatecontracts. Preserveoverflow,allnegativeIDs/rawbankobservers/sourceonehotreaders. Sourceconstantsnotinferredfromemptyshape.')
        rows.append(dict(regime_id=rid,module_candidates=r['module_candidates'],operator=o,role=r['role'],origin=r['origin'],context=r['context'],signature=d,
            executions=r['executions'],missing_PMU=missing[rid],native_TASK_keys=ids[rid],native_TASK_ns=stat(times[rid]),work_model=model,
            one_read_write_logical_value_bytes=io,consumer_fusion_review_complete_in_this_turn=complete,physical_numeric_ceiling_proved=False))
    done=[r for r in rows if r['consumer_fusion_review_complete_in_this_turn']];pending=[r for r in rows if not r['consumer_fusion_review_complete_in_this_turn']]
    assert len(done)==230 and sum(r['executions']for r in done)==7224
    assert len(pending)==6 and sum(r['executions']for r in pending)==392
    write(O/'embedding-initialization-regime-models-v67.json',dict(regimes=rows,regime_count=236,captured_execution_count=7616,
        scoped_reviewed_regimes=230,scoped_reviewed_executions=7224,explicit_pending_regimes=6,explicit_pending_executions=392,
        MemSet_pending_regimes=4,MemSet_pending_executions=312,BOOL_prefix_field_pending_regimes=2,BOOL_prefix_field_pending_executions=80,
        scope='Operator-onlyM05/M06/M19/M04/M10plusM05|M21ambiguousprefix;alltypedlogicalwork/consumer/branchconditions,notframeworkaudit.',overall_complete=False))
    write(O/'embedding-initialization-native-chains-v67.json',dict(embedding_groups=embedding,target_HC_initializations=target,draft_initializations=draft,
        sampled_ID_groups=sample,draft_ID_groups=draft_ids,position_groups=positions,
        limits='Source-orderedtypedTASKgroupsnotoperandaddress/actualindexorrequest-generationproof. 104draftmaskNonebranchpositivesdistinctfrom80unresolvedgraphprefixboolclears. MemSetbyte/scopeunproved.',overall_complete=False))
    write(O/'initialization-profile-insufficiency-v67.json',dict(records=unknown,new_MemSet_regimes=4,new_MemSet_executions=312,BOOL_field_regimes=2,BOOL_field_executions=80,
        reused_raw_schema_audit='router-MemSet-profile-insufficiency-v63.json',scope='Keepall6pending. NativeMemSetConstantPadNdname/caller/nearbyextentnotaddress/length;prefixBOOLzeros24knownworkbutunknownfield/readersstillchangesfusiondecision.',overall_complete=False))
    print(dict(regimes=236,executions=7616,reviewed=[230,7224],pending=[6,392],embedding=272,target_init=104,draft_init=104,sample_ID=72,draft_ID=72,position=88))

if __name__=='__main__':main()
