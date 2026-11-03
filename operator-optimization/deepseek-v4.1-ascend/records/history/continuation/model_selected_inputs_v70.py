"""Exact selected IO, conditional fused alternatives, and operator def-use witnesses."""
from collections import Counter
from itertools import product
from pathlib import Path
import ast,hashlib
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_state_engram_v62 import dimensions

def main():
    assert not(O/'selected-input-fusion-costs-v70.json').exists()
    p=read(O/'selected-input-regime-models-v70.json');g=read(O/'selected-input-native-chains-v70.json')['chains'];bykey={tuple(k):r for r in p['regimes']for k in r['native_TASK_keys']}
    def rows(x):return [bykey[(x['database_index'],s['TASK_rowid'])]for s in x['stages']]
    def io(x):return sum(r['known_logical_value_bytes_excluding_control']for r in rows(x))
    composed=[];aux=[];prefix=[];tails=[];seed=[];simple=[]
    for x in g['COMPOSED_DRAFT_ID_GATHER']:
        a=[r for r in rows(x)if r['operator']=='Index'];S=dimensions(a[0]['signature']['outputShapes_text'])[0][0];D=dimensions(a[1]['signature']['outputShapes_text'])[0][0];old=io(x);assert old==24*S+24*D
        composed.append(dict(database_index=x['database_index'],S=S,D=D,source_5TASK_known_value_bytes=old,legal_domain_selected_only_fused_reference_bytes=20*D,full_L_validation_with_local_index_reuse_reference_bytes=8*S+12*D,full_L_validation_interface_budget_bytes=16*S+12*D,small_L_local_storage_bytes=8*S,required_INT32_output_bytes=4*D,source_L_still_published_metadata=True,unknown_controls_not_valued=True))
    for x in g['AUX_GATHER_CONCAT']:
        rr=rows(x);c=next(r for r in rr if r['operator']=='ConcatD');T,K=dimensions(c['signature']['outputShapes_text'])[0];assert K==15360;H=5120;old=io(x);assert old==24*T+24*T*H
        aux.append(dict(database_index=x['database_index'],T=T,H=H,source_7TASK_known_value_bytes=old,joint_gather_concat_reference_bytes=4*T+12*T*H,joint_interface_budget_bytes=20*T+12*T*H,identity_prefix_copy_concat_reference_bytes=12*T*H,required_materialized_concat_output_bytes=6*T*H,existing_F27_indirect_main_projection_prep_budget_bytes=old-4*T,source_projection_INPUT_read_not_in_this_7TASK=True,prefix_or_projection_alias_epoch_proof_not_from_shape=True))
    for x in g['AUX_PREFIX_CONCAT']:
        r=rows(x)[0];T,K=dimensions(r['signature']['outputShapes_text'])[0];assert K==15360 and io(x)==12*T*5120
        prefix.append(dict(database_index=x['database_index'],T=T,H=5120,source_concat_known_value_bytes=io(x),direct_F27_projection_prep_value_bytes=0,indirect_projection_input_reads_still_required=True,existing_F27_mechanism_not_new=True))
    for x in g['DRAFT_FIXED_BANK_TAILS']:
        rr=rows(x);assert [r['role'].split('(')[1].split(')')[0]for r in rr]==['1212','1216','1220','1343']
        lengths=[dimensions(r['signature']['outputShapes_text'])[0][0]for r in rr];B=8200-lengths[1];Q=8208-lengths[0];assert B in [2,4]and Q==5*B and lengths[2]==8200-(B+1)and lengths[3]==8192-Q
        assert io(x)==4*sum(lengths)
        tails.append(dict(database_index=x['database_index'],B=B,query_rows=Q,bank_capacities=dict(slot=8208,seq=8200,QSL=8200,sample=8192),active_prefix_lengths=[Q,B,B+1,Q],tail_lengths=lengths,source_4Fill_write_bytes=io(x),dirty_tail_write_bytes='4*sum(max(previous_active_prefix_i-current_active_prefix_i,0))',steady_same_B_dirty_tail_write_bytes=0,first_three_before_metadata_builder=True,last_sample_tail_later=True,target_G13_slot_capacity_not_bound_by_this=True,source_prefix_copies_not_in_4Fill_valuation=True))
    for x in g['MARKOV_SEED_COLUMN']:
        r=rows(x)[0];B=r['work_model']['selected_INT64_seed_elements'];assert B in [2,4]and io(x)==16*B
        seed.append(dict(database_index=x['database_index'],B=B,source_selected_copy_bytes=16*B,source_base_capacity_bytes=192,possible_copy_elision_bytes=16*B,required_conditions='First Markov source uses original seed; dynamic_spec receives correctfullbuffer/seed and no othercol0reader;seedbanklastreader/generation protected. No unconditionaldeadseedclaim.'))
    for f,width in [('TARGET_INPUT_ID_GATHER',4),('TARGET_POSITION_GATHER',8),('NEXT_STEP_POSITIONS',4),('SAMPLE_HIDDEN_ROWS',10240)]:
        for x in g[f]:
            r=next(r for r in rows(x)if r['operator']=='Index');T=r['work_model']['selected_rows'];old=io(x);assert old==(8+2*width)*T
            simple.append(dict(family=f,database_index=x['database_index'],T=T,row_bytes=width,source_2TASK_known_value_bytes=old,same_prefix_copy_reference_bytes=2*width*T,known_index_work_budget_bytes=8*T,view_or_consumer_indirect_prep_bytes=0,identity_condition='paddedtargetarangeormatchingDSparkrealqueryselector;NOTshapeequality/dummyzeroindex.',next_step_value_dead_only_under_parallel_guard=f=='NEXT_STEP_POSITIONS'))
    assert Counter((x['S'],x['D'])for x in composed)=={(12,10):16,(24,20):48,(14,10):8}
    assert Counter(x['T']for x in aux)=={12:16,24:48,6668:8}
    assert Counter(x['T']for x in prefix)=={6400:8,6528:8}
    assert Counter(x['B']for x in tails)=={2:32,4:56}and sum(x['source_4Fill_write_bytes']for x in tails)==11531424
    assert Counter(x['B']for x in seed)=={2:32,4:72}
    # Local AST questions concern only exact producer, consumer and guard clauses.
    contract=read(O/'selected-input-operator-contract-v70.json');src={s['remote']:Path(s['path'])for s in contract['source_files']}
    base=src['/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py'];bt=ast.parse(base.read_text());fn=next(n for n in ast.walk(bt)if isinstance(n,ast.FunctionDef)and n.name=='_propose');parents={c:n for n in ast.walk(fn)for c in ast.iter_child_nodes(n)}
    loads=[n for n in ast.walk(fn)if isinstance(n,ast.Name)and n.id=='used_update_positions'and isinstance(n.ctx,ast.Load)];assert sorted(n.lineno for n in loads)==[1312,1332]
    for n in loads:
        a=n;anc=[]
        while a in parents:a=parents[a];anc.append(a)
        assert any(isinstance(z,ast.If)and isinstance(z.test,ast.Name)and z.test.id=='should_update_next_steps'for z in anc)
    guard=next(n for n in ast.walk(fn)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='should_update_next_steps'for t in n.targets));assert ast.unparse(guard.value)=='not self.parallel_drafting and (self.dcp_size == 1 or dcp_mtp_inputs is not None)'
    dp=src['/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py'];dt=ast.parse(dp.read_text());first=next(n for n in ast.walk(dt)if isinstance(n,ast.FunctionDef)and n.name=='set_inputs_first_pass')
    stores=[n for n in ast.walk(first)if isinstance(n,ast.Name)and n.id=='token_indices_to_sample'and isinstance(n.ctx,ast.Store)];first_store=min(n.lineno for n in stores)
    assert not any(isinstance(n,ast.Name)and n.id=='token_indices_to_sample'and isinstance(n.ctx,ast.Load)and n.lineno<first_store for n in ast.walk(first))
    st=ast.parse(src['/vllm-workspace/vllm/vllm/config/speculative.py'].read_text());parallel=next(n for n in ast.walk(st)if isinstance(n,ast.If)and ast.unparse(n.test)=="self.method in ('dflash', 'dspark')"and any(isinstance(z,ast.Assign)and ast.unparse(z)=='self.parallel_drafting = True'for z in n.body))
    up=ast.parse(src['/vllm-workspace/vllm/vllm/v1/spec_decode/llm_base_proposer.py'].read_text());constructor=next(n for n in ast.walk(up)if isinstance(n,ast.AnnAssign)and ast.unparse(n.target)=='self.parallel_drafting'and ast.unparse(n.value)=='self.speculative_config.parallel_drafting')
    merged=next(n for n in ast.walk(bt)if isinstance(n,ast.FunctionDef)and n.name=='_run_merged_draft');observers=[n for n in ast.walk(merged)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and ast.unparse(n.func)=='self.dynamic_spec.update'and any(k.arg=='draft_token_ids'and isinstance(k.value,ast.Name)and k.value.id=='draft_token_ids'for k in n.keywords)];assert len(observers)==1
    # Pure integer/permutation math; no model, tensor runtime or device import.
    wrap32=lambda x:(x+2**31)%2**32-2**31
    normalize=lambda x,n:x+n if x<0 else x
    compose_cases=0
    for A_len in range(1,6):
        A=list(range(10,10+A_len));vals=sorted(set([-A_len,-1,0,A_len-1]))
        for size in range(1,5):
            for L in product(vals,repeat=size):
                tmp=[A[i]for i in L]
                for j in sorted(set([-size-1,-size,-2,-1,0,size-2,size-1])):
                    jj=wrap32(j+1)
                    if not -size<=jj<size:continue
                    i=normalize(jj,size);li=L[i];direct=A[normalize(li,A_len)];assert direct==tmp[jj];compose_cases+=1
    assert wrap32(2147483647+1)==-2147483648
    A=[10,20];L=[7,0];J=[0];assert A[L[J[0]+1]]==10 and any(not -len(A)<=i<len(A)for i in L)
    aux_cases=0;planes=[[(100*p+10*t,100*p+10*t+1)for t in range(4)]for p in range(3)]
    for count in range(1,5):
        for I in product([-4,-1,0,1,3],repeat=count):
            selected=[[plane[i]for i in I]for plane in planes];baseline=[sum((selected[p][t]for p in range(3)),())for t in range(count)];joint=[sum((planes[p][i]for p in range(3)),())for i in I];assert baseline==joint;aux_cases+=1
    assert [planes[0][0]]*4!=planes[0] # dummy all-zero indices are not identity.
    caps=[8208,8200,8200,8192];pads=[-1,0,0,0];banks=[[pad]*C for pad,C in zip(pads,caps)];previous=[0]*4;dirty_bytes=[]
    for k,B in enumerate([2,4,2,4,4,2]):
        active=[5*B,B,B+1,5*B];written=0
        for i,(C,pad,old,new)in enumerate(zip(caps,pads,previous,active)):
            banks[i][:new]=[1000+k]*new
            banks[i][new:old]=[pad]*max(old-new,0)
            assert banks[i]==[1000+k]*new+[pad]*(C-new)
            written+=4*max(old-new,0)
        dirty_bytes.append(written);previous=active
    assert dirty_bytes==[0,0,96,0,0,96]
    variants=[]
    def v(i,choice,cost,conditions,state='CONDITIONAL_OPERATOR_OPTION'):
        variants.append(dict(id=i,choice=choice,logical_cost_model=cost,conditions=conditions,decision=state,legal_for_all_actual_values_proved=False,native_speedup=None))
    v('INPUT-COMPOSE','Reusev38composedIDgatherasoneoperator','24S+24D→20D selectedonly OR8S+12D retainingfullLvalidation/localreuse.','L/Jready,INT32J+1wrap/negativeper-levelnormalization;preservevalidationofunselectedL/errorstatusorprovelegaldomain;publishedLmetadataisnotdeleted.')
    v('INPUT-COMPOSE-VALIDATE','Dropfirst-indexvalidationusingonlyselectedL','ExcludedforunprovenfullLdomain: A[10,20],L[7,0],J[0] selectedcompose10butbaselineerrors.','Fullindexvalidityisscope-specific,notinferredbyselectedoutputs.','EXCLUDED_WITH_COUNTEREXAMPLE')
    v('INPUT-AUX-JOINT','Fuse3auxgatherswithconcat','24T+24TH→4T+12TH,materializedoutput6THkept.','All3sourcebounds/negativewrap/order/repeats/readiness/strides;indexonceperrowreference,physicalcache/spillingunbound.')
    v('INPUT-AUX-PREFIX','Usepaddedarangeasthreeprefixviewsbeforeconcat','24T+24TH→12THwhenstableprefixidentity;arbitraryunpaddedselectionseparate.','Sourcearangebankimmutable/rangeandallsourceoldgenerations;viewsnotcopiesrequirelastreader/aliasfences.')
    v('INPUT-AUX-F27','ReuseexistingF27-v34-INDIRECTsingle15360Dprojection','Avoidselectedplanes+concatprep;inputloadandprojection2*T*15360*640perrankstillrequired.','New3source/ordered-rowABI,singleKcontraction,BF16projectionboundary/fullfeatureRMS/TPdataready;donotthreeindependentlyroundedGEMMs. Notanewmissingmechanism.')
    v('INPUT-AUX-ORIGINAL-CONCAT','ReadoriginalthreeprefixplanesdirectlyinF27','16nativeconcat:12THprep→0;T6400budget393216000Bpercalllogical.','Sameexistingindirectcontract;all3auxlastreaders/overlapandoldcheckpointsstilllive;nospeedproportionalbytesclaim.')
    v('INPUT-ID-POS-PREFIX','UseprefixcopyforpaddedID/INT64positiongathers','TwoTASK(8+2w)T→2wT preservescopy,viewcanremoveprivatecopyseparately.','T<=sourcecapacity,arangevalues0:Tnoexternalwriters;samepositionsemantics/MMbranch;othernonpaddedselectionkept.')
    v('INPUT-SHARED-INDEX','Shareindexload/checkandjointpublishID/position/auxfields','EachfieldscanssameIcurrently;oneproducerundernewmultioutABIcanamortize;countoverlapwithauxfusiononce.','Allfieldsreadytogether,sourceextentintersection/negativeper-sourcewrap,copydestinationbanks/typesandlastreader;largerworking-set/latewaitcanloseoverlap.')
    v('INPUT-SAMPLE-HIDDEN','Useidentitysamplevieworindirectheadrowreader','2TASK(8+20480)T→prefixcopy20480TorseparateheadABI0prep.','Onlyactualidentity,not104allcalls;dummyzeroindicesandlmheadpaddingretainrowmap/snapshot;BF16inputandTProwsame.')
    v('INPUT-SHAPE-IDENTITY','Treatsameinput/outputhiddenrowcountasidentity','Excluded: all-zeroindexhasidenticalshapeandduplicatesrow0.','Realqueryproducer/range/aliasproofrequired.','EXCLUDED_WITH_COUNTEREXAMPLE')
    v('INPUT-UNUSED-INITIAL','EliminateQSLminus1temporaryoverwrittenbyDSparkfirstpass','16SubTASKknown16B/call→0valuework.','Matchingoverrideunconditionallyignoresincomingselector,nootherobserver/failureeffects;othermethods/brancheskept.')
    v('INPUT-UNUSED-POS','Guardused_update_positionsbygenuinefuture-stepconsumer','88IndexCheck+IndexgroupsvalueunusedwhenparallelTrue;16Tknownbytescanbeavoidedunderlegaldomain.','Only2loadsitesunder!parallel/CPguard;DSparkconfigparallelTrue+constructorcopylocalSHAsame166revisioncondition;keepboundsvalidationunlessproducerprovesdomain.')
    v('INPUT-TAIL-DIRTY','Maintainfourbankconstanttailsusingexistingv69suffixlemma','88calls4Fill11531424B;steadyBdirtywrites0;B4→2clears24INT32=96B.','Capacities8208/8200/8200/8192areDRAFTbanks,notunknowntargetG13slotC;establishfulltails,allwriters/epochs/dummy/reset/addressrebindtracked;prefixcopyworkstillrequired.')
    v('INPUT-TAIL-COALESCE','Coalescefirstthreetailfieldsorpublishprefix+tailfromproducer','Potential3Fill→1butalltheirwritesremainunlessdirty/elisioncontract.','First3beforemetadatareader,lastsampletailafterotherwork;donotdelayearlywritesuntil4thfillwithoutreaderproof.Producer-outABIcanavoidprivatecopiesonlyallreadersbound.')
    v('INPUT-SEED-DIRECT','FeedfirstMarkovembeddingdirectseedinsteadofcolumn0copy','104ViewCopywritesB8bytes,16Bknowncopytraffic;descriptor192Bnotactualcopy.','dynamic_specreceivesfullbankargumentincludingseedcolumn;mustretainmaterializationoradaptallconsumers/disableguard;seedbankoldgenerationlastreaderprotected.')
    v('INPUT-SEED-RETURN','Deletecolumn0becausereturnsliceexcludesseed','Notproved: dynamic_spec.update(draft_token_ids=fullbuffer) is anotherobserver before return.','Do notinferdeadnessfromreturnalone;internaldynamicconsumerreadsetunbound.','EXCLUDED_WITH_UNCLOSED_CONSUMER')
    source_serve=N/'original-profile-cache-v53/runtime/scripts/serve.sh';assert hashlib.sha256(source_serve.read_bytes()).hexdigest()=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
    write(O/'selected-input-fusion-costs-v70.json',dict(composed_ID_models=composed,aux_gather_concat_models=aux,aux_prefix_concat_models=prefix,draft_tail_models=tails,Markov_seed_copy_models=seed,simple_gather_models=simple,typed_bound_model='T>=typedindex/predicatework/Ptyped,actualHBMbytes/BHBM,actualL2bytes/BL2,launch/dependencycriticalpathinitsdeclaredscope. Nativechainstimesareobservations,notrequestcriticalpath. SavedlogicalbytesarenotmeasuredHBM;unknownindex/Vectorcapacitypreventsunconditionalphysicalusceiling. Nesteddependentloadsdonotvanishbecauseonelaunch.',projection_work_unchanged='F27indirectsinglecontraction2*T*15360*640perTPcard+samefeatureAG/normunlessseparatereorganization. BF16/INT32/INT64modesnotinterchangeable.',not_additive='Auxjoint/prefix/indirectarealternatives;input-indexsharingoverlapandoldF27projection/bankoutbudgetsnotdoublecounted.',whole_modules_closed=0,overall_complete=False))
    write(O/'selected-input-def-use-and-guard-proof-v70.json',dict(used_update_positions_load_lines=sorted(n.lineno for n in loads),all_loads_under_should_update_next_steps=True,should_update_next_steps_expression=ast.unparse(guard.value),DSpark_parallel_config_guard_line=parallel.lineno,parallel_constructor_copy_line=constructor.lineno,first_incoming_selector_store_line=first_store,no_incoming_selector_load_before_overwrite=True,dynamic_spec_full_buffer_observer_line=observers[0].lineno,source_support_version_condition=contract['support_config_binding'],draft_bank_capacity_binding=dict(serve=str(source_serve),serve_sha256=hashlib.sha256(source_serve.read_bytes()).hexdigest(),max_batched_tokens=8192,max_requests=4,num_speculative_tokens=5,slot=8208,seq=8200,QSL=8200,sample=8192,profile_all88four_fill_groups_match=True),scope='Exactoperatorvalueconsumersandguards,notframeworkorwholeloadedcodeattestation.',overall_complete=False))
    write(O/'selected-input-fusion-variants-v70.json',dict(variants=variants,variant_count=len(variants),whole_modules_closed=0,overall_complete=False))
    write(O/'selected-input-math-validation-v70.json',dict(composed_negative_repeat_valid_cases=compose_cases,aux_selection_concat_cases=aux_cases,unselected_invalid_L_counterexample=dict(A=[10,20],L=[7,0],J=[0],naive_selected_value=10,baseline='FirstfullLIndexCheckfails'),INT32_add_wrap=dict(input=2147483647,plus1=-2147483648),dummy_zero_indices_not_identity=True,dirty_B_sequence=[2,4,2,4,4,2],dirty_tail_write_bytes=dirty_bytes,all72composed_72aux_88tail_104seed_costs_MATCH=True,no_model_Torch_NPU_native_test=True,overall_complete=False))
    print(dict(composed=72,aux=72,prefix=16,tails=88,seed=104,simple=len(simple),variants=len(variants),math=[compose_cases,aux_cases],tail_native_write_bytes=11531424,largest_aux_logical_budget=max(x['joint_interface_budget_bytes']for x in aux)))

if __name__=='__main__':main()
