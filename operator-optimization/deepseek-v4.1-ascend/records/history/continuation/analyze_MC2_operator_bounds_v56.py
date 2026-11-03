"""Exact MC2/shared operator profile plus separate count-work/communication models."""
from datetime import datetime,timezone
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,math,statistics
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE


def stat(xs):
    a=sorted(xs);return dict(n=len(a),minimum=a[0],median=statistics.median(a),p90=a[int(.9*(len(a)-1))],maximum=a[-1])if a else None


def main():
    assert not(O/'MC2-operator-profile-v56.json').exists()
    stamp=datetime.now(timezone.utc).isoformat();inputs=read(O/'MC2-operator-inputs-v56.json')
    assert len(inputs['operator_source'])==8 and all(r['status']=='READ'for r in inputs['operator_source'])
    graph=read(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json')
    mc2={(r['database_index'],r['task_record_rowid']):r for r in graph['MC2_executions']}
    shapes=defaultdict(list);bound=[];unbound=[];pending=defaultdict(list)
    for g in graph['graph_executions']:pending[g['database_index']].append(g)
    for i in range(32):
        native=read(CACHE/f'analysis/native-execution-identity-v14/db-{i:02d}.json')
        tasks={r['task_record_rowid']:r for r in native['TASK_records']};defs={r['globalTaskId']:r for r in native['COMPUTE_definitions']}
        for r in graph['MC2_executions']:
            if r['database_index']==i:shapes[r['definition']['inputShapes_text']].append(r['endNs']-r['startNs'])
        for g in pending[i]:
            for b in g['blocks']:
                routes=[mc2[(i,t)]for t in b['task_record_rowids']if(i,t)in mc2];assert len(routes)==1
                route=routes[0];after=[]
                for tid in b['task_record_rowids']:
                    t=tasks[tid];d=defs[t['globalTaskId']]
                    if t['startNs']>=route['endNs'] and d['opType_text']in ['DynamicQuant','QuantBatchMatmulV3','DequantSwigluQuant','ForeachAddList']:
                        after.append(dict(op=d['opType_text'],start=t['startNs'],end=t['endNs'],duration_ns=t['endNs']-t['startNs'],task_rowid=tid,streamId=t['streamId'],shapes=d['inputShapes_text']))
                after.sort(key=lambda r:r['start']);assert Counter(r['op']for r in after)=={'DynamicQuant':1,'QuantBatchMatmulV3':2,'DequantSwigluQuant':1,'ForeachAddList':1}
                assert after[-1]['op']=='ForeachAddList'
                premerge_span=after[-2]['end']-after[0]['start']
                shared_span=after[-1]['end']-after[0]['start'];route_ns=route['endNs']-route['startNs'];endpoint=after[-1]['end']-route['startNs']
                row=dict(database_index=i,graph_ordinal=g['graph_ordinal'],block_ordinal=b['block_ordinal'],route_task_rowid=route['task_record_rowid'],
                    route_duration_ns=route_ns,shared_stages=after,shared_span_ns=shared_span,shared_premerge_span_ns=premerge_span,endpoint_span_ns=endpoint,
                    unconstrained_no_work_change_overlap_cap_ns=min(route_ns,premerge_span),
                    source_scope='ExactgraphlayerTASKs;sharedworkcurrentlyafterMC2,samestream. Overlapcap excludesdependentfinaladd andassumesindependentreadyinputs/unlimitedresources;not achievablegain.')
                (bound if g['prepare_root_keys']else unbound).append(row)
    assert len(bound)==2560 and len(unbound)==640 and sum(map(len,shapes.values()))==4472
    write(O/'MC2-operator-profile-v56.json',dict(generated_utc=stamp,MC2_executions=4472,
        shape_timings=[dict(input_shapes=s,executions=len(v),duration_ns=stat(v))for s,v in shapes.items()],
        root_bound_graph_layers=bound,phase_unbound_graph_layers=unbound,
        root_bound_route_ns=stat([r['route_duration_ns']for r in bound]),root_bound_shared_span_ns=stat([r['shared_span_ns']for r in bound]),
        root_bound_shared_premerge_ns=stat([r['shared_premerge_span_ns']for r in bound]),
        root_bound_endpoint_span_ns=stat([r['endpoint_span_ns']for r in bound]),
        unlimited_resource_overlap_cap_ns=stat([r['unconstrained_no_work_change_overlap_cap_ns']for r in bound]),
        routed_and_shared_activation_are_different=True,no_profile_internal_phase_timing_proved=True,new_capture=False,overall_complete=False))
    source_parts=[]
    ranges={
       'dispatch_ffn_combine.h':[(199,248),(258,273)],
       'dispatch_ffn_combine_kernel.hpp':[(214,253),(313,359),(435,537),(539,638),(793,978),(981,1023),(1107,1145)],
       'dispatch_ffn_combine_tiling.cpp':[(198,214)],
       'block_epilogue_pertoken_swiglu.hpp':[(204,281)],
    }
    for r in inputs['operator_source']:
        p=Path(r['local_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        lines=p.read_text().splitlines()
        for lo,hi in ranges.get(p.name,[]):source_parts.append(dict(path=str(p),sha256=r['sha256'],first_line=lo,last_line=hi,source=lines[lo-1:hi]))
    write(O/'MC2-kernel-boundary-and-tiling-contract-v56.json',dict(generated_utc=stamp,parts=source_parts,
        facts=[
          'Onekernel alreadyfuses routingquant,rankdispatch,GMM1,SwiGLU+requant,GMM2,dequant-return andweightedunpermute. SharedDequantSwigluQuant is a distinctbranch.',
          'A/B areINT8;C/GMM1 andGMM2local intermediate isfloat16_t,notexternalBF16. SwiGLU castsFP16toFP32,multipliespertokenscale,clamps/SiLU thenRINTquantizesINT8 withFP32scale.',
          'GMM1 andGMM2 use actualper-expertcurrentM,capacity-clipped byprefix;zeroM bypassesblockMmad. currentM<=128 requestsCACHE_MODE_DISABLE onweights.',
          'GMM1→SwiGLU usesgmC;requant→GMM2 usesgmPermutedToken;GMM2→dequant usesgmC2. Singlekernel isnot zeroGMintermediate.',
          'Workspace aliasesptrC==ptrC2 andptrA==ptrPermutedToken. Moreoverlap/neworder requires oldreader completion androw/stride ownership;cannot freely allocate one bank per stage.',
          'L1tile128x256x512 andL0tile128x256x128 are sourceconstants,not proofeverytinyM executes128fullrows. Expert readiness andtwoepiloguegroups haveexplicitcross-coreflags.',
          'Defaultfirstepiloguegroup isE_local-3;target24→21+3,draft8→5+3. It isexpert-index based,not balancedbyactualtokens/work.',
          'Dispatch CopyGMToGMPerToken copiesquantizedH+32byte records withoneFP32scale embedded;H+4 issemanticminimum,not currentrecordstride.',
          'Returnepilogue writesexternalBF16peer rows;source rank waitsCrossRankSync andperformsweightedunpermute usingprobs. Movingweightedaggregation acrossrank changesmetadata/order/numericsunless separatelypreserved.'],
        source_identity='Currentoriginal166stage filesystem withW8historicalcontrol;not per-execution loadedbinary proof.',
        conditional_operator_questions_answered=['Existingfusion boundary','InternalGMinterfaces andaliasing','InternalFP16/FP32/INT8 boundaries','Zeroexpertskip andL2hintpredicate','Expert-indexpipelinecut andlogicaldispatchrecord'],
        unknown=['Actualpercallinternalphasecost,instructiontiling/issuecounts andHBMtransactions','Currenthostprofper-tokenexpertchoices/n_e/capacity/aliases/events','Allnewkernelprecision/hardwarehandoff/supportandtypedphysicalcapacities'],overall_complete=False))
    counts=read(N/'MC2-operator-source-v56/saved-observer/EP-count-distribution.json')
    routes=read(N/'MC2-operator-source-v56/saved-observer/EP-routes.json')
    lookup={(r['role'],r['layer'],r['ordinal'],r['phase']):r for r in routes['cohorts']}
    models=[];H=5120;I=2304;W=3*H*I;S=8*(2*I+H);P=752e12;B=1.6e12
    for r in counts['cohorts']:
        if r['phase']!='formal_C8_measure':continue
        key=(r['role'],r['layer'],r['ordinal'],r['phase']);rr=lookup[key]
        ne=r['receiver_expert_rows_by_logical_id'];E=r['E_local'];world=16;k=r['K'];matrix=rr['sender_receiver_assignment_matrix'];V=r['valid_sender_rows_by_rank']
        assert len(ne)==E*world and sum(ne)==r['total_assignments']
        A=[sum(ne[j*E:(j+1)*E])for j in range(world)];active=[sum(x>0 for x in ne[j*E:(j+1)*E])for j in range(world)]
        assert A==r['receiver_rows_by_rank'] and max(A)<65536
        assert all(sum(matrix[s])==V[s]*k for s in range(world))
        assert [sum(matrix[s][t]for s in range(world))for t in range(world)]==A
        remote=sum(matrix[s][t]for s in range(world)for t in range(world)if s!=t)
        dlow=sum(math.ceil(matrix[s][t]/min(k,E))for s in range(world)for t in range(world)if s!=t)
        dhigh=sum(min(V[s],matrix[s][t])for s in range(world)for t in range(world)if s!=t)
        assert 0<=dlow<=dhigh<=remote
        rank_models=[]
        for j in range(world):
            group=ne[j*E:(j+1)*E];cut=E-3 if E>4 else E-1;prefix=sum(group[:cut]);activeprefix=sum(x>0 for x in group[:cut])
            f=6*H*I*A[j];q=(W+S)*active[j]
            rank_models.append(dict(rank=j,receiver_assignments=A[j],active_native_experts=active[j],
                dense_INT8_ops_2perMAC=f,one_read_active_weights_bytes=W*active[j],packedscale_bytes=S*active[j],
                conditional_reference_compute_us=f/P*1e6,conditional_reference_cold_weight_us=q/B*1e6,
                GM_internal_interface_logical_bytes=A[j]*(10*I+4*H),
                first_epilogue_group_experts=cut,first_group_row_fraction=prefix/A[j]if A[j]else None,
                first_group_active_expert_fraction=activeprefix/active[j]if active[j]else None))
        models.append(dict(role=r['role'],layer=r['layer'],ordinal=r['ordinal'],native_expert_namespace='Legacyfieldname sayslogical;use observednativephysical dispatch IDs,linearplacement/0redundancy only.',
            receiver_work_by_rank=rank_models,total_assignments=sum(A),active_global_experts=sum(active),
            remote_assignments=remote,unique_remote_token_destination_pairs_lower=dlow,unique_pairs_upper=dhigh,
            current_dispatch_record_bytes_model=remote*(H+32),current_return_BF16_bytes_model=remote*2*H,
            hypothetical_coalesced_dispatch_minimum_bytes_interval=[dlow*(H+4),dhigh*(H+4)],
            coalescing_model_scope='DistincttopKexperts,validrowsandpreservedrouting;matrixmarginalsbounduniquetoken-destinationcounts butdo notattestjointchoices. Networkheaders/control/peermemtransactions excluded.',
            straggler_reference_floor_us=max(max(x['conditional_reference_compute_us'],x['conditional_reference_cold_weight_us'])for x in rank_models)))
    assert Counter(r['role']for r in models)=={'target':80,'draft':7}
    summary={}
    for role in ['target','draft']:
        rs=[r for r in models if r['role']==role];ranks=[x for r in rs for x in r['receiver_work_by_rank']]
        summary[role]=dict(cohorts=len(rs),active_experts_per_rank=stat([r['active_native_experts']for r in ranks]),
            max_received_assignments=stat([max(x['receiver_assignments']for x in r['receiver_work_by_rank'])for r in rs]),
            conditional_straggler_floor_us=stat([r['straggler_reference_floor_us']for r in rs]),
            first_epilogue_row_fraction=stat([r['first_group_row_fraction']for r in ranks if r['first_group_row_fraction']is not None]),
            first_epilogue_active_weight_fraction=stat([r['first_group_active_expert_fraction']for r in ranks if r['first_group_active_expert_fraction']is not None]),
            current_dispatch_data_MB=stat([r['current_dispatch_record_bytes_model']/1e6 for r in rs]),
            current_return_data_MB=stat([r['current_return_BF16_bytes_model']/1e6 for r in rs]))
    write(O/'MC2-receiver-work-and-communication-model-v56.json',dict(generated_utc=stamp,cohorts=models,summary=summary,
        per_expert_INT8_weight_bytes=W,per_expert_packedscale_bytes=S,per_assignment_GEMM_ops=6*H*I,
        per_assignment_internal_logical_interface_bytes=10*I+4*H,
        formulas=['A_r=sum_e n_e onreceiver r,not senderM*topK. DenseGMM1+GMM2 ops=6HI*A_r.',
            'Activeweight loweraccounting forstatedone-read/coldmodel=3HI*E_r_plus;allocated24-expert storage isnot invocationreadvolume.',
            'GMinterface descriptor model:GMM1FP16 roundtrip8I +activatedINT8roundtrip2I +GMM2FP16roundtrip4H,perreceiverassignment;not measuredHBMbytes.',
            'T>=max_r(max(6HI*A_r/P_int8,(3HI+8*(2I+H))*Eplus_r/B_HBM,Q_link/B_link,criticaldependencybounds)) underdeclarednecessarywork/cold/capacity assumptions.',
            'For remotesender→receiverrank:ceil(assignments/min(topK,E_local))<=distincttoken-destinationpairs<=min(valid_sender_rows,assignments). Sum interval bounds are conservative,not actualduplicate rate.'],
        population='87formalcohorts in observer-20261030T160315Z;separate fromlaterhostprof tasktimings. No samecalljoin/efficiency/speedup inference.',
        capacity='Observer savedreceipt reports65536and0snapshotsabove;selectedcountmodelsmaxrank<65536. Laterprofileattr_1also65536;not a timingjoin.',
        packet_and_bytes='Currentdispatchdata recordH+32=5152B,semanticINT8+scaleH+4=5124B;returnBF16H=10240B. These are code-levelpayloadmodels,not physicalwirepackets.',
        reference='Reusehardware-reference-scope-v55.json:752INT8TOPS,1.6TB/s publishedreference,notactualcurrent capacity.',
        conditional_variants=[
            dict(id='F21-FINE',change='Finerreceiverpipeline /onchipintermediatehandoff',constraints='PreserveinternalFP16round/FP32scale/clamp/rowamax/RINT,aliasedA/activationandC1/C2lastuse,all flags/credit/capacity. Currentkernelalreadyfused;newhand-offmustbe supported.'),
            dict(id='F21-CUT',change='Receiverwork-awareepiloguecut instead offixed21+3 or5+3 expertindices',constraints='Useexistingcounts withoutnewhostsync;preserveexpertrowoffsets/scale/routingmapping/aliasepochs/nativeflags. Rowcountalone isnotcost;activeweightandperexperttileloads matter.'),
            dict(id='F21-COALESCE',change='Quantizetokenonce /coalesceremote copiesperdestination andfanoutlocally',constraints='Needjointper-tokenroutingmap,newreceiverformat/index/quant-scaleownership,all controlparticipantsandmask/capacity. Marginalmatrix cannotpricetheactualsaving.'),
            dict(id='F21-PARTIAL',change='Receiverweightedpartialcombine beforereturn',constraints='Not anunconditionalbitwise fusion:BF16/FP32rounding andsumorder change,weightsmust reachreceiver,newpayloadmaydoublewidth. Preservedeclaredprecisionorclassifyasnumericalalgorithmchange.'),
            dict(id='F22-OVERLAP',change='Schedule sharedbranchduringroutedcomm/compute gaps',constraints='CurrentMC2==1disablesexistingoverlap;bothreserveCube/Vector/BW. Profileoraclecap isnotfreegain;newcompatibleschedule andjointresource bound required.')],
        implementation=False,native_test=False,measured_speedup=None,whole_module_closed=False,overall_complete=False))
    print(json.dumps(dict(MC2_executions=4472,root_bound_layers=len(bound),source_parts=len(source_parts),formal_count_cohorts=len(models),summary=summary,overall_complete=False)))


if __name__=='__main__':main()
