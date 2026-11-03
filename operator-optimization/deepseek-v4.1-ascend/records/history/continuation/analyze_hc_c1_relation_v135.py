"""HC/C1-SWA compound graph, typed delivery cuts and readiness costs."""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib,json
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def timing(r):
    s,b,c,g,gp,w,i,l,wo,extra=[r[k]for k in ['s','b','c','g','gp','w','i','l','wo','extra']]
    joint_arrival=max(s,b)+gp;split_s=s+g;split_b=max(b,split_s)+g
    joint=max(c,max(joint_arrival+1,joint_arrival+w+i,joint_arrival+w+l,3)+2)
    split=max(c,max(split_s+1,split_b+w+i,split_b+w+l,3)+2)
    encoded_arrival=max(s,b+wo+1)+gp+extra
    encoded=max(c,max(encoded_arrival+1,3)+2)
    return dict(joint=joint,ordered_split=split,owner_encoded=encoded)


def main():
    assert not(O/'REVIEW-CHECKPOINT-v135.json').exists()
    prof=read(O/'hc-c1-native-consumers-v135.json');assert prof['frontier_count']==24
    root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/model.py',417,425),('models/deepseek_v41/model.py',724,728),('models/deepseek_v41/model.py',887,920),
        ('models/deepseek_v41/compressor.py',16,24),('models/deepseek_v41/compressor.py',69,71),('attention/dsa_v41.py',487,494),('attention/dsa_v41.py',506,566),('models/deepseek_v41/indexer.py',106,125)]:
        p=root/rel;parts.append(dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b]))
    write(O/'hc-c1-source-and-precision-contract-v135.json',dict(parts=parts,prior_answers=['v27distinctmatrices/two512norms/old-latentWAR','v93HCcoeff/externalYfork','v99partialpanelrelease','v134largeMtypedcarrier'],
        facts=['SWAReplicatedLinearandC1nn.Linearareseparatelyconstructedbias-freeH5120→D512BF16sourceobjectswithseparatenorms. Thisisnotpostloadvalueinequalityproof. CSEneedsequalparameters/epochs/rounds/readers;shared-inputtilingdoesnotneedW_A=W_B.',
        'HCcoeffusesraw4HoriginalX/FLOAT-HF32/RMS/SinkhornwhileSWA/C1usenormalizedroundedBF16Y. Sharedrawstateloadisnotonesame-inputwideGEMMorautomaticcurrentcoefffactorizationthroughY.',
        'KeepYBF16→fullH5120normBF16→twoGEMMBF16outputs→twoindependentD512normBF16andindexWkBF16→D128normBF16→RoPE/quant/FP16scale. No1024RMSorunroundedaccumulatorinputsubstitution.',
        'IndexerWkreadsoldunrotatedC1latentbeforeinplacelongRoPEoverlappingwrites,unlesspreservedcopy/newoutofplacefork. Applyperrange/row,notuniversalwholeindexpublicationbarrier. Current24recordspositivelytakefallbackINT8key/FLOAT16scalestores.',
        'CurrentHCpost/combdeadlineatPostandpre4atnextFFNremain. OriginalHCXstillneededbycoeff/Postresidual;carrier/latentcompletiondoesnotretireX. NewencoderworkcancontendwithcoeffanddelayPost.',
        'Packed1024outputhasrowstride1024for512views;nativecurrentcontiguousshapedoesnotprovestridedsupport. Directseparateoutputs/newnormepiloguemayavoidadaptersundernewlayout/visibilitycontracts.',
        'OwnerprecomputeofindexWk/norm/RoPE/quant/FP16scaleisaconditionalnewdeliverycut. Requiresreceiverparamreplicas,position/slot/quant/nativeM/capture/epochandallunrotatedlatentreadersclosed;notautomaticfromsourceReplicateddeclarations.',
        'Sendingencodedresultsdoesnotdelaycurrentcachevisibilityuntilacceptance. Actualrequestedrows/readsetsmayallowsmallerpipelinesbutareunbound. Empty/!owns_k/packed/foldedcache/C2domainsremainseparate.'],
        no_actual_weight_alias_new_M_numeric_or_all_observer_proof=True,new_device_work=False,overall_complete=False))
    geoms=Counter()
    for r in prof['records']:
        p=r['eager_HC_SP_parent'];n=p['SP_allGather']['COMM']['count']//5120;N=int(p['SWA_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]);q=int(p['QA']['definition']['inputShapes_text'].strip('"').split(',')[0]);geoms[n,N,q]+=1
    models=[]
    for (m,n,q),count in sorted(geoms.items()):
        # m=sender rows,n=currentglobalprojectionrows,q=currentCPQArows.
        models.append(dict(sender_rows=m,global_rows=n,QA_rows=q,frontiers=count,
            old_hidden_sender_content=10240*m,latent_cut_sender_content=2048*m,index_norm_cut_sender_content=2304*m,encoded_endpoint_cut_sender_content=2178*m,
            with_QA_latent_content=4608*m,with_QA_index_norm_content=4864*m,with_QA_encoded_endpoint_content=4738*m,
            encoded_extra_vs_latent_content=130*m,index_norm_extra_vs_latent_content=256*m,
            current_C1_index_WK_dense_ops=2*n*512*128,new_owner_index_WK_dense_ops=2*m*512*128,
            current_two_projection_input_value_reads=4*n*5120,joint_input_value_read_reference=2*n*5120,
            optional_packed_to_two_contiguous_projection_value_IO=8*n*512,
            two_BF16_projection_weight_value_content=2*2*5120*512,HC_FLOAT_coeff_weight_value_content=4*20480*24,
            union_HC_two_projection_weight_value_content=12451840,with_QA_union_weight_value_content=25559040,
            index_WK_weight_value_content=131072,with_index_union_matrix_value_content=12582912,with_QA_and_index_union_matrix_value_content=25690112,
            actual_HBM_wire_allocated_bytes_or_necessary_all_algorithm_work=False))
    write(O/'hc-c1-delivery-cuts-and-work-v135.json',dict(models=models,
        cuts=[dict(id='PROJECTION_BEFORE_NORMS',planes='SWAprojected512BF16+C1projected512BF16',bytes_per_sender_row=2048,receiver_work='Twooriginal512norms,thenindexandlongbranches;Wreplica/newMneeded,normgammacanremainrecipient-specific.'),
            dict(id='LATENT',planes='SWA512BF16+C1preRoPE512BF16',bytes_per_sender_row=2048,receiver_work='Wk128/norm128/RoPE/INT8quant/FP16scale/indexstoreandlongRoPE/store;moving512normsrequiresmatchingrecipientnormparameters.'),
            dict(id='INDEX_NORM',planes='SWA512BF16+longC1RoPE512BF16+indexpreRoPE128BF16',bytes_per_sender_row=2304,receiver_work='IndexRoPE/quant/FP16scale/storeandlongstore'),
            dict(id='ENCODED_ENDPOINTS',planes='SWA512BF16+longC1RoPE512BF16+indexkey128INT8+scaleFLOAT16',bytes_per_sender_row=2178,receiver_work='Indexkey/scale/longstores;SWAroleRoPE/storeasdeclared')],
        gates='Namedfallbackunpackedcachecutsonly;normalnewM/sourceposition/param/quantizer/rowtrim/scale/slot/duplicates/epochs/oldreaderclosure. MovingonlyGEMMdoesnotrequiremovingrecipient-specific512norms;movingnorm/indexencodingrequiresitsadditionalparameter/numericgates. No actual nativeequivalenceorminimalwiretrafficproved. QA1280adds2560B/rowunlessqueryrawneedsareprovedlocallycovered.',
        local_joint='Priorv27sharedinput/packed1024directionretained. Two512norms,old-latentread,FP16scalecastandHCfull4Hcoeffworkstay. Adapter8NDvalueIOisoptionalundernewABI,notHBMnecessarytraffic.',
        resource_cut='Oneunionofdistinctuncompressedparameterplanesinadeclaredmandatoryservicedomain,oneinitialresidencygrant;T>=sup_lambda sum(lambda_j*W_j)/beta_joint(lambda)onlywithnecessaryworklowersandvalidjointcapacityupper. ActualmixedHF32/BF16/Vector/linkbetaunknown.',
        theory_status='SYMBOLIC_CEILING',new_device_work=False,overall_complete=False))
    cases=[];classes=Counter();witness={}
    for s,b,c in product([1,4,12],repeat=3):
        for g,mult,w,i,l,wo,extra in product([0,2],[1,2],[1,4],[1,3],[1,3],[1,4],[0,2]):
            r=dict(s=s,b=b,c=c,g=g,gp=g*mult,w=w,i=i,l=l,wo=wo,extra=extra);r.update(timing(r));cases.append(r)
            tag='JOINT_FASTER'if r['joint']<r['ordered_split']else'SPLIT_FASTER'if r['joint']>r['ordered_split']else'TIED';classes[tag]+=1;witness.setdefault(tag,r)
    write(O/'hc-c1-packet-readiness-models-v135.json',dict(cases=cases,case_count=len(cases),classes=dict(classes),witnesses=witness,
        symbols='s=SWAplane ready,b=C1latent ready,c=HCcoeffPostready,g=oneorderedcollectiveservice,gp=jointservice,w=indexWkreadservice,i=indexnorm/RoPE/quant/storetail,l=longRoPE/storetail,wo=ownerWkreadservice. FixedSWAstore1,queryready3,attention/output2;allhypotheticalunits.',
        domain='Chosenfullcarrieratomicdeliverywithoneorderedcommunicatorforseparatemessages,equalpricedserviceandotherwiseunlimitednoncontendingresources. EndpointisattentionHCPost;laterpre4/FFNserviceisnotpriced. Endpointsownerencodeadds1parallelbranchserviceandencodedpacketextra. Notactualrankepisodes,hardwareoptimal-timeorlatencybounds;repriceCprime/queues/pack/params.',
        current_index_read_before_overlapping_long_write_retained=True,no_universal_whole_operator_barrier=True,new_device_work=False,overall_complete=False))
    write(O/'hc-c1-layout-and-old-value-witnesses-v135.json',dict(
        packed_view=dict(N=2,D=2,packed=[10,11,20,21,30,31,40,41],C1_offset=2,correct_C1_indices=[2,3,6,7],wrong_contiguous_C1_indices=[2,3,4,5],SWA_row1_indices=[4,5],meaning='WronglegacycontiguousC1writerwouldoverwriteSWArow1;notanobservednativealias.'),
        old_latent=dict(latent=[1,0],Wk=[1,0],long_rotation=[0,1],old_index_dot=1,wrong_after_rotation_dot=0,meaning='SourceWARfromv27retainedperrange;copy/outofplacecanrelaxmutationorderwithcost.'),
        norm_domains=dict(SWA=[1,1],C1=[3,3],per_group_rstd=[1,1/3],wrong_combined_rstd=1/(5**0.5),domain='Finiteepsilon0toy;actualeps/gamma/bias/roundskept.'),
        native_contiguity_pointer_or_new_kernel_proved=False,new_device_work=False,overall_complete=False))
    write(O/'hc-c1-joint-graph-and-clauses-v135.json',dict(pair=['F01','F33'],new_pairs=1,new_groups=0,existing_groups_to_extend=['HG93-HC-ATTENTION-FORK-DEADLINE','HG74-C1'],
        edges=[['OLD_MIX','Y_BF16'],['X','Y_BF16'],['X','HC_COEFF'],['HC_COEFF','HC_POST'],['X','HC_POST'],['Y_BF16','NORM_H_BF16'],['NORM_H_BF16','SWA_PROJ'],['NORM_H_BF16','C1_PROJ'],['SWA_PROJ','SWA_NORM512'],['C1_PROJ','C1_NORM512'],['C1_NORM512','INDEX_WK_OLD_READ'],['C1_NORM512','LONG_ROPE'],['INDEX_WK_OLD_READ','LONG_ROPE'],['INDEX_WK_OLD_READ','INDEX_NORM128'],['INDEX_NORM128','INDEX_ROPE_QUANT_SCALE'],['INDEX_ROPE_QUANT_SCALE','INDEX_PUBLICATION'],['LONG_ROPE','LONG_PUBLICATION'],['SWA_NORM512','SWA_PUBLICATION'],['INDEX_PUBLICATION','ATTENTION_OUTPUT'],['LONG_PUBLICATION','ATTENTION_OUTPUT'],['SWA_PUBLICATION','ATTENTION_OUTPUT'],['ATTENTION_OUTPUT','HC_POST']],
        next_FFN_edges=[['HC_COEFF','CURRENT_PRE4'],['CURRENT_PRE4','NEXT_FFN_PRE'],['HC_POST','NEXT_FFN_PRE']],
        edge_scope='Sourcevalue/read-generationrelationsandonechosenfull-demandfrontier;realQLI/SMLAneedonlyactualrequestedrows. KeepperrangeWAR,nouniversalindex-publish→longRoPEbarrier. Carrier/local/endpointcutspricealternativegraphs.',
        no_new_native_graph_nodes=True,one_HC_shared_baseline=True,all_runtime_legal_fusion_or_all_subsets_closed=False,new_device_work=False,overall_complete=False))
    names=['hc-sp-eager-postcheck-result-v134p.json','candidate-relation-clauses-v134.json','candidate-pair-navigation-v134.json','hc-c1-native-consumers-v135.json','hc-sp-eager-native-frontiers-v134.json',
        'F33-consumer-and-numerical-contract-v27.json','F33-condition-ledger-v27.json','F33-joint-projection-variants-v27.json','HC-attention-deadline-and-precision-models-v93.json',
        'projection-panel-release-DAG-models-v99.json','hc-sp-reader-generation-and-row-witnesses-v133.json','hc-sp-eager-typed-carrier-and-padding-models-v134.json','KV-publication-variants-v87.json']
    ins=[O/n for n in names]+[Path(p['path'])for p in parts]
    write(O/'hc-c1-inputs-v135.json',dict(inputs=[pin(p)for p in sorted(set(ins))],profile_first=True,questions=['HowdoHCcurrentcoeffandC1/SWAproducer/carrierforksjoinwithoutdoublecredit?',
        'Whichsourcepositiveold-latentread/RoPE/quant/scalestoresbounddifferentdeliverycuts?',
        'Howdocontiguity/two512norms/ownerencoding/readiness/weight-and-reader-gateschangefusioncosts?'],new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_C1_SWA_CONDITIONALLY_RECONCILED',frontiers=24,geometry_cases=len(models),packet_cases=len(cases),packet_classes=dict(classes),new_scoped_pair=1,new_device_work=False)))


if __name__=='__main__':main()
