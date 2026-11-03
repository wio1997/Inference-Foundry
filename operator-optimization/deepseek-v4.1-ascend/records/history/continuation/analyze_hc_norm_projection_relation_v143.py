"""Typed HC/input-RMS/projection cuts and owner-versus-recipient alternatives."""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib,json,struct,math
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]


def bf16(x):
    u=struct.unpack('<I',struct.pack('<f',x))[0];return struct.unpack('<f',struct.pack('<I',(u+32767+((u>>16)&1))&0xffff0000))[0]


def schedule(p,carrier='baseline',early_kv=False,reprice=False):
    hc=max(p['hc_y'],p['hc_coeff'])
    if carrier=='baseline':g=hc+p['owner_norm']+4+p['gather']
    elif carrier=='owner_fused':g=hc+p['owner_norm']+(p['extra']if reprice else 0)+p['gather']
    else:g=hc+p['gather']+p['recipient_norm']+(p['extra']if reprice else 0)
    qa=g+p['qa'];kv=qa+p['kv'];qr=qa+p['qr_quant']+(p['extra']if reprice and early_kv else 0);qb=max(kv,qr)+p['qb']
    kt=(kv if early_kv else max(kv,qr))+p['kv_tail']+(p['extra']if reprice and early_kv else 0)
    return dict(QA=qa,KV=kv,QR_DQ=qr,QB=qb,KV_PUBLISHED=kt,Query_RoPE=max(qb,kt)+2,Post_input=max(p['hc_coeff'],max(qb,kt)+5))


def main():
    assert not(O/'REVIEW-CHECKPOINT-v143.json').exists()
    prof=read(O/'hc-norm-projection-native-frontiers-v143.json');idx={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-index-query-weight-native-producers-v139.json')['records']};root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/model.py',791,831),('models/deepseek_v41/model.py',855,907),('attention/dsa_v41.py',319,385),('attention/dsa_v41.py',396,412),('attention/dsa_v41.py',439,494),('attention/dsa_v41.py',503,534),('models/deepseek_v41/compressor.py',35,45),('models/deepseek_v41/compressor.py',55,71)]:
        p=root/rel;parts.append(dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b]))
    wrapper=W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/layernorm.py';parts.append(dict(**pin(wrapper),first_line=94,last_line=104,source=wrapper.read_text().splitlines()[93:104]))
    facts=[
      'No-residual RMS wrapper can add optional bias after BF16 RMS result;keep that BF16Add and loadedbeta/replica/readers gates if applicable. Kernel rstd islocally discarded inthisdirectwrapper,notproofallnative/debug/hookobserversdead. The real scalar-cancellation formula below assumesbeta0;it doesnot coverbeta branch. Current chain HC Y_BF16 -> inputRMS5120 Z_BF16 -> SP allGather -> QA1280 BF16 / SWAKV512 BF16. QA norm1280 BF16 -> native DynamicQuant INT8+FLOAT -> QB BF16 -> tail64 QueryRoPE. InputRMS,QRnorm andKVnorm are distinct reduction/parameter/rounding domains.',
      '960eager frontiers retain actual M800/816/834 localowner rows andQ800/816/834/830 CPquery rows versusglobalcacheN6400/6528/6668. Same H5120 does not make their token domains equal. New raw/normalized/projected carrier choices require exact SP/CP/request/epoch/weight matching,not onlycounts.',
      'Current HC coefficient normalization consumes full4H X,not weighted collapsed Y. Equal raw4H sumsq cannot determine mean(Y^2) because weighted crosschannel terms differ. Initial exact four-way repeat+onehot has a restricted real identity ofmeans;it requires actual equalchannel/finite/mask/premix/epsilon/reduction/nativehandoff conditions,not all layers.',
      'InputRMS norm_eps equals HC norm_eps in reviewed caller,while HC epsilon for mixes is a differentparameter. Sharing a sourceparameter doesnot attest identical native reduction order or available rstd ABI. Existing HC outputs do not expose a new input-normalizer rstd handoff by default.',
      'In exact real arithmetic let r=(mu(Y)+eps1)^-1/2 andv=Bdiag(gamma1)Y. A downstream RMS(r v;gamma2,eps2)=gamma2 v/sqrt(mu(v)+eps2/r^2) forr>0. Deletingr withunchanged eps2 changesfunction;adaptiveepsilon still depends onfullH inputstat. BF16 normalized-Z rounding beforeGEMM independently blocks this unrounded refactor.',
      'Fusion may retain BF16 Y,calculatefullH inputRMS,roundZ_BF16,thenhandit tobothmatrixconsumers usinga proved AIV/AIC/UB/L1/GM interface. EliminatingaGMmaterializationdoesnoteliminate full-row statistics,typed rounds orresourcecapacity;provisional/recompute/correction schedules needtheir owncosts,not an invented universal wholeoperatorbarrier.',
      'Owner HC+RMS emits normalizedZ before rowcommunication andkeeps ownerM normwork. Recipient RMS+projection afterrawY allGather normalizesN rowsperrank;separate QA/KV recipient normalizers may normalizeN+Q. RawY andZ carrier are alternative graphs;cannot claim both localinterfacecredits without addingdualoutputs/work.',
      'QA/SWA joint panel can avoidone duplicatedread onlyonactualsharedrows. QA-on-allN overcomputes N-Q rows unlessa tagged/ragged mixedpanel map is introduced. Full-row norm mustround each outputgroup separately;QA1280/SWA512/C1512 are not one1792/2304norm.',
      'C1 projection reads same normalized cache-domain Z buthasdistinctweights and512norm;C2 reads FLOAT cast of roundedZ andtwoFLOATprojections/poolstate.192positive indexWeight frontiers are additional Zqueryreaders. PrivateZ deletion needs every matchingreaderinternal orcompatibleextraoutput,notjust QA+SWA.',
      'QueryRoPE belongs after QB BF16output,withno postQB RMS512 inthissource. InputRMS cannotbe algebraicallypushed through QA1280norm/DQ/QB/RoPE asone linear map. QR floatingreaders/IndexQ andnativequant scales remain;F02-F07 is higher-order composition,not a new direct inputNorm->RoPE edge.',
      'Source multistream QA precedes KV whileQRnorm/quant overlapsKV;part3gatesKVnorm aswell asQB. KVnorm semanticdata needs KV output,not QRquant. Releasingit earlier changesVectorcompetition;it is not free overlap ornewphysicalgain. Source CStable/query/index/cache lastreader/queuedwriter contracts remain.',
      'CurrentHC rawX/currentcoeff/attentionPost residual/pre4 are retained. Normalizedcarrier completion andnew projectionreadyfrontiers do notretire those inputs orclose padding/emptyQ/cache/metadata/error/nativeprecision branches.'
    ]
    write(O/'hc-norm-projection-source-contract-v143.json',dict(parts=parts,facts=facts,prior_answers=['v61 typed projection/numerical wrappers','v94 QRnorm/nativequant precision','v99 Npanels/Kcross terms/groupstats','v133/v134 ownercarrier/HCfork','v135/v136 C1/C2 consumers','v139 extraIndexWeight/QRreaders','v140 QueryRoPE join/version'],new_device_work=False,all_runtime_or_new_native_gates_closed=False,overall_complete=False))
    observed=[];geoms=Counter();other=Counter();work=[]
    for r in prof['records']:
        key=r['database_index'],r['forward_API_rowid'],r['source_block_ordinal'];s=r['stages'];m=int(s['INPUT_NORM']['definition']['inputShapes_text'].strip('"').split(',')[0]);q=int(s['QA']['definition']['inputShapes_text'].strip('"').split(',')[0]);n=int(s['SWA_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]);geoms[m,q,n]+=1
        observed.append(dict(key=list(key),M=m,Q=q,N=n,QA_norm_native_bound=s['QA_NORM']['TASK']is not None,QA_quant_native_bound=s['QA_QUANT']['TASK']is not None,C1='C1_PROJ'in s,C2='C2_KV'in s,IndexWeight_positive=key in idx,normalizer_rstd_pointer_or_actual_same_Z_subrows_bound=False))
        other['C1']+='C1_PROJ'in s;other['C2']+='C2_KV'in s;other['INDEX_WEIGHT']+=key in idx
    for(m,q,n),count in sorted(geoms.items()):
        h=5120;work.append(dict(M=m,Q=q,N=n,frontiers=count,H=h,owner_input_norm_rows=m,recipient_union_norm_rows=n,separate_recipient_QA_KV_norm_rows=n+q,recipient_extra_norm_rows=n-m,
          current_Y_private_interface=4*m*h,current_Z_sender_private_interface=4*m*h,current_global_Z_union_interface=4*n*h+2*q*h,
          owner_HC_RMS_chosen_interface_credit=4*m*h,rawY_recipient_union_chosen_interface_credit=4*m*h+2*q*h,
          joint_QA_KV_shared_row_input_read_reference=2*q*h,rawY_recipient_full_row_interface=4*n*h,
          QA_fullN_extra_classical_ops=2*(n-q)*h*1280,QA_KV_owner_classical_ops=2*m*h*(1280+512),QA_KV_current_classical_ops=2*h*(q*1280+n*512),
          changed_normalizer_plane_or_actual_HBM_saved=None,all_Z_other_readers_closed=False,physical_capacity_or_speedup=None))
    write(O/'hc-norm-projection-observed-and-work-v143.json',dict(records=observed,geometry=work,other_Z_positive_reader_frontiers=dict(other),logical_scope='Declared materialized interfaces/value reads only,not physical HBM/alloc/time lowerbounds. Baseline localY->RMS4MH,localZ->sender4MH,globalZ->QA/KV4NH+2QH. Owner HCnorm removesfirst4MH only. RawYcarrier+oneunionrecipientnorm/panel useslocal4MH+global4NH,so chosenreference difference4MH+2QH whileduplicatingnormalizer rowsN-M. These alternatives sharecredits andcannotbeadded. Other Zreaders/collectiveownership/private-output gatesnotclosed.',bound_scope='Classical direct dense references only,not allalgorithm necessary work. Validphysicalcapacity upper andnecessary nonduplicated work/readsets required forL;sum standalone bounds not allowed. Actual PMU/criticaltime unknown.',new_device_work=False,overall_complete=False))
    y=[1.,0.3359375]+[0.]*5118;mu=f32(f32(sum(f32(a*a)for a in y))/5120);r=f32(1/math.sqrt(f32(mu+1e-6)));zf=[f32(a*r)for a in y[:2]];zb=list(map(bf16,zf));original=f32(f32(zb[0]*0.3359375)-zb[1]);unrounded=f32(f32(zf[0]*0.3359375)-zf[1]);assert original!=unrounded
    eps1=eps2=1e-6;rr=1/math.sqrt(.5+eps1);v=1e-4;correct=rr*v/math.sqrt((rr*v)**2+eps2);skip=v/math.sqrt(v*v+eps2);adapt=v/math.sqrt(v*v+eps2/(rr*rr));assert abs(correct-adapt)<1e-14
    write(O/'hc-norm-projection-stat-and-round-witnesses-v143.json',dict(
      HCraw_not_collapsed=dict(Xplus=[[1,0],[1,0],[0,0],[0,0]],Xminus=[[1,0],[-1,0],[0,0],[0,0]],premix=[1,1,0,0],raw_mean_both=.25,Yplus=[2,0],Yminus=[0,0],collapsed_mean_plus=2,collapsed_mean_minus=0),
      initial_repeat_stat=dict(E=[2.,-3.],repeat=[[2.,-3.]]*4,premix=[1,0,0,0],raw_mean=6.5,collapsed_mean=6.5,only_initial_exact_repeat_domain=True,loaded_native_rstd_abi_or_reduction_equality_attested=False),
      real_RMS_linear_RMS=dict(Y=[1.,0.],gamma1=[1.,1.],B=[1e-4,0.],eps1=eps1,eps2=eps2,r=rr,v=v,correct=correct,illegal_delete_input_scalar=skip,adapt_epsilon=eps2/(rr*rr),adaptive_epsilon_result=adapt,source_native_precision_identity=False),
      BF16_Z_GEMM=dict(H=5120,Y_prefix=y[:2],zeros=5118,gamma=1.,eps1=1e-6,mean_f32=mu,rstd_f32=r,Z_FLOAT_prefix=zf,Z_BF16_prefix=zb,B_BF16_prefix=[0.3359375,-1.],projection_from_rounded_Z=original,projection_from_unrounded_Z=unrounded,actual_frozen_weights_or_native_gemm_policy=False),
      prefix_not_fullH=dict(prefix=[1.,0.],suffixA=[0.,0.],suffixB=[3.,0.],meanA=.25,meanB=2.5,epsilon=1e-6,first_normalized_A=1/math.sqrt(.25+1e-6),first_normalized_B=1/math.sqrt(2.5+1e-6),provisional_or_correction_recompute_algorithms_excluded=False),
      gamma_cannot_commute=dict(Ycases=[[1,0],[0,1],[1,-1]],gamma=[1,2],B=[1,1],raw_projection=[1,1,0],correct_after_feature_gamma=[1,2,-1],universal_scalar_gamma_exists=False,real_gamma_fold_into_weights_allowed=True),
      witness_scope='Constructed real/FLOAT/BF16 semantics,not actual loaded values,native parity,tolerance authorization or physical bounds.',new_device_work=False,overall_complete=False))
    cases=[];classes=Counter();credits=Counter()
    for vals in product([2,10],[5,17],[1,7],[1,35],[3,14],[2,9],[3,16],[4,20],[2,9],[3,12],[0,9,40]):
        p=dict(zip(['hc_y','hc_coeff','owner_norm','recipient_norm','gather','qa','kv','qr_quant','qb','kv_tail','extra'],vals));ans={k:schedule(p,c,e,pr)for k,c,e,pr in [('baseline','baseline',False,False),('owner_fused','owner_fused',False,True),('early_KV_only','baseline',True,True),('owner_joint','owner_fused',True,True),('recipient_joint','rawY',True,True)]};b=ans['baseline']['Post_input'];v=ans['recipient_joint']['Post_input'];classes['RECIPIENT_FASTER'if v<b else'RECIPIENT_SLOWER'if v>b else'TIED']+=1
        cases.append(dict(parameters=p,**ans))
    write(O/'hc-norm-projection-readiness-DAGs-v143.json',dict(cases=cases,count=len(cases),recipient_joint_classes=dict(classes),scope='Plain positive-query two-projection reference only;C1/C2/IndexWeight andallotherreaders mustbeprovided/retained separately. CurrentHCcompletionwaitsY+coeff inallmodels,F01earlyforknotrewardedagain. Timesarbitrary;ownerinputnorm/rankrecipientnorm services independentlypriced,not inferred fromrowcounts. Part3earlyKV modeledwithaddedVectorpenalty.',endpoint='AttentionPost input-ready incl currentcoeff;actualPost/nextFFNpre4 andsamplingnotpriced',not_physical_lower_or_upper_bound=True,new_device_work=False,overall_complete=False))
    pairs=[['F02',x]for x in ['F05','F06','F07','F08','F33']];contracts=[
      'HC inputRMS5120 and QA1280norm/nativeDQ/QB compose onlywith allBF16rounds/fullrowstats/quant-scale/readers;rawHCstat or downstreamhomogeneity cannotunconditionallycancel inputRMS.',
      'Owner HCnorm,recipientrawYnorm+QA/KV union andownerprojectedcarriers aredifferentgraphs. RecipientnormalizerduplicatesN-M rows;actual CP/SP QA subset,panelstrides,allZreaders andresource repricing required.',
      'InputRMS to QueryRoPE isahigher-order path throughQA,independentQRnorm,nativeDQ,QB andBF16tail64rotation;not adirectlinear fold,keepQR/IndexQreaders/scales and currentSWAjoin/CSepochcontracts.',
      'InputRMS+SWA512projection/norm/RoPE/cache maysharehand-off whilekeepinginputH5120stat,SWA512stat,bothBF16rounds,positions/slot/mask/noQcachewrites andnewproducer/communication ABI.',
      'InputRMS output feeds distinct C1/SWA matrices andseparate512norms;C1 mayjoin a typed shared-row panel butIndex/longfanout/oldvalueWAW andallZ/C2/IndexWeightreaders remain. Full1024 jointnorm orfusedallrowsQA notdefault equivalence.'
    ]
    ex=[['HG73-Q','F02'],['HG73-Q','F07'],['HG73-Q','F08'],['HG73-Q','F33'],['HG87-KV-CURRENT-READ-AND-COMMIT','F02']]
    write(O/'hc-norm-projection-composition-v143.json',dict(clauses=[dict(id='REL143-0'+str(i+1),a=a,b=b,contract=c)for i,((a,b),c)in enumerate(zip(pairs,contracts))],new_pairs=pairs,new_pair_clauses=5,group_extensions=ex,all_runtime_gates_proved=False,whole_pair_or_all_subsets_closed=False,alternative_carrier_gains_additive=False,overall_complete=False))
    names=['hc-norm-projection-native-frontiers-v143.json','hc-sp-eager-native-frontiers-v134.json','hc-rope-latent-native-frontiers-v140.json','hc-c1-native-consumers-v135.json','hc-c2-native-consumers-v136.json','hc-index-query-weight-native-producers-v139.json','hc-sp-source-and-branch-contract-v133.json','hc-sp-eager-typed-carrier-and-padding-models-v134.json','norm-quant-operator-source-contract-v94.json','norm-quant-precision-and-row-models-v94.json','norm-quant-variants-v94.json','projection-panel-source-contract-v99.json','projection-panel-reduction-and-precision-v99.json','F33-projection-algebra-and-layout-v27.json','embedding-initialization-operator-contract-v67.json','hc-c2-norm-state-source-contract-v137.json','candidate-relation-clauses-v142.json','hc-draft-postcheck-result-v142p.json']
    write(O/'hc-norm-projection-inputs-v143.json',dict(inputs=[pin(O/n)for n in names],questions=['Can HC raw4H norm statistics or downstream512/1280norm remove inputRMS5120 undercurrentBF16 semantics?','Whichnormalized/raw/projected rowcarrier allows InputRMS/projectionfusion,andwhichrowwork/interfacesarealternatives?','WhichC1/C2/IndexWeight/QR readers prevent treatingnormalizedZ as private toQA+SWA?','Howdo inputnormalizer cuts compose withnativeQRquant,QB/RoPE andactualSWAstreamjoin without additivegain?'],profile_first=True,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='MODELED',frontiers=960,TASK_bindings=prof['TASK_bindings'],new_QR_source_roles=prof['new_QR_source_roles'],other_readers=other,DAGs=len(cases),classes=classes,new_pairs=5,overall_complete=False)))


if __name__=='__main__':main()
