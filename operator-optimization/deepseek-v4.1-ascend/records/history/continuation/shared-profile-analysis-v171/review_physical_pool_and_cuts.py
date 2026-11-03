"""Bind current device domains, price shared cuts, keep clock ceilings honest."""
import collections
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path
H=Path(__file__).resolve().parent
O=H.parent/'profile-first-20261031'

def read(name): return json.loads((H/name).read_text())
def write(name,d): (H/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
acl=read('ACL-device-metadata-query.json')['review']
meta=read('physical-mapping-and-capture-metadata.json')
topo=read('topology-platform-frequency-query.json')
capture=read('captured-group-device-rows.json')['rows']
bridge=read('cohort-device-bridge-inputs.json')
dcmi=read('dcmi-frequency-observation.json')['review']
assert acl['aclInit_rc']==acl['aclFinalize_rc']==0 and acl['device_count']=={'rc':0,'value':16}
assert len(acl['rows'])==len(capture)==len(meta['captured_device_info'])==16
mapping={}
for line in meta['commands'][0]['text'].splitlines():
    a=line.split()
    if len(a)>=5 and all(x.isdigit() for x in a[:4]):
        card,chip,logic,phy=map(int,a[:4]);mapping[logic]={'NPU_card_id':card,'chip_id':chip,'driver_physical_device_id':phy,'chip_name':a[4]}
assert set(mapping)==set(range(16)) and len({x['NPU_card_id'] for x in mapping.values()})==8
matrix={}
for line in topo['commands'][0]['text'].splitlines():
    a=line.split()
    if a and re.fullmatch(r'Phy-ID\d+',a[0]) and len(a)==17 and a[1] in ['X','SIO','HCCS_SW']:
        matrix[int(a[0][6:])]=a[1:]
assert set(matrix)==set(range(16))
for i in range(16):
    for j in range(16):
        assert matrix[i][j]==('X' if i==j else 'SIO' if mapping[i]['NPU_card_id']==mapping[j]['NPU_card_id'] else 'HCCS_SW')
info_by_pid={int(r['data']['pid']):r for r in meta['captured_device_info']}
by_user={r['user_device_id']:r for r in acl['rows']}
joined=[];tp_groups={};ep_groups={}
for r in capture:
    assert r['native_device_ids']==[r['ep_rank']]
    user=r['native_device_ids'][0];q=by_user[user];a=q['attributes'];m=q['mapping'];pid=r['pid']
    assert all(v['rc']==0 for v in m.values())
    assert m['user_to_logic']['value']==m['logic_to_user']['value']==user
    logic=m['user_to_logic']['value'];phy=m['logic_to_physical']['value']
    assert m['direct_user_to_physical']['value']==phy==mapping[logic]['driver_physical_device_id']
    assert m['aclrtGetLogicDevIdByPhyDevId']['value']==logic and m['aclrtGetUserDevIdByPhyDevId']['value']==user
    assert a['NPU_ARCH']['value']==2201 and a['CUBE_CORE_NUM']['value']==a['AICORE_CORE_NUM']['value']==24 and a['VECTOR_CORE_NUM']['value']==48 and a['IS_VIRTUAL']['value']==0
    ci=info_by_pid[pid]['data'];di=ci['DeviceInfo'][0]
    assert di['id']==user and di['ai_core_num']==24 and di['aiv_num']==48 and ci['cannVersion']=='9.2.0'
    assert ci['hostname']==bridge['current_container']['hostname']
    for name,g in r['parallel_group_info'].items():
        if g['group_name']=='tp':
            assert g['group_rank']==r['tp_rank'] and g['global_ranks'][g['group_rank']]==r['ep_rank'];tp_groups.setdefault(tuple(g['global_ranks']),[]).append(user)
        if g['group_name'] in ['ep','mc2']:
            assert g['group_rank']==r['ep_rank'];ep_groups.setdefault((g['group_name'],tuple(g['global_ranks'])),[]).append(user)
    joined.append({'pid':pid,'process_uuid':r['process_uuid'],'EP_global_rank':r['ep_rank'],'DP_rank':r['dp_rank'],'TP_rank':r['tp_rank'],'native_device_id':user,'user_device_id':user,'logic_device_id':logic,**mapping[logic],'ACL_reported_PHY_CHIP_ID':a['PHY_CHIP_ID']['value'],'SMP_ID':a['SMP_ID']['value'],'MAINBOARD_ID':a['MAINBOARD_ID']['value'],'arch':2201,'CUBE_cores':24,'Vector_cores':48,'is_virtual':False,'L2_reported_bytes':a['L2_CACHE_SIZE']['value'],'global_memory_reported_bytes':a['TOTAL_GLOBAL_MEM_SIZE']['value'],'UB_attribute_raw_value':a['UBUF_PER_VECTOR_CORE']['value'],'profile_capture_frequency_field':di['aic_frequency'],'captured_metadata_sha256':r['metadata_sha256'],'captured_device_info_sha256':info_by_pid[pid]['sha256']})
assert sorted(tp_groups)==[tuple(range(8)),tuple(range(8,16))]
assert all(sorted(ranks)==list(range(16)) for ranks in ep_groups.values())
placements=[]
for group,members in sorted(tp_groups.items()):
    assert sorted(members)==list(group)
    cards=sorted({mapping[x]['NPU_card_id'] for x in members})
    edge=collections.Counter(matrix[i][j] for i in members for j in members if i!=j)
    assert len(cards)==4 and edge=={'SIO':8,'HCCS_SW':48}
    placements.append({'global_ranks':list(group),'NPU_cards':cards,'compute_devices':8,'directed_nonself_endpoint_pairs':56,'directed_SIO_pairs':8,'directed_HCCS_SW_pairs':48})
write('current-171-physical-pool-bridge.json',{
 'status':'CURRENT_RUNTIME_DEVICE_MAPPINGS_AND_CAPTURE_GROUP_PLACEMENT_RECONCILED',
 'rows':sorted(joined,key=lambda r:r['EP_global_rank']),
 'observed_compute_devices':16,'observed_NPU_cards':8,'reported_CUBE_core_slots_per_compute_device':24,'reported_Vector_core_slots_per_compute_device':48,'reported_nonvirtual_devices':16,
 'TP_placements':placements,'EP_MC2_memberships':[{'group':k[0],'global_ranks':list(k[1]),'observed_devices':sorted(v)} for k,v in ep_groups.items()],
 'topology_pair_counts':{'SIO_directed':16,'HCCS_SW_directed':224},
 'namespace_findings':['ExplicitACL mapping APIs and reverse/direct round trips establish user/logic/driverphysical IDs,not equal-number guessing.','NPU-smi maps two compute devices to eachNPUcard. ACL_PHY_CHIP_ID has8values andSMP_ID has8groups,while driverphysicaldeviceID has16values. Keep these namespaces separate;the wordchip is not a license to doublecount.','MAINBOARD_ID is20 forall16devices;it is not theNPUcard index.','All16UB attribute204 queries returnrc0,value0. That value is unusable as an actualUBcapacity bound;do not interpret it as physicalzeroUB or claim a fusion plan impossible from it.','L2 is reported192MiB percompute-device query. Do not merge independent/private caches into onefreelyaddressablepool or assume exclusivecapacity. Globalmemory isAPI-exposedtotal,notfree memoryavailable tothisoptimization.'],
 'AICPU_pool_limit':'Eachdevicequeryreports6AICPUcores,butpaireddeviceshavethesameSMP_ID. Do notmultiply6by16asprovedindependentAICPUcapacity. ActualsharedOS/coreownershipandAICPUinstructionratesremainopenforM21.',
 'scope':['Current171 management/runtime queries plus same-containerCANN9.2 captured device andgroupmetadata.','Applying the currentdriverphysicalmapping to the captured171interval assumes no interveningdriverremap;containeridentity/visibility andcapturedcorecounts areconsistent. No historical166placement isbackfilled.','Distinctnonvirtualexecutiondevices arethe currentper-devicecapacityunits. Numericwallclockcapacity stillrequires arithmeticmode/clockbound andresource-sharingcontracts.','Topology labels documentSIO/HCCS_SW paths,not linkbandwidth,independence orcollectivewireprotocol.'],
 'new_historical_native_or_phase_promotions':0,'whole_modules_closed':0,'overall_complete':False})

url='https://www.hiascend.com/doc_center/source/zh/HDK/2610/A3/A3DCMI/dcmia3_049.html'
assert dcmi['dcmi_init_rc']==0 and len(dcmi['rows'])==16
assert all(r['frequencies']['AICORE_MAX']=={'enum':9,'rc':0,'value':1800} and r['frequencies']['AICORE_CURRENT']=={'enum':7,'rc':0,'value':800} for r in dcmi['rows'])
write('clock-field-primary-semantics-and-bound-status.json',{
 'status':'DOCUMENTED_RATED_FREQUENCY_BOUNDARY_ACTUAL_UPPER_STILL_UNKNOWN',
 'official_source':{'url':url,'title':'A3 DCMI:dcmi_get_device_frequency','retrieved_date':'2026-11-03','section':'Parameter input_type,frequency andplatformsupport','specific_supported_claim':'Enum9 is ratedAIcorefrequency sustained underTDP conditions;outputunitMHz. This doesnotdefine anabsolute fmax. A3supports2,6,7,9;Vectorcurrent12 isnotlisted assupported.'},
 'installed_header':dcmi['header'],'observed_rows':dcmi['rows'],'actual_queries':{'rated_enum9_MHz':1800,'idle_current_enum7_MHz':800,'HBM_enum6_MHz':1600,'Vector_current_enum12_return_code':-8255},
 'symbol_name_semantic_mismatch':'InstalledDCMI_FREQ_AICORE_MAX=9 hasMAXinitsname;theofficialA3fieldmeaning isratedfrequency. Do notpromoteit toahardwareupperbound.',
 'profile_frequency_field':'All16captured info.json fieldsreport1800;theycanbeusedasareferencefield,notsufficientproof ofinstantaneousorhighestclock.',
 'physical_peak_frequency_upper_Hz':None,'reference_frequency_Hz':1800000000,'new_frequency_setter_calls':0,
 'decision':'Stoprepeatingthisqueryfor anabsoluteupperbound. Usefrequency-freecyclecuts andexplicitrated-clock referencecases;anactualhardwareclockcapremainsaspecificmissingphysicalpremise,notaprerequisiteforeachoperatorsemanticconclusion.',
 'overall_complete':False})

old=json.loads((O/'typed-capacity-primary-contract-v89.json').read_text())
matrices=[]
for name,r in old['cube_A3_2201'].items():
    per_core=r['MAC_per_AIC_cycle'];per_device=24*per_core
    matrices.append({'mode':name,'documented_MAC_per_AIC_cycle':per_core,'current_reported_CUBE_cores':24,'MAC_per_full_device_reference_cycle':per_device,'MAC_per_second_at_rated1800MHz':per_device*1800000000,'status':'DOCUMENTED_MODE_RATE_WITH_MEASURED_ARCH_CORE_COUNT_AND_REFERENCE_CLOCK_NOT_PHYSICAL_TIME_UPPER'})
prev=json.loads((H.parent/'shared-profile-analysis-v170/mc2-aggregate-work-review.json').read_text())['review']
per_assignment=35389440;per_cycle=24*old['cube_A3_2201']['INT8']['MAC_per_AIC_cycle'];assert per_assignment//per_cycle==180 and per_assignment%per_cycle==0
cases=[]
for role,a in prev['aggregate'].items():
    cycles=Fraction(a['direct_MAC'],per_cycle);seconds=cycles/Fraction(1800000000)
    cases.append({'role':role,'source_domain_direct_MAC':a['direct_MAC'],'native_calls':a['native_records'],'cumulative_native_seconds':a['native_work_ns']/1e9,'necessary_sum_full_device_equivalent_INT8_Cube_cycles':float(cycles),'rated1800MHz_reference_compute_seconds':float(seconds),'ratio_reference_compute_floor_over_cumulative_kernel_time':float(seconds/Fraction(a['native_work_ns'],1000000000))})
write('current-arch-cycle-and-reference-capacity-cuts.json',{
 'ceiling_state':'CONDITIONAL_CEILING','status':'CYCLE_PROOF_REVIEWED_WITH_CURRENT_ARCH_AND_CORE_POOL',
 'actual_arch':2201,'reported_core_counts':{'CUBE':24,'Vector':48},'instruction_source_reused':str(O/'typed-capacity-primary-contract-v89.json'),'mode_cases':matrices,
 'F21_domain':'Same pinned directdenseINT8Cube algorithm,actualreceiverrows,H5120,D2304,currentoutputstreamcountercontract. NoINT4decomposition,alternatealgebra,compressionorcrosscallworkreuseinthedeclaredworkcut.',
 'proof':'ForQrequiredINT8MAC,the8192MAC/core/cycleupper gives requiredsumofcorecycles>=Q/8192. Dividingby24givesfull-deviceequivalentcycles>=Q/(24*8192). Oneassignmenthas35389440MACandthereforeneedsatleast180full-deviceequivalentcycles. No measuredmaximumthroughputorfastestscheduleisinverted.',
 'time_bound_with_a_real_frequency_upper':'Ifallrelevantcoreclocksare<=F_Hz,thenT>=Q/(24*8192*F_Hz). Fhasnotbeenprovedbythecurrentquery. UnderDVFSusea validintegralcapacitybound;theidle800or rated1800valueisnotanabsoluteclockcap.',
 'cases':cases,
 'comparison_scope':'Sumofper-device/per-callfullkernelintervals,notrequestwalltime. Whole-serverelapsedconstraintsneedthe16devicecapacitysumandacompleteworkwindow,notthiscumulativetimesum.',
 'mixed_resource_rule':'Usemax ofindependentnecessaryresourcecutsunlessanexplicitcommonresourceandnonduplicateworkproofpermitssumming. Separatemode-rate rowsdo notproveaweightedjointCubeissue-simplex. MIX/COMMlabelsdo notcreateindependentfreecapacitypools.',
 'all_new_fusion_domains':'RecomputeworkandIOafterfusion/reorganization. Currentdirectdenseworkisnotanunconditionallowerboundforallalgorithms.',
 'physical_MFU_or_speedup':None,'whole_modules_closed':0,'overall_complete':False})

# Board ingress cut for a full, uncompressed all-gather. The proof permits
# forwarding/multicast; it does not equate rank-pair deliveries with wire bytes.
checks=[]
R=2*5120
for n in range(65):
    c=(n+7)//8;valid=[max(0,min(c,n-r*c)) for r in range(8)]
    local=[sum(valid[2*b:2*b+2]) for b in range(4)]
    remote=[n-x for x in local]
    assert sum(valid)==n and sum(remote)==3*n
    assert max(remote,default=0)<=6*c
    if n>=6*c:assert max(remote,default=0)==6*c
    checks.append({'n':n,'chunk':c,'local_valid_rows_per_card':local,'required_remote_rows_per_card':remote,'aggregate_card_ingress_rows':sum(remote),'max_card_ingress_rows':max(remote,default=0)})
example=checks[20];assert example['local_valid_rows_per_card']==[6,6,6,2] and example['required_remote_rows_per_card']==[14,14,14,18]
write('TP8-board-ingress-lower-bound-and-padding-domain.json',{
 'ceiling_state':'CONDITIONAL_CEILING','status':'ANALYTIC_BOARD_INGRESS_PAYLOAD_CUT_REVIEWED',
 'placement_evidence':str(H/'current-171-physical-pool-bridge.json'),
 'actual_groups':placements,
 'domain':['EachTP8group occupiesfourNPUcardswithtwocomputedevicespercard.','Everyrankrequiresallrawshardsaspublicfullallgatheroutputs;arbitraryrequiredinputbytesaresingleownedinitially,withnoremotepriorcopiesorvaluecompression/dedup.','Onboardforwarding/multicastandpacketaggregationareallowed. Knownpaddingcanbereconstructedonlywhenitsconstantvalueatthecollectiveboundaryandallconsumercontractsareproved.','Ifconsumerfusionremovesrawsharddemand,ormemo/compression/ownershipchanges,deriveanewcut;thisisnotanunconditionalfloorforallmodelalgorithms.'],
 'proof':{'general':'LetGbethetotalrequiredindependentrawpayloadandC_bthepayloadinitiallyoncardb. Cardb mustreceiveatleastG-C_bbytesacrossitsboundary. Thusaggregatecardingress>=sum_b(G-C_b)=(k-1)Gforkcards,withnoinitialduplication. Abyteenteringonecardcountsjustonceatthatcardboundary,evenifonboardforwardedtoitsotherrank.','equal_full_shards':'With8rankshardsSbytesand4cards,eachcardinitiallyowns2Sof8S. Ingress>=6Spercardand>=24Sinaggregate. The48crosscardrank-pairdeliveries*Sisadirectdeliveryreference,notanallalgorithmswirefloor.','time':'Foranintervalcoveringallrequiredcrossingsandcard-ingresscapacityupperB_b,T>=max_b((G-C_b)/B_b). Anadditionalaggregateboundusestotalingresscapacityupper. Requiredresourcecutsmayoverlapcomputation;donotforcewholeproducercompletionbeforefirstcommunicationtile.','not_a_schedule_inversion':'Theproofisacutnecessityargument,notaninvertedfeasiblemessageexecutiontime.'},
 'per_rank_S_bytes_example_full_M3_BF16_H5120':30720,
 'full_M3_example':{'required_ingress_bytes_per_card':184320,'required_aggregate_ingress_bytes':737280,'direct_rank_delivery_crosscard_reference_bytes':1474560},
 'known_padding_n20_p8_example':{'status':'CONDITIONAL_FUNCTION_GEOMETRY_EXAMPLE_NOT_A_NATIVE_ALLGATHER_OPERAND_PROMOTION','source_function_evidence':str(H.parent/'shared-profile-analysis-v169/SP-ma<REDACTED_TOKEN>.json'),'additional_guard':'TheAllGatherinputmustbethecorresponding20->24paddedtensorandpaddingmuststillbeknownconstantatthisboundary;maskshapealonecannotproveit.','row_bytes_for_BF16_H5120':R,'valid_rows_per_rank':[3,3,3,3,3,3,2,0],**example,'required_ingress_bytes_per_card':[x*R for x in example['required_remote_rows_per_card']],'required_aggregate_ingress_bytes':3*20*R,'required_max_card_ingress_bytes':18*R,'key_result':'Removingknownpaddingreducesaggregateingressfloorfrom720KiBto600KiBbutleavesthemax-cardfloorat180KiB. Withidenticalcardingresscapacities,thisparticularbottleneckcutisunchanged. Thatdoesnotprovenoactualbenefit;itpreventscountingbytefractionaslatencyspeedup.'},
 'candidate_reuse':['F26','F32','F33','F31'],
 'composition':'Forproject-before-gatherorHCcollapsevariants,substitutethenewnecessarypayloadonlyafterexistingnumerical/ownership/allconsumer/readinessconditions;pricearithmetic,localreconstruction,lifetimeandnewcommunicationinonegraph. Existingbaselineprecollapseisnotcountednew. TPandEPtrafficonthe samecardingressshareabudgetwhenoverlapoccurs.',
 'independent_finite_geometry_checks':len(checks),'geometry_checks':checks,
 'unknown_bandwidths':True,'actual_wire_bytes_or_speedup':None,'new_native_operand_or_phase_promotions':0,'whole_modules_closed':0,'overall_complete':False})
write('physical-pool-analysis-question-review.json',{
 'previous_goal_turn':'PROGRESS','status':'CURRENT_POOL_AND_TWO_THEORY_DOMAINS_REFINED',
 'conclusions':['Current16runtimeuser/logic/driverphysicaldevice mappings roundtrip,join8NPUcardsx2computedevices;all16CANNprofilesandrecordedTP/EP/MC2membershipsreconciled.','Arch2201/24Cube/48Vectorarenowqueriedcurrentfacts;8pairedPHY_CHIP_IDgroupsarenot16driverphysicaldeviceIDs.','Enum9returns1800MHzbutofficialA3semanticsareratedfrequency;currentidle800MHzandVectorqueryunsupported. Noabsolutefmaxclaimed.','INT8MACrequiredcyclecutsareinstantiatedwithoutfrequency;1800MHzvaluesarereferenceonly.','ActualTP8fourcardplacementenables6Sper-card/24Stotalallgatheringresscut. Known20->24paddingmayreduceaggregatepayloadwhileleavingmaxcardcutunchanged.'],
 'source_reread_scope':'InstalledACL/DCMIinterfaceheadersandcapturemetadataonly;noinferenceframeworksourceorfullCasewasreread/replayed.',
 'new_historical_native_or_phase_promotions':0,'whole_modules_closed':0,'overall_complete':False})
print(json.dumps({'status':'PROGRESS','device_card_rows':len(joined),'TP_groups':placements,'clock_is_rated_not_absolute_upper':True,'F21_cycle_reference_cases':cases,'allgather_padding_example':{k:v for k,v in example.items() if k!='n'},'n_geometry_checks':len(checks)},ensure_ascii=False))
