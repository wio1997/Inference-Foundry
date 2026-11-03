from copy import deepcopy
from datetime import datetime,timezone,timedelta
from collections import Counter
import json,hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl

def main():
 assert not(O/'REVIEW-CHECKPOINT-v25.json').exists()
 now=datetime.now(timezone.utc);stamp=now.isoformat()
 labels=['C2-cache-spec166-v25','C2-allocation166-v25','C2-compiler166-v25','V41-cache-dispatch166-v25','current-cache-log166-v25','V41-hook-install166-v25']
 hashes=['30569f0fa87afbcd66e317fa5fa1b5af13395b222b222df9bc2ce9fd00e6f159','d23ab42da5b10bbd017a629e89b338148d8a2221fd892ebe164b8b7e1c92798f','9cb5ed173954d56e5aa9e238a68e46e9e1a0b5c9c2b0ceba7709aeb8af0f9c00','1d1d173101887b8aaa022fbebe1442c150f017dc6e1a2ce73277aca237b76c52','396eb33c775a939eb1a7f1941ffe024f824ac4a1062acba91bef003cd67141e4','bfbaa99e3c39270770c8e973f61c8f1c01fde73715dd61fd5197ce8e504de75c']
 data={};refs=[]
 for label,sha in zip(labels,hashes):
  r=read(N/(label+'.json'));assert r['returncode']==0 and not r['stderr'];z=json.loads(r['stdout']);assert z['result_sha256']==sha;data[label]=z;refs.append(dict(receipt=str(N/(label+'.json')),path=z['full_result'],sha256=sha))
 review=read(O/'V41-tuple-capacity-and-F34-graph-review-v25-v2.json');assert review['finite_capacity_cases']==72
 assert len(review['old_graph_undefined_edges'])==2
 failed=read(N/'zcode-V41-capacity-v25-output-20261031.json');assert 'rc=1'in failed['response']and'FileNotFoundError'in failed['response']
 logs=data[labels[4]]['result'];params=data[labels[5]]['result']['saved_startup_whitelisted_fields'];assert params['max_num_seqs']==['4']
 assert any('LBNHC'in r['text']for r in logs['technical_cache_lines'])and any('8.72 GiB'in r['text']for r in logs['technical_cache_lines'])
 install=data[labels[5]]['result']['installed_source']['hook_assignment_or_call_contexts'];assert len(install)==3 and all(not r['enclosing_functions']for r in install)
 compiler=data[labels[2]];cc=compiler['C2_compiler_candidates'];assert len(cc)==28
 compiler_counts=Counter(c['metadata']['name']for c in cc);assert compiler_counts==Counter(_pool_kernel=14,_cache_update_kernel=14)
 facts=[
  dict(id='C25-1',grade='DOCUMENTED',finding='The current compressor state object is a DeepseekV41CacheLayer registered in static_forward_context. Its methods bind a supplied tensor,return its spec/backend,and do not perform ring-value arithmetic. Bounded direct state-object identifiers lead to compressor.pool_projected as the explicit value reader in the inspected model/DSA subtree;generic/external aliases are not ruled out.',evidence=refs[0]),
  dict(id='C25-2',grade='DOCUMENTED',finding='CircularBufferSpec reserves at most one block per request,hasprefix_cacheable=False anduses_slot_mapping=False. AscendCircularBufferSpec retains this lifecycle policy andchanges uniform-type grouping. These contracts do not prove an independent small allocation or a live-request limit onallphysicalpages.',evidence=refs[0]),
  dict(id='C25-3',grade='DOCUMENTED',finding='V41 has a model-specific cache planner. get_layer_tuples combines longKV+index(+optional folded)payload withstate/SWA aliases;tuple capacity ismax(persistentpayload,eachalias unpaddedpage). group_cache_specs padsaliases tothatcapacity;get_deepseek_v41_kv_cache_config returns one independently sizeddescriptor pertuple andoneglobalblock-ID count fromavailable_memory/sum(tuplecapacities).',evidence='V41-tuple-capacity-and-F34-graph-review-v25-v2.json'),
  dict(id='C25-4',grade='DOCUMENTED',finding='The worker V41 branch allocates one backing perdescriptor andmaps all its layer names tothatbacking. Reshape usesdescriptor block_stride,zero initial offset foraliases,andlongKV offset forindex planes. Thegenericupstream max-group/shared-allocation formula isnot substituted forthisV41sum-oftuple model.',evidence=[refs[1],'V41-tuple-capacity-and-F34-graph-review-v25-v2.json']),
  dict(id='C25-5',grade='DOCUMENTED',finding='Exact Ascend hook functions select V41 grouping,pool-byte divisor andconfig builders. Module-level assignments installthepacked-group,config-from-groups andpool-byte hooks. This binds thenormal source-dispatch contract,not a historical livePythonfunction-object snapshot;KVPP andotherallocatorpreemption branches remain separatelyconditional.',evidence=[refs[3],refs[5]]),
  dict(id='C25-6',grade='MEASURED',finding='The exact captured service.log states LBNHC layout,8.72GiB availableKVmemory,andper-engine max_num_seqs4. ItsKV sizing line reports353802tokens andestimated10.16concurrencyfor34816tokens/request. These are saved initialization/configuration estimates,not newhealth/idleobservations or measured throughput/hardwareconcurrencyceilings.',evidence=[refs[4],refs[5]]),
  dict(id='C25-7',grade='DERIVED',finding='Let an unchanged SWA alias contributeS>=R_old>=R_new>=0 andlong/index/foldedpayloadL>=0. Thenmax(L,S,R_old)=max(L,S,R_new). In the profile-consistent pageexample,SWA128x512BF16=131072bytes equalsring32x1024FP32;ring16slots=65536. Thusshrinkingringalone leaves tuplecapacities unchanged inthisSWA-anchor domain;withsamegroups,budget andoverride itleavesnum_blocks/descriptorsizes unchanged.',evidence='V41-tuple-capacity-and-F34-graph-review-v25-v2.json'),
  dict(id='C25-8',grade='DERIVED',finding='F34-B isexcluded asastandalonepool-capacity optimization intheunchangedV41tuple/SWA-anchor domain. Itslogical2.268GB->1.134GB array model isnotextra independent ringallocationorsavings. Changingtuplemembership,SWArepresentation orglobalpoolpolicy wouldbeaseparatewhole-cache domainrequiringallconsumers/lifetimes.',evidence='F34-cache-allocation-variants-v25.json'),
  dict(id='C25-9',grade='DERIVED',finding='F34-A odd-position maintenance pruning remainsconditional. Thecorrectstate namespace includesdescriptor/tuple identity,globalblockID,slot andgeneration. Rawringviews canaliasSWA/long/indexbacking,so safety requires validblockownership andlifecyclehandoff,not merelydifferentTensorobjects. Theold-read-before-write WAR andall retained-even values/generations persist.',evidence='F34-cache-allocation-variants-v25.json'),
  dict(id='C25-10',grade='MEASURED',finding='Savedruntimecontains73compiled-entrymetadatafiles with73same-stem binaries;28namesmatchC2:14pool and14update,eachwith14distinctbinaryhashes/TTIRhashes. Their.sourcefiles areIR,notPython source snapshots. ComputeInfo names alone do not select an executedbinary,andthese runtimecopies maycontain earlier artifacts. No loaded-native identity isclaimed.',evidence=refs[2]),
  dict(id='C25-11',grade='DERIVED',finding='F34v24graph used2undefinedshorthand endpoints. v25 replaces them withconsistent operation/external-reader nodes and13defined edges,including original update->norm andindex-publish->long-publish schedule controls forF34alone. F18/F19ordering changes remainseparateconditionalcompositions. This isartifactdependency validation,not nativecorrectness.',evidence='V41-tuple-capacity-and-F34-graph-review-v25-v2.json'),
  dict(id='C25-12',grade='UNKNOWN',finding='Actualpointer/groupdescriptors andblock-owner generations were notsnapshotted. External/full-byte ABI consumers,KVPP/transfer branches notboundbyselectedparams andexecution->binary identity remainopen. Sourcecontract capacityexclusion isnot wholemoduleclosure or ageneralno-fusion theorem.',evidence='C2-cache-allocation-condition-ledger-v25.json')]
 write(O/'C2-cache-object-and-allocation-contract-v25.json',dict(generated_utc=stamp,status='V41_TUPLE_ALLOCATION_CONTRACT_AND_RESTRICTED_CAPACITY_EXCLUSION_REVIEWED',findings=facts,overall_complete=False,whole_module_closed=False,measured_gain=None))
 f34=deepcopy(review['corrected_candidate']);f34['profile_review_state']='SOURCE_OBJECT_AND_V41_TUPLE_POLICY_REVIEWED;ODD_WRITE_PRUNING_CONDITIONAL;STANDALONE_CAPACITY_GAIN_EXCLUDED_IN_ANCHOR_DOMAIN'
 f34['allocation_v25']['current_tuple_contract']='Views in oneV41descriptor sharebacking. EqualglobalblockIDs canaddress overlappingbytes acrosscacheroles;validconcurrentownershipofdistinctIDs keepspageregionsseparate.'
 for v in f34['variants']:
  if v['id']=='F34-B':v.update(status='EXCLUDED_AS_STANDALONE_POOL_CAPACITY_OPTIMIZATION_IN_CURRENT_V41_SWA_ANCHOR_DOMAIN',requires='A broaderrepresentation/tupleallocationchange needsitsown allconsumer/ABI proof;simply16logicalslotsdoesnotshrinkexistingbacking.')
 f34['state_namespace_v25']=['descriptor_or_tuple','global_block_ID','token_slot','ownership_generation']
 f34['allocation_question_closed_scope_v25']='SameV41tupleplanner,SWAanchor>=oldringpage,unchangedotherresources/groups/budget/override:halvingringalonecannotreducepooldescriptorallocation.'
 f34['remaining_native_obligations_v25']=['Actualblockownership/aliasgeneration andalllegitimatestateobservers','PreservedFP32evenwritevalues/order andWAR','Nativeexecution/cachebinarybinding','Changed-layout ABI/contiguity/graphaddressesifbroaderrewritechosen']
 f34['evidence_files']+=['C2-cache-object-and-allocation-contract-v25.json','V41-tuple-capacity-and-F34-graph-review-v25-v2.json','C2-cache-allocation-condition-ledger-v25.json']
 nodes={n['id']for n in f34['candidate_graph']['nodes']};assert len(nodes)==10 and len(f34['candidate_graph']['retained_edges'])==13
 assert all(e['a']in nodes and e['b']in nodes for e in f34['candidate_graph']['retained_edges'])
 adj={n:[]for n in nodes}
 for e in f34['candidate_graph']['retained_edges']:adj[e['a']].append(e['b'])
 state={}
 def visit(n):
  assert state.get(n)!=1,'cycle'
  if state.get(n)==2:return
  state[n]=1
  for k in adj[n]:visit(k)
  state[n]=2
 for n in nodes:visit(n)
 write(O/'F34-cache-allocation-variants-v25.json',f34);(O/'F34-current-before-v25.json').write_bytes((R/'candidates/F34.json').read_bytes());write(R/'candidates/F34.json',f34)
 sourcefiles={}
 for label in [labels[0],labels[1],labels[3]]:
  for f in data[label]['result']['files']:sourcefiles[f['path']]=f['sha256']
 compiler_summary=[]
 for name in sorted(compiler_counts):
  rows=[c for c in cc if c['metadata']['name']==name]
  compiler_summary.append(dict(name=name,entries=len(rows),distinct_binary_hashes=len({a['sha256']for c in rows for a in c['artifacts']if a['path'].endswith('.npubin')}),distinct_TTIR_hashes=len({a['sha256']for c in rows for a in c['artifacts']if a['path'].endswith('.ttir')}),architectures=sorted({c['metadata']['target']['arch']for c in rows})))
 write(O/'C2-compiler-lineage-contract-v25.json',dict(generated_utc=stamp,saved_compiled_metadata_entries=73,C2_candidate_entries=28,candidate_sets=compiler_summary,execution_to_binary_hash_link=False,source_extension_content='MLIR/TTIR-likeIR;notPythonfileidentity',historical_home_log_subset='Thefirstbounded12home-logcandidatesarefromOct27;theywerenotusedascurrentcaptureconfiguration. Currentservice.logwasreadseparately.',source_files_with_exact_SHA=[dict(path=p,sha256=s)for p,s in sourcefiles.items()],overall_complete=False))
 branches=[
 ('STATE-OBJECT','Cache-only registeredDeepseekV41CacheLayer','Bind/returnspec/backend,noringarithmetic;onlyexplicitmodelreaderininspectedsubtreeispool_projected.','Genericbyteobservers/aliases remainseparate.'),
 ('STATE-LIFETIME','CircularBufferSpec','Oneblock/request;notprefixcacheable;noordinaryslotmapping.','Doesnotimplyindependentallocationorallresidentrequestbounds.'),
 ('HARDWARE-ABI','uses_a5_packed_cache capability','Predicateisfrozenhardwarecapabilityquery;Ascendvariantseparate.','No historicalbooleanobjectsnapshotclaimedfromsource.'),
 ('ALLOCATOR-PREEMPT','KVPPsize>1 /V41normalbranch','WorkerKVPPbranchcanreturnbeforeV41;normalV41descriptorsseparatebackings.','Do notapplynormalcontracttootherPD/KVPPconfigurationswithoutbinding.'),
 ('V41-DISPATCH','is_deepseek_v41_cache +modulehooks','ExactV41helperbranchesand3moduleassignmentsbound.','Normal sourcecontract,notlivefunction-objectinspection.'),
 ('V41-TUPLE','long/index/folded withstate/SWAaliases','Capacitymax(payload,aliases),onebacking/tuple.','Logicalviewbytesarenotadditivephysicalallocations.'),
 ('SWA-ANCHOR','SWApage>=oldringpage','131072Bprofile-consistentanchorleavesmaxcapacityunchangedwhenringhalves.','Standalonecapacitygainexcludedonlyinthiscontract;otherwholereorganizationsnotexcluded.'),
 ('UNPADDED-PADDED','logicalstatebytes /tupleblockstride','Aliasespaddedtosamecapacityandviewsuseblockstride.','Currenthelperrequires32rowstateandcontiguity;16slotnotdrop-in.'),
 ('BLOCK-OWNERSHIP','samebacking,differentsemanticgroups','Use(tuple,blockID,slot,generation)namespace.','Retainexclusivevalidbyteownership andinitializationbeforehandoff.'),
 ('F34-A-ORDER','oddwritefilteralone /F18/F19composition','F34aloneretainsoriginalupdate->norm andindexpublish->longpublishcontrol.','Relaxesnootherorderingbydefault.'),
 ('SOURCE-PROFILE','currentinitializationlog /oldhomelogs','CurrentlogLBNHC/8.72GiB/maxseq4;otherselectedlogshistorical.','Sizingestimateisnotnewhealth,throughputorhardwareceiling.'),
 ('COMPILED-IDENTITY','14pool/14updatecachedbinaries /actualexecution','All28hashesretained,notsingle-nameidentity.','No executions assignedtobinarybyname/mtime/nearesttime.'),
 ('FULL-BYTE-ABI','evenvalueobservation /fullstateobserver','F34-Arequiresobservationaldomain;oddbytesmaydiffer.','Prefixcachefalsealone doesnotruleouttransfer/checkpoint/debugobservers.'),
 ('ARTIFACT-GRAPH','oldshorthand /typedsourcegraph','2undefinedendpointsfixed;10nodes13edgespassstructuralDAGcheck.','Not allconsumer/native semanticclosure.'),
 ('CAPACITY-ACCOUNTING','fixedbudget/divisor/override','Unchangedtuplecapacities=>unchangedplannerN/descriptorsizes.','No1GiBreleasablememoryclaimfromhalf logicalringarray.')]
 write(O/'C2-cache-allocation-condition-ledger-v25.json',dict(generated_utc=stamp,branches=[dict(id=a,condition=b,reviewed=c,remaining=d)for a,b,c,d in branches],whole_module_closed=False,overall_complete=False))
 model=deepcopy(read(O/'candidate-profile-ledger-v24.json'));assert len(model['candidates'])==34
 for i,c in enumerate(model['candidates']):
  if c['id']=='F34':model['candidates'][i]=f34
  elif c['id']in ['F17','F18','F19','F31']:c['state_allocation_v25']='C2-cache-object-and-allocation-contract-v25.json;V41tuplealiases/sharedglobalblockIDs;noadditivelogicalringallocationorstandalonehalf-capacityclaim.'
 model.update(generated_utc=stamp,parent_ledger='candidate-profile-ledger-v24.json');write(O/'candidate-profile-ledger-v25.json',model)
 framework=deepcopy(read(O/'framework-candidate-ledger-v24.json'));framework.update(generated_utc=stamp,parent_ledger='framework-candidate-ledger-v24.json');write(O/'framework-candidate-ledger-v25.json',framework)
 mods=deepcopy(read(O/'module-profile-ledger-v24.json'));matrix=deepcopy(read(O/'module-phase-branch-matrix-v24.json'))
 notes={'M02':'V41specific4-tupleplanner,descriptorbackingaliases,CircularBufferoneblock/request/prefixfalse andSWA-anchorcapacityexclusionreviewed;notgenericmax-groupformula.','M08':'Directringobjectreaderlocalized;stateviewaliasesothercachetypesbypageownership.F34-Aconditional,F34-Bstandalonecapacitygainexcludedinunchangedanchorcontract.','M22':'F34graphfixed2undefinedendpointsandretainscurrentordering;28cachedC2binariesarenotexecutionidentity.Captureaddresses/aliasgenerationsremain.','M23':'V41pooldivisorhookusesSUMtuplepagesforrankreplan;donotuseupstreamMAXgroupsortreatperrankbudget/numblocksasindependentwithoutprotocolproof.','M24':'Oneblock/request/uncacheableprefixstateisnotall-byteABIclosure;ownershipnamespaceincludestuple/globalblockID/slot/generation andhandoffinitialization.'}
 for m in mods['modules']:
  m['phase_branch_record']='module-phase-branch-matrix-v25.json#'+m['id']
  if m['id']in notes:m['review_v25']=dict(finding=notes[m['id']],evidence=['C2-cache-object-and-allocation-contract-v25.json','F34-cache-allocation-variants-v25.json','C2-cache-allocation-condition-ledger-v25.json'],whole_module_closed=False)
 for m in matrix['modules']:
  if m['id']in notes:m['branches'].append(dict(branch='F34stateobject/V41tupleallocation/currentcaptureconfiguration andartifactbinding',state='SCOPED_SOURCE_CONTRACT_CAPACITY_EXCLUSION_REVIEWED;WHOLE_MODULE_OPEN',evidence=notes[m['id']],gap='Actualownership/full-byteobservers/nativebinding/otherfullmoduleconditions remainexplicit.'))
 mods.update(generated_utc=stamp,parent_ledger='module-profile-ledger-v24.json');matrix.update(generated_utc=stamp,parent_matrix='module-phase-branch-matrix-v24.json');write(O/'module-profile-ledger-v25.json',mods);write(O/'module-phase-branch-matrix-v25.json',matrix)
 remaining=deepcopy(read(O/'module-coverage-gaps-v24.json')['remaining']);remaining[1]='F34sourceobject/spec/readersubtreeandV41tupleallocationarenowreviewed. Stoprepeatingstandalone16slotmemorypromise:underunchangedSWAanchorplannerallocationcannotshrink.F34-Aneedsactualblockownership/stateABI/nativebindingifpursued;differentwholecachepool/layoutdomainmustbepricedseparately.'
 remaining.insert(3,'Currentcapturestartupisnowboundtosavedservice.logLBNHC/8.72GiB/perenginemaxseq4;theseareconfiguration/sizingfactsnotCPUcost/newhealth.28C2compilecandidates14+14stilllackexecutionhashlink;donotexhaustallcompilerinternalsmerelytorepeatconditionalcandidate.')
 write(O/'module-coverage-gaps-v25.json',dict(generated_utc=stamp,remaining=remaining,whole_modules_closed=0,overall_complete=False))
 summary=dict(generated_utc=stamp,status='PROGRESS',previous_turn_classification='PROGRESS:v24newF34/stateintegerproof/current240C2boundaries',model_candidates=34,framework_candidates=7,local_capacity_question_closed='No standalone ring-half pool-size reduction underunchangedV41tuple/SWA-anchor contract.',current_saved_cache_layout='LBNHC',current_saved_available_KV_memory_GiB=8.72,current_saved_max_num_seqs_per_engine=4,cache_spec_one_block_per_request=True,cache_spec_prefix_cacheable=False,C2_compiler_candidates=28,C2_compiler_binary_identity_bound=False,finite_capacity_examples=72,F34_graph_nodes=10,F34_graph_edges=13,corrected_undefined_graph_endpoints=2,unique_point_source_files=len(sourcefiles),branch_conditions=len(branches),new_model_import_or_device_or_service_work=False,all_jobs_terminal=True,whole_modules_closed=0,overall_complete=False,evidence_refs=refs,remaining=remaining)
 write(O/'C2-allocation-summary-v25.json',summary)
 gate=read(O/'PROFILE-CODE-QUESTION-GATE-v25.json');gate.update(answered_by=['C2-cache-object-and-allocation-contract-v25.json','F34-cache-allocation-variants-v25.json'],closed_scope='Directcacheobject/spec/prefixcontract,V41tupleplanner/worker/hooksourcepath,restrictedstandalonecapacityexclusion,currentcachedlogsettings.',not_closed='Actualpointer/blockownership/allbyteobservers/nativeexecutionhash/wholemoduleclosure.',stop_rule='Do notcontinuewholeKVorcompilerinternalsmerelytoupgradealreadyconditionalcandidate. AdvanceH07/F33/otherprofile-raisedremainingcoverage;onlypointqueryfurtherifaparticularconclusionneedsit.',overall_complete=False);write(O/'PROFILE-CODE-QUESTION-GATE-v25.json',gate)
 report='''本轮补齐F34的缓存对象与V4.1专用分配证据，并在明确合同下关闭“ring单独减半能否减少pool分配”的局部问题：若tuple内不变的SWA页仍不小于原ring页，单独把ring改为16槽不会缩小该tuple backing。状态写入裁剪与容量收益分开判断。整体仍未完成，没有实现优化或运行NPU。

沿profile已定位的对象定点读取后，确认CircularBufferSpec只保留每请求一个block，不参与前缀缓存，也不使用普通slot mapping。DeepseekV41CacheLayer只注册/绑定cache视图并提供spec/backend；在本次限定model/DSA标识范围中，pool_projected是ring的显式取值入口。这不等于完整raw backing没有其他读者：对象会进入static_forward_context和worker的缓存列表。

关键分配路径是V4.1专用逻辑。hook的V41分支选择group_cache_specs、get_deepseek_v41_kv_cache_config及专用pool-byte divisor，三个module-level安装赋值也已取证。每个tuple有独立backing，tuple内部的long-KV/index/可选folded及ring/SWA别名复用该backing。worker在早期V41分支按descriptor分别分配并以block_stride构造视图。通用upstream按最大group计算的规则不能直接替代这个V41按tuple容量之和定预算的合同；KVPP等优先分支另列条件，没有从源码冒充历史Python对象snapshot。

tuple容量为max(long/index/folded总payload,各alias的未填充页大小)。令不变SWA页S>=原ring页R>=新ring页R'，则对任何其他payloadL都有max(L,S,R)=max(L,S,R')。profile相容的页几何中，128×512 BF16 SWA页与32×1024 FP32 ring页同为131,072字节；16槽ring为65,536字节，但SWA下限仍在。因此在不改变tuple成员、其他页、预算及override的域内，容量之和、num_blocks和descriptor allocation均不变。原先2.268GB→1.134GB只是ring逻辑视图模型，不能当作额外独立分配或可释放显存。若改变SWA表示或tuple/pool策略，属于更大的整缓存重组，需另审所有消费者和生命周期。

F34-A的奇数位置维护裁剪仍保留条件：必须维护所有偶数FP32值、写顺序及旧读→新写WAR。状态命名空间应包括descriptor/tuple、global block ID、slot与generation。不同Tensor视图可能覆盖同一backing，安全依据是有效block所有权与回收/重新初始化合同，不能只说“ring对象私有”。完整byte ABI、transfer/checkpoint等外部观察者仍未全部闭合，prefix_cacheable=False不自动排除它们。F34单独变体保留原update→norm及index publish→long publish顺序，F18/F19重排仍是独立条件组合。

进一步复用本次采集controller指定的service.log，确认10月30日21:35初始化采用LBNHC、可用KV预算日志为8.72GiB，启动参数每个engine max_num_seqs=4。日志另报353,802 tokens与34,816-token请求的10.16并发估计；这些是配置和sizing估计，不是新健康/空闲观测、吞吐测试或硬件上界。先前runtime/home中选到的10月27日历史日志没有用来替代这些当前采集证据。

还取回保存runtime中的73个编译metadata条目，其中C2 pool/update各14个，分别有14个不同binary/TTIR hash。.source后缀文件实际是IR，并非原始Python文件；存在缓存也不证明某个实例被执行，ComputeInfo的_pool_kernel名称不足以从14个候选中选择一个。未按名字、mtime或最近时间配binary，也没有把这些编译选项当成实际已加载内核的精度证明。

F34候选图原先有2个未声明的简写端点，本轮已改成10个明确节点、13条定义完整的边，并保留原顺序约束。结构DAG校验不能替代native语义验证。一项本地摘录因upstream文件未在本地快照中保存而停止，错误保留；修正版只使用两个已与服务器SHA核对的必要快照完成72组容量算术检查，没有实现优化。

主台账保持34项模型候选、7项框架候选。新增15项缓存对象/分配/alias/编译证据条件，并将F34-B的独立容量收益在当前SWA-anchor域内标为排除，避免重复追查这个已回答问题。全模块、其他消费者、当前EngineCore成本、MC2内部原因与同域理论比较仍有缺口，26整模块未标闭合，Goal继续active。最后健康观测仍是旧v12post。
'''
 (O/'PROFILE-FINDINGS-v25.md').write_text(report)
 pub=R/'publication-draft/FUSION-INVENTORY.md';old=pub.read_text();(O/'FUSION-INVENTORY-before-v25.md').write_text(old)
 text=old.replace('当前持续分析台账（v24）','当前持续分析台账（v25）').replace(str(O/'candidate-profile-ledger-v24.json'),str(O/'candidate-profile-ledger-v25.json')).replace(str(O/'PROFILE-FINDINGS-v24.md'),str(O/'PROFILE-FINDINGS-v25.md'))
 lines=text.splitlines();lines=[('| F34 | C2 ring状态的可观测性裁剪、稀疏维护与表示重组 | 条件/受限排除 | 偶数历史值不变量支持有条件的奇数维护裁剪；保留全部读者、块所有权与回滚版本。v25确认V41tuple内SWA页下限不变时，单独16槽不减少backing分配；不把逻辑数组大小当实际显存收益。 |')if l.startswith('| F34 |')else l for l in lines];pub.write_text('\n'.join(lines)+'\n')
 action=('继续Loop004，v25实质进展。先读C2-allocation-summary-v25.json、C2-cache-object-and-allocation-contract-v25.json、F34-cache-allocation-variants-v25.json、module-coverage-gaps-v25.json。'
  'F34-A仍条件；局部内存问题已回答：V41get_layer_tuples容量max(long+index+folded,各alias unpaddedpage)，Ring32x1024FP32与SWA128x512BF16均131072B；SWAanchor>=旧ring时单独ring16slot65536不改变tuplecapacity,sum页/numblocks/descriptorsize不变。不要再把2.268->1.134GB逻辑view当独立allocation或释放显存。'
  'CircularBufferSpec1block/request,prefix_cacheableFalse,uses_slot_mappingFalse；DeepseekV41CacheLayer注册staticcontext并bind视图。标识局部范围唯一显式ringvalue取入口pool_projected不等于rawbacking所有读者关闭。V41tuple内ring/SWA/long/index有alias，命名空间(tuple,globalblockID,slot,generation)，必须owner/lifecycle隔离，不是Tensor对象disjoint。'
  '当前workerV41earlybranch每descriptor独立backing，tuple层名alias；KVPP>1可优先别路。普通sourcecontract通过patch平台3个modulelevelassignments及V41helperguards绑定，历史livefunction对象未snapshot。genericKVcache MAXgroup公式不是V41 SUMtuple策略，别混。cache_config.pySHA467dddca82e2528ebbec14c78682486a39bffb4f096675d8b1d2d140f1a7014b；workerSHA568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c；patchkvutilsSHA5fc49ebb9b64a2f14c4eb38b8c66d3daf1b7612961cf76392cf113d76f555aab。'
  '当前capture service.log SHAa02e582afaf94f0911b0bfb62ddac9028f2ac25936e4776373fc0ebfd2e1c61e在P/runtime/logs/service.log，LBNHC/8.72GiB/max_num_seqs4perengine，353802token/10.16concurrency为sizing估计非硬件界/实测TPS/新health；selected oldhome10月27日志未冒充当前配置。'
  '保存compiler73metadata/73binary，C2pool14/update14均各14distinctbinary+TTIR。source后缀是IR，ComputeInfo仅名未executed hash绑定，不按mtime/近邻配。停止为已可定性的conditional候选追扫整个compiler。F34graph2undefinedshorthand修成10nodes13edges并保留U->N/indexpublish->longpublish当前控制，F18/F19fork另域。'
  'v25remote PMU结果SHAs：C2-cache-reader-and-cache-spec-contract-v25=30569f0fa87afbcd66e317fa5fa1b5af13395b222b222df9bc2ce9fd00e6f159；C2-cache-allocation-and-binding-policy-v25=d23ab42da5b10bbd017a629e89b338148d8a2221fd892ebe164b8b7e1c92798f；saved-C2-compiler-and-cache-log-surfaces-v25=9cb5ed173954d56e5aa9e238a68e46e9e1a0b5c9c2b0ceba7709aeb8af0f9c00；V41-tuple-cache-policy-dispatch-v25=1d1d173101887b8aaa022fbebe1442c150f017dc6e1a2ce73277aca237b76c52；current-capture-service-cache-log-v25=396eb33c775a939eb1a7f1941ffe024f824ac4a1062acba91bef003cd67141e4；V41-cache-hook-installation-and-startup-params-v25=bfbaa99e3c39270770c8e973f61c8f1c01fde73715dd61fd5197ce8e504de75c。'
  '下一批转H07原始ID cachedreplay外部embeddingproducer/consumer具体缺口，或F33其它边界，先用现有profile fullcaller/entryguard/copy evidence，再点必要代码；不要继续重复已闭合F34独立容量问题。F34-A若后续实施才需native/owner ABI证据，本任务不实现。当前EngineCore/producerready/prefixhash timing若关键，仍按既有授权freshhealth/ownership/frozenworkload/clone/restore准备有限诊断，别用旧diagPID或假定空闲。'
  '34model/7framework/0wholemodule;3136candidate PMU角色/849missingPMU/16unboundgraphs/26theoryINCOMPLETE保留。1local missing-snapshotfile失败保留，修正72容量casespass，6remoteCPUjob terminalpass，无modelimport/NPU/服务变更，lasthealthv12post，Goalactive。')
 scripts=['read_C2_cache_specs_and_readers_v25_166.py','read_C2_cache_allocation_policy_v25_166.py','query_saved_C2_compiler_artifacts_v25_166.py','locate_v41_cache_policy_dispatch_v25_166.py','read_saved_capture_cache_log_v25_166.py','read_v41_hook_installation_v25_166.py']
 descs=['Exactcacheobject/Circularspec/prefixcontracts andbounded directstate-reader localization;notallaliasclosure.','Genericallocationcontexts andworker/DSAcachebinding source read;subsequentV41specificpathkeptdistinct.','73savedcompilemetadata entries/28C2binarycandidates andboundedhistoricalhomelogsurfaces retainedwithout executionidentityclaim.','ExactV41modelcachehelper dispatchfunctions located;normal sourcecontract bound.','Exactsavedcapture service.logtechnicalKVinitializationsummary SHA/stat pinned;notnewhealth.','3modulelevelhookassignments andwhitelistedcurrentstartupmaxseq4parameter boundwithout import/activation.']
 jobs=[('profile'if i in[2,4]else'review','pass',descs[i],[N/(labels[i]+'.json'),N/scripts[i]])for i in range(6)]
 jobs.insert(4,('review','error','Initial local sourceexcerpt stopped beforearithmetic because oneupstreaminterface filewasnotinthepartiallocalsnapshot;originalscript/output preserved.',[N/'review_v41_tuple_capacity_v25.py',N/'zcode-V41-capacity-v25-output-20261031.json']))
 jobs +=[('review','pass','Corrected necessary2fileSHA-boundexcerpt,72finitecapacity examples andF34typedgraph consistency reviewed;no nativeimplementation.',[N/'review_v41_tuple_capacity_v25_v2.py',O/'V41-tuple-capacity-and-F34-graph-review-v25-v2.json',N/'zcode-V41-capacity-v25-v2-output-20261031.json']),('review','pass','Main reconcilesV41SWA-anchorcapacityexclusion,odd-writeconditionaldomain/currentlog/compilelineage and34candidate15conditionledgers.',[O/'C2-cache-object-and-allocation-contract-v25.json',O/'F34-cache-allocation-variants-v25.json',O/'C2-cache-allocation-condition-ledger-v25.json',O/'C2-compiler-lineage-contract-v25.json',R/'candidates/F34.json',pub,O/'PROFILE-FINDINGS-v25.md'])]
 runs=[]
 for i,(kind,status,desc,arts)in enumerate(jobs):
  rid='run-'+(now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ');run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90-166','--kind',kind,'--command','Preparedv25retrospectiveregistration: '+desc,'--code-ref','Saved166profile/cacheartifacts/exactcachedependencyfunctions;no modelimport/NPU/targetimplementation/servicework','--working-directory',str(N));args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',desc]
  for p in arts:assert p.exists(),p;args+=['--artifact',str(p)]
  ctl(*args);runs.append(dict(run_dir=run,status=status,description=desc))
 cp=dict(status='PROGRESS',previous_turn_classification=summary['previous_turn_classification'],current_verified_changes=['V41cacheobject/spec/tupleallocation/hookcontract bound;standaloneringhalfcapacitygain excludedinunchangedSWA-anchor domain.','Currentcapture logLBNHC/8.72GiB/maxseq4boundseparatelyfromhistoricallogs;28compiledC2candidatesnotexecutionidentity.','F34graph2undefinedendpoints fixedandoriginalF34ordering retained;15finiteallocation/aliasconditionsrecorded.'],summary=str(O/'C2-allocation-summary-v25.json'),candidate_ledger=str(O/'candidate-profile-ledger-v25.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v25.json'),module_ledger=str(O/'module-profile-ledger-v25.json'),branch_matrix=str(O/'module-phase-branch-matrix-v25.json'),report=str(O/'PROFILE-FINDINGS-v25.md'),source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v25.json'),coverage_gaps=str(O/'module-coverage-gaps-v25.json'),pending_full_scope_domains=remaining,model_candidates=34,framework_candidates=7,remote_job_status='6CPU-onlyremotejobs terminalpass;1localmissingsnapshotexcerptfailure retained;correctedlocalreviewpass.No model/NPU/service/healthwork.',next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=stamp);write(O/'REVIEW-CHECKPOINT-v25.json',cp)
 for p in[L/'state.json',T/'index.json']:
  d=read(p);assert d['status']in['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=stamp);write(p,d)
 for p in[L/'status.md',T/'status.md']:p.write_text('\n'.join(('- 下一步：'+action)if x.startswith('- 下一步：')else('- 更新时间：`'+stamp+'`')if x.startswith('- 更新时间：')else x for x in p.read_text().splitlines())+'\n')
 (O/'profile-summary-before-v25.json').write_bytes((R/'profile-summary.json').read_bytes());prof=read(R/'profile-summary.json');prof['supplementary_C2_allocation_v25']=dict(summary=cp['summary'],standalone_ring_half_capacity='EXCLUDED_IN_UNCHANGED_V41_SWA_ANCHOR_DOMAIN',model_candidates=34,framework_candidates=7,whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',prof)
 for p in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
  d=read(p);d.update(latest_profile_first_review=cp,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);d['next_read_only_action'if p==R/'STATE.json'else'next_action']=action
  if p==W/'STATE.json':d.update(profile_candidate_ledger=cp['candidate_ledger'],profile_module_ledger=cp['module_ledger'])
  if p in[N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[cp['summary'],str(O/'C2-cache-object-and-allocation-contract-v25.json'),str(O/'F34-cache-allocation-variants-v25.json'),cp['coverage_gaps']]
  write(p,d)
 compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T));write(O/'task-run-registration-v25.json',dict(runs=runs,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
 arts=[p for p in O.glob('*v25*')if p.is_file()and not p.name.startswith('checkpoint-manifest')]+[N/(k+'.json')for k in labels]+[N/s for s in scripts]+[N/'review_v41_tuple_capacity_v25.py',N/'review_v41_tuple_capacity_v25_v2.py',N/'zcode-V41-capacity-v25-output-20261031.json',N/'zcode-V41-capacity-v25-v2-output-20261031.json',N/'finalize_C2_allocation_v25.py',R/'candidates/F34.json',pub,N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
 write(O/'checkpoint-manifest-v25.json',dict(generated_utc=stamp,artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in arts],whole_modules_closed=0,overall_complete=False))
 print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v25.json'),runs=len(runs),pass_runs=sum(r['status']=='pass'for r in runs),retained_error_runs=sum(r['status']=='error'for r in runs),model_candidates=34,framework_candidates=7,validation=validation,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
