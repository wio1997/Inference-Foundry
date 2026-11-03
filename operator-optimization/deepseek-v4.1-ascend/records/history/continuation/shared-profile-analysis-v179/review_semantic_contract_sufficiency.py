"""Record the actual module/domain review, with explicit corrections and limits."""
import copy
import hashlib
import json
from pathlib import Path
N=Path('/Users/wio/work/166-vllm-analysis-continuation-20261030');O=N/'profile-first-20261031';H=N/'shared-profile-analysis-v179';P=N/'shared-profile-analysis-v178'
def read(p):return json.loads(p.read_text())
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def save(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

# Each finding below was reviewed against all named branch descriptions,not inferred from a count.
findings={
 'M00':'Keep selected implementation/mode/precision/quantization/parallelism and load/publication generations as selectors. Initial packing/registration is separate from steady operators. Unattested native mode constrains that private claim rather than making all public analysis unread.',
 'M01':'Frozen successful text requests bind the current input/sampling case. Tokenizer/defaults/constraints/logprob/seed/stop/multiplicity/stream/prefix and error/lifecycle alternatives keep original control/output ports. Host batching/caching does not change request isolation or claim zero frontend cost.',
 'M02':'Hash namespace,global block-ID/SWA anchor,normal/full32/compact16/pairedowner layouts are distinct cases. Packed halves keep ID reservation until both lifetimes and queued writers drain. Detach/compaction is a new representation with metadata/ABA proof obligations,not a measured half-capacity win.',
 'M03':'Scheduled/new/speculative/valid/padded rows and DP/TP/EP participation are different domains. Empty local and cache/encoder-only branches can still publish state or receive work. FIFO buffer publication,cancellation and accepted-state generations remain readers of device outputs.',
 'M04':'Combined gather/scatter must preserve negative-source indexing versus destination error rules,duplicate destinations,sample-then-draft priority,old-source snapshots and untouched prefix/tails. Position/count/slot planes keep widths and sentinel writes. Batch remap snapshots3gathers before computedwrite;acceptedELSE is read after that write,so snapshot-all needs disjointplanes or original alias-effect emulation.',
 'M05':'Invalid vocab row0 then positivezero,nonfinite/cast behavior,STANDARD versus alternate collectives and noalias SPclone are separate contracts. Direct persistent embedding output needs every prefix/tail reader and bank retirement. Historical80BOOL24records do not identify current171SP/C2fields;current role evidence is separate.',
 'M06':'HC4residual,onehot initial premix,external/current coefficient forks,roundedY,Post accumulation and terminal/aux paths remain distinct. Do not reassociate/replace0*Inf or BF16rounds. EarlyY/newhandoff,virtualHC and directPost->Pre are conditional protocols with finite resources and Engram/aux readers retained.',
 'M07':'Input5120RMS,QR1280RMS,KV512RMS and partial64RoPE differ. QR quantizer reuse requires the same rounded value/quantizer/generation;localQ and globalKV rows differ. EmptyQ keeps required cache/global output. Legacy postQBheadRMS is excluded from this fixed V4.1 query entry by source,not absence of a trace event.',
 'M08':'C1/C2typed projection,pool completion/odd residual/fullprezero/norm and oldringread-before-update form a fork with cache/key/latent/state readers. Fullrow norm must see required unwritten-zero rows. Full32/compact16/paired layouts preserve allocator/SWAanchor/half-owner lifetimes;capacity is not traffic.',
 'M09':'Separate mode1fresh candidate2048block output/publication from512position TopK and aliased prepare-indices output. Mode2four filters preserve finite outside fillers/latestvisible/ties. QueryFP16scale and keyFP32scale quantization differ;sort/index occurrences preserve multiplicity. All8TopKsourceepochs and candidatebank retirement retained.',
 'M10':'SWA/C1/C2/draft attention retain original visible multiset,sinkonce,score/exp/PV rounds and cache generation. LocalnoQ differs from uniformmetadataNone. Provisional selection needs certified final sets or retained logits/partialPV eviction rules. CurrentKV readthrough/delayedpublication requires all existing cache readers and actual acceptance.',
 'M11':'WA/WB keep BF16intermediate and per-rank roundedWBbeforeHCCLreduction. G1permutations are already native-fused;no extra transpose saving. Repartition/panel/WA-beforeexchange alternatives retain actualA/M/padding,all8owner weight groups,send layout and shared Cube/link budget;fullweight precomposition is numerically excluded.',
 'M12':'Keep rounded NormCast dual outputs,router nonlinear/tie/bias/hash contracts,receiver-based work and existing fused MC2. Sender0/padcapacity do not imply receiver0/active128rows. V1/V2return,clip/credit/flag/parameter protocols differ. Publicpad replacement is separate from unsupported privateMemSet deletion;rowprune needs every raw/state reader.',
 'M13':'Shared layout/TP/SP/DP/quant/activation cases preserve fullIamax,INT32partials,dequant/BF16round,down output/legacytuple and routedmerge readers. Replica-to-shard/newquantized ABI and overlap are separate protocols,not additive free gains;same hidden does not mean equal quantizers.',
 'M14':'Hash history/slot/DEAD/masks,group32formats,ownedlookup and layer1/14gate retain all publication/future readers. Positive host-UVA/overlapFalse snapshot replaces historical async label. Inactivegate keeps hidden+0*value specials;prepared parameters/unchanged pre-rotationhidden reuse is generation-keyed. Hostpayload and HBM remain separate pools.',
 'M15':'Keep target rawHC/premix snapshot versus normalizedhead/orderedrowunion and draft rawconfidence/aux inputs. Localcollapse/no-buffer/packedraw alternatives preserve original two rounds,fullHstatistics,rank/trim/capture and other readers. No terminal output deletion from sourceclass arguments alone.',
 'M16':'Target/draft TP8H5120heads and replicatedH256Markov have different ownership/weights. Postprocessed roundedties require processedbest. BestID summaries,fullscore/distributedprobability and sharded/owner Markov are distinct protocols with explicit collective-support fallback;dummy allrow0 reuse remains a guarded value case.',
 'M17':'Preserve raw/processed/probability/entropy/logprob forks and exact RNG effects,including seeded unused/default draws. Actualn1 differs from capacityS. GreedyTriton,PyTorchgeneral/n1,randomrecovery and pivot/cutoff/race paths keep their comparator,ties,oldoutput/suffix and scratchownership. Preserve-score argmax conversion does not require all-scoredeadness.',
 'M18':'Three ordered target HCmeans feed commonmain/context5120norm,then3distinct layer40/41/42WKV,norm/RoPE/cache/group/slot paths. Commoninput and joint1536projectionloadreuse do not equate outputs. Guards originally after projection,fullcontextbanktail/readers and FLOATtable->BF16C/Sround remain. Aux deferral needs all3sources/lateI/lifetime and lostoverlap priced.',
 'M19':'Draft SPpadding keeps suppliedTrueprefix/Trueinvalidtail and clone semantics. MaskNone/metadataNone/localnoQ are distinct. Local2/3Q differs from global10/20KV;eachquerylayer uses its own contextKVparameters/cache. FinalHCcollapsebeforeSPgather and rawconfidence are baseline,not new elimination.',
 'M20':'Five serial MarkovIDresolutions preserve roundedbase+bias,seed plus4previous embeddings,rawhiddenconfidence and policy state. SelectedW1rows are not fulltable traffic;bias memo/certifiedprune needs exact roundedgeneration/nativeerror bounds. Futurekeep cannot prune its own producer or arbitrary noncausaldraft queries.',
 'M21':'Metadata keeps request/token/group/ratio/page/slot coordinates,validity/tail/empty/capture writes,QLIplanning in graphmode and deferredproducer/event/bank lifetimes. Candidate,TopK,C2mask and SPmask are separate effect domains. Dummyzeroindices require callbackdemand proof;no shape-only reuse or deleted queued writer.',
 'M22':'Capture/warmup/replay/update/pool/weakworkspace and exception/finally retain original control/object lifetimes. Opaque native task effects stay inside preserved publiccomposites. Constant address/empty descriptor does not establish no writes or readydata. Startup capture cost is not automatically recurrent latency.',
 'M23':'Keep exact operation count convention,rank/group/physicalcard/owner namespace and original reduction rounds. HCCLTP8 does not identify MC2EP16topology. A2A/WA/WB/RS,embeddingownerexchange,auxbatching/sharedshard alternatives are protocol choices with allrank readers and finite cuts,not independent additive speedups.',
 'M24':'Computed/accepted updates have3oldgathers before computedwrite and acceptedELSE read after computedwrite. Preserve both that cross-row readset and any cross-plane write->read alias effect;allsource hoisting is conditional. C2/KV/Engram/index/aux/policy/expert/copy banks use content/request/parameter/slot epochs plus lastreader AND lastqueuedwriter retirement. Pointer equality permits no cross-iteration CSE.',
 'M25':'Output IDs/logprobs/stop/UTF8/stream/parent/resume/cancel/error/release and IPC readiness retain original request-generation and queue observers. H01/H05 may coalesce only their declared notify/transfer/payload boundaries;no inference from sampled host idle time to request criticalpath saving.'}
matrix=read(O/'module-phase-branch-matrix-v178.json')
domains=copy.deepcopy(read(P/'exact-guard-operator-contract-review.json'))
ports=read(H/'host-score-and-state-port-contracts.json')
g=read(H/'hierarchical-G0-with-boundary-and-state-ports.json')
catalog=read(N/'shared-profile-analysis-v177/profile-selected-operator-guard-census.json')
assert set(findings)=={m['id'] for m in matrix['modules']}
domain_addenda={
 'DSPARK_MODEL':'Context KV method is invoked per layer:3separate WKV/norm/cache/group/slot identities. Commoncontext is not commonKV result;originalpostprojection slot guard is retained.',
 'PROPOSER_STATE':'Batch correction/remap snapshots3oldgathers beforecomputedwrite;acceptedELSEisreadafterthatwrite. Hoistallinputs onlywithdisjointplanes ororiginalalias-effect emulation;preserve accepted/nextinput/position generations.',
 'INHERITED_RUNNER':'Combined scatters retain negativeindex versus destinationerror,duplicates,priority and oldsourcealias snapshots. No per-row inplace remap without a complete readset proof.',
 'DSA_RUNTIME_METADATA':'Mode1candidate_out is fresh and copied to candidate2048bank;mode2candidate is input-only. PositionTopK512/prepare selfalias/candidatepublication are separate APIs and epochs.',
 'INDEXER_QUERY_PREP':'Keep separate candidate2048block and positionTopK512banks,all4mode2filters and8position sourceepochs. Filler/tie/latestvisible/multiset rules survive newrepresentation.',
 'SEQUENCE_PARTITION':'SPprovided-maskTrueprefix andTrueinvalidtail are distinct from C2completionmask views. Whole publicpad replacement preservesclone/noalias;private helperfields remain unknown.',
 'TARGET_MODEL':'Preserve originalHCaux ordering,per-layerKV identities and guardedallrow0dummyhead case;expanded publicgraph is an accounting hierarchy,not a native schedule.',
 'SLOT_MAPPING':'Different draftlayer_group_idx values may select different slotbanks;baseline projection may run beforeNone/empty store guard. Earlierprune is a conditional newdemand domain.'}
for d in domains['domains']:
    d['semantic_review_v179']={
      'verdict':'CONDITIONAL_PUBLIC_OPERATOR_EFFECT_ANSWER_ACCEPTED',
      'module_case_reviews':[m for m in d['operator_modules']],
      'specific_addendum':domain_addenda.get(d['id'],'Existingtyped/precision/consumer/state and protected-control answer retained;no additional publiceffect required by the360rowreview.'),
      'native_branch_taken_or_private_mutability_not_inferred':True}
domains.update(parent=pin(P/'exact-guard-operator-contract-review.json'),
    semantic_sufficiency_review=str(H/'semantic-contract-sufficiency-review.json'),
    hierarchy_authority=str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),overall_complete=False)
save('exact-guard-operator-contract-review.json',domains)

module_reviews=[];row_reviews=[]
for m in matrix['modules']:
    mid=m['id'];envelopes=[s['id'] for s in ports['state_envelopes'] if mid in s['modules']]
    guard_domains=[d['id'] for d in domains['domains'] if mid in d['operator_modules']]
    module_reviews.append({'module':mid,'semantic_rows':[r['id'] for r in m['branches']],
      'reviewed_answer_and_consumer_effects':findings[mid],
      'state_envelopes':envelopes,'guarded_effect_domains':guard_domains,
      'verdict':'REGISTERED_FIXED_CASE_PUBLIC_OPERATOR_ANALYSIS_REVIEWED_WITH_EXPLICIT_CONDITIONAL_GATES',
      'unanalysed_public_effects_in_these_rows_after_corrections':[],
      'native_private_aliases_modes_and_newkernel_precision':'CONDITIONAL_OR_UNSUPPORTED_PRIVATE_EDIT;not silently legal or0'})
    for r in m['branches']:
        row_reviews.append({'semantic_row':r['id'],'module':mid,'original_operator_answer':r['branch'],
          'source_contract_evidence':r.get('evidence',[]),
          'current_module_review':mid,'current_state_envelopes':envelopes,'current_guarded_effect_domains':guard_domains,
          'conditional_limits_retained':r.get('current_operator_review_v173',{}).get('source_or_runtime_conditions',r.get('gap','')),
          'scope_status':'SEMANTIC_ANSWER_REVIEWED;runtime/native applicability remains a separately typed condition',
          'original_runtime/cohort_scope_is_not_new_observation':True})

# Verify reused contracts are attached to the same source files as the current profile-selected census.
def suffix(path):
    for marker in ['/frozen-source-snapshot/','/vllm-workspace/']:
        if marker in path:return path.split(marker,1)[1]
    return path
current={suffix(f['path']):f['sha256'] for f in catalog['files']}
reuse=[]
for name in ['index-context-operator-contract-v64.json','heads-sampling-operator-contract-v65.json']:
    for s in read(O/name)['source_files']:
        key=suffix(s['path'])
        if key in current:
            assert s['sha256']==current[key],(name,key)
            reuse.append({'contract':name,'source':key,'sha256':s['sha256'],'current_profile_source_hash_match':True})
assert any('/deepseek_v41/dspark.py' in r['source'] for r in reuse)
assert len(row_reviews)==360 and len({r['semantic_row'] for r in row_reviews})==360
review={
 'status':'SEMANTIC_CONTRACT_REVIEW_COMPLETED_WITH4_PUBLIC_GRAPH_REFINEMENTS;FINAL_REQUIREMENT_AUDIT_PENDING',
 'previous_goal_turn':'PROGRESS',
 'pins':[pin(O/'module-phase-branch-matrix-v178.json'),pin(P/'exact-guard-operator-contract-review.json'),
         pin(H/'hierarchical-G0-with-boundary-and-state-ports.json'),pin(H/'host-score-and-state-port-contracts.json')],
 'module_reviews':module_reviews,'semantic_row_reviews':row_reviews,'guarded_effect_domain_reviews':len(domains['domains']),
 'reused_source_contract_hash_bindings':reuse,
 'corrections':[{'finding':'DSparkcommoncontextwasovercoarsenedintoequalKVstate','answer':'3layer-specific projection/norm/cache/group/slot paths restored','families':['F27','F29','F30','F31']},
   {'finding':'candidatefreshoutput/publicationdistinctfromaliasedprepareandpositionTopK','answer':'separate candidate2048bank with4readers and next-epoch retirement','families':['F09','F10','F12','F13','F19','F31']},
   {'finding':'batchremap3gathersbeforecomputedwritebutacceptedELSEreadafterit','answer':'crossrowgatheredsnapshotplusorderedcrossplanealiascontract;[21,11]versusunsafe[21,22]witnessretained','families':['F20','F31','H02','H06','H07']},
   {'finding':'SPvaliditymaskmustnotbeidentifiedwithC2maskbyBOOLshape','answer':'separate provided/Noneprefix andTruepaddingtail effectenvelope','families':['F15','F20','F24','F28','F31','H02','H06','H07']}],
 'adequacy_argument':[
   'Every registered semantic row was read in this review. Detailed typed/numerical/cost answers remain in the pinned original contracts;the26findings address their publicconsumer/alias/state requirements.',
   'The27guard-domain answers preserve wholefunction continuations and opaque imported/native calls unless a candidate explicitly replaces that typed publicinterface. An unknownprivateeffect remains inside that retained interface,not an omittedindependence assumption.',
   'Profile/source-selected72files/573functions/1867refs and retained1101staticreviews provide the fixed source/case frontier. Frozen textpayload excludes vision-only execution by inputcontract,not by absenceofevents;loadedvisionresidency retained.',
   'This is a bounded fixedmodel/version/hardware/load analysis and declaredrule search,not a theorem over allpossiblealgorithms or newworkloads. Native feasibility,newkernel precision and physicalrates remain conditional exactly where used.'
 ],
 'no_new_NPU_model_restart':True,'target_source_bodies_read_this_review':[str(H/'targeted-batch-source-order-review.json')],
 'registered_360_public_effects_after_correction':'NO_UNANALYSED_EFFECT_IDENTIFIED_IN_REVIEW',
 'completion_status':'UNPROVEN_UNTIL_REQUIREMENT_BY_REQUIREMENT_CURRENT_STATE_AUDIT',
 'overall_complete':False}
save('semantic-contract-sufficiency-review.json',review)

# Changed-effects rule round; all answers refer to existing families,not newCartesianpairs.
rules=[
 ('PAIR_PRODUCER_CONSUMER','F29mayfuseeachlayerWKV/norm/RoPE/scatterbutcannotreuseanothervaluebysamecontextalone. F12candidatepublication isfreshoutputcopyunlessnewoutABI.'),
 ('TRIPLE_CHAIN_OR_FORK_JOIN','Commoncontext1536jointprojection retains3rounded512outputs/norms/cachewriters;candidate512/2048forks andbatch3gathers/2orderedwrites retain every branch.'),
 ('MULTI_EXTEND_ONE_BOUNDARY','Composecontext/banks/metadata/nextiterationonlywithperlayerparameters/groups,fullcontextbanktailandreaders,lastqueuedwriter and coherentold/newstateplanes;budgets counted once.'),
 ('SPLIT_COMPOSITE_AT_DOCUMENTED_BOUNDARY','Publicperlayercontext andcandidateout/bank are documentedboundaries. PrivateMemSet/MC2/taskpackets stayunsupported/conditional;no independenthelper deletion.'),
 ('SHARED_INPUT_REUSE','Sharecommoncontext/inputloads,not3layerKVresults orsource24indices/source20KV. Preparedquery/keyquantizers andBF16C/Srounds differ. Cross-rowgathers requireoldvalues;acceptedELSEmustretainpostcomputedwriteepochifcrossplanealiases.'),
 ('CROSS_MODULE_BOUNDARY','Source20candidatebank/readers24/28/32/36,3draftcacheowners,andbatchacceptedstate eachretain distinctgenerations,perrangeRAW/WAR/WAWandnextiterationretirement.'),
 ('MULTICONSUMER_PRESERVE_ALL_USES','Keepfullcontexttail/queryWKVparameterreaders,allcandidatefilterfiller/tierules,providedSPmaskTrueprefix,alloldbatchgathers and nextinput/output/cache/state observers.')]
save('semantic-refinement-rule-round.json',{
 'status':'DECLARED7RULE_ROUND_ON_SEMANTIC_REFINEMENTS_DONE',
 'source_hierarchy':pin(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
 'all42_composition_authority':pin(N/'shared-profile-analysis-v174/all42-composition-coverage.json'),
 'prior27source_template_rule_round':pin(N/'shared-profile-analysis-v176/declared-source-template-rule-review.json'),
 'rules':[{'rule':r,'answer':a,'new_non_equivalent_candidate_family':None} for r,a in rules],
 'new_candidate_family_count':0,'new_conditional_variants_already_in360rows':True,
 'all_possible_pairs_or_subsets_not_enumerated':True,
 'whole_scope_SCAN_COMPLETE_claim':'Pending final scope/requirements audit;no newfamily in this changed-effect round alone is not fullmodelproof',
 'overall_complete':False})
save('semantic-review-checks.json',{'status':'PASS','module_reviews':len(module_reviews),'semantic_rows':len(row_reviews),
 'guard_domains':len(domains['domains']),'source_contract_bindings_matched':len(reuse),'refined_state_envelopes':len(ports['state_envelopes']),
 'review_was_not_reference_counts_only':True,'no_new_native_eligibility_or_performance_claim':True,'overall_complete':False})
print(json.dumps(read(H/'semantic-review-checks.json'),ensure_ascii=False))
