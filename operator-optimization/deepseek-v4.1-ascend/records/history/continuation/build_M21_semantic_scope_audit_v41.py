from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
from finalize_origin_coverage_v9 import O,read,write
from finalize_Engram_whole_frontier_v30 import packet


def main():
    stamp=datetime.now(timezone.utc).isoformat()
    scope=packet('M21-scope166-v41',False)
    union=packet('M21-union166-v41',True)['result']
    m20=packet('M20-profile166-v36',False)
    m12source=packet('F04-MC2-alignment166-v29',True)
    assert union['overlap_TASKs']==0 and union['union_TASKs']==84297 and union['union_APIs']==64476
    outside=[h for h in scope['chain_headers']if not h['in_v37_selector']]
    assert len(outside)==23 and sum(h['TASKs']for h in outside)==8264
    m20keys={h['chain_key']for h in m20['native_chain_headers']}
    reconciliation=[]
    for h in outside:
        leaf=h['project_frames'][0]
        if h['chain_key']in m20keys:
            state='REUSED_EXACT_M20_CHAIN';evidence=['M20-confidence-and-next-ID-summary-v36.json','M20-Markov-confidence-and-verification-contract-v36.json']
        elif '/ops/fused_moe/prepare_finalize.py('in leaf:
            state='REUSED_M12_SOURCE_CONTRACT';evidence=['F04-precision-routing-and-row-contract-v29.json','F20-routing-ID-reuse-review-v29.json']
        else:
            state='NEW_LOCAL_INITIALIZATION_INTERFACE_REVIEW';evidence=['M21-positive-outside-selector-initialization-source-v41.json']
        reconciliation.append(dict(h,review_state=state,evidence=evidence))
    hist={s:dict(chains=sum(x['review_state']==s for x in reconciliation),TASKs=sum(x['TASKs']for x in reconciliation if x['review_state']==s))for s in sorted({x['review_state']for x in reconciliation})}
    assert hist=={'NEW_LOCAL_INITIALIZATION_INTERFACE_REVIEW':{'chains':9,'TASKs':592},'REUSED_EXACT_M20_CHAIN':{'chains':10,'TASKs':4472},'REUSED_M12_SOURCE_CONTRACT':{'chains':4,'TASKs':3200}}
    gapfile='M21-original-profile-gap-reconciliation-v41.json'
    write(O/gapfile,dict(generated_utc=stamp,original_entrypoints=36,entry_API_counts=scope['summary']['entry_API_counts'],outside_chains=reconciliation,categories=hist,
        exact_union={k:v for k,v in union.items()if k!='additional_TASK_keys'},current_native_navigation=dict(chains=209,APIs=64476,TASKs=84297),
        source_evidence=dict(scope=dict(path=scope['full_result'],sha256=scope['result_sha256']),m20=dict(path=m20['full_result'],sha256=m20['result_sha256']),m12=dict(path=m12source['full_result'],sha256=m12source['result_sha256'])),
        limits=['209-chainnavigationextendsthe186-chainmetadata/inputselectorusing36oldEP3entryhints. Itisnotallprocess/host/graph/branchcoverage.',
            'Markov10chains4472TASKmatchv36chainIDs exactly. MoE4chains3200TASKmatchthereviewedprepare/pad-and-split sourceABI;donotreassignv20functionalbucket.',
            'FreshAPI/TASKtotalsverifiedby(database_index,rowid);nohistoricalEP3countsorIndexFillconditionpromotedtocurrent166.',
            'LegacyIndexFill[3,1]belongsoldEP3scope;current88empty-indexhostcallsandv36genericdiscardproofsupersedeitastheactivebranchquestion,notasanewcurrentnativeexecution.'],whole_module_closed=False,overall_complete=False))
    # Branch families are stable semantic scopes, not one entry per report version.
    branches=[]
    def add(i,label,state,scope_text,ev,gap,owner):
        branches.append(dict(id='M21-S'+str(i).zfill(2),branch=label,state=state,scope=scope_text,evidence=ev,gap=gap,responsible_modules=owner,whole_module_closed=False))
    add(1,'Backend/cache-kind/constructor/group dispatch and capacities','DISPATCH_CASE_BINDING_INCOMPLETE',
        'Original constructors/get_v41_cp_classes anddraftgroupinitialization;currentlegacypositivecallersversusalternativepacked/PCP/device paths.',
        ['M21-original-scope-and-static-branch-index-v41.json','M21-slot-coordinate-draft-cache-and-QLI-consumer-contract-v40.json'],
        'Needfrozenconfiguration/dispatchguardcrosswalk;currentpositivelegacybranchesalonecannotclosealloriginalconstructor/caseconditions.',['M00','M18','M21'])
    add(2,'Host metadata values/config/request-count helpers','INPUT_CONTRACT_BOUND_CPU_COST_UNMEASURED',
        'CanonicalCPUmax/requestcount/configinputs;notGPUlengthsubstitution.',
        ['M21-slot-coordinate-draft-cache-and-QLI-consumer-contract-v40.json','H02-field-and-synchronization-ledger-v13.json'],
        'CurrentEngineCore/producerCPUcostnotcaptured;separateM03scopeifrankingdependsuponit.',['M03','M04','M21'])
    add(3,'Authoritative batch snapshot andshared C2 lengths','ANALYZED_CONDITIONAL_LOCAL',
        'Target/draft/dummy batchcopy,inactivezero,longpositions andexistinglengths:c2cache.',
        ['M21-slot-coordinate-draft-cache-and-QLI-consumer-contract-v40.json'],
        'Actualsnapshot/generation/newconsumerABIarevariantconditions;globalcapturebindingisS15.',['M04','M21','M24'])
    add(4,'Per-group physical/compressed slot andpage/flat outputs','ANALYZED_CONDITIONAL_LOCAL',
        'Fused ordinary/adaptive groups,compressedphysicalparity,legacy/paddedmaskdifferences,statePADbranch andpaddingtails.',
        ['M21-slot-rejection-state-and-packing-contract-v38.json','F31-slot-fusion-domain-and-draft-cache-gap-proof-v40.json'],
        'Preserveallgroupowners/dtypes/physical+logicalcompletion. Ratio1helperextramasknotunconditionaldrop-in.',['M10','M21'])
    add(5,'CP global/local request andtoken coordinates','ANALYZED_CONDITIONAL_LOCAL',
        'CPslice/clipping/QSLaddresses/GPUlengthsuffix;worker/dummycachehitsversusdraftfreshmiss.',
        ['H02-CP-reuse-contract-v13.json','H02-CP-target-dummy-draft-publication-review-v40.json'],
        'Currentdummydead-countandcriticalpathnotinferred;actualcross-rankcapturebufferscoveredbyS15/M23.',['M11','M19','M21','M23'])
    add(6,'SWA index production andexisting count propagation','ANALYZED_CONDITIONAL_LOCAL',
        'Logicalindices/rawlength,CPindex-onlyhandoff,localcountcap/rowmap;physicalfallbackkeptseparate.',
        ['F31-existing-SWA-length-propagation-proof-v37.json'],
        'Newconsumedcountinterface/capacity/epoch/nativevalidationarevariantconditions,notmissingproducer.',['M10','M18','M21'])
    add(7,'SMLA plan publication includingempty-query writes','ANALYZED_CONDITIONAL_LOCAL',
        'SMLAarguments/fixedbank/321metadata+79positivezero callbacks;packedalternativekeptseparate.',
        ['M21-count-metadata-and-output-publication-contract-v37.json'],
        'Opaqueformat/addressbindinglimitsplanreuse;packedcaseeligibilityisS01,andcapturebindingsS15.',['M10','M21','M22'])
    add(8,'QLI plan andphysicalempty/captured reader boundaries','LOCAL_GUARDS_ANALYZED_CAPTURE_BODY_PENDING',
        'Physicalquery0/zeromaxcachebypassversuscapturedlogicalactual0;legacy/packedformatsdistinct.',
        ['F31-QLI-empty-query-and-capture-reader-review-v40.json'],
        'Actualgraphbody/parameterbankreadersmustbeboundbeforeper-replayplan-skipping.',['M09','M21','M22'])
    add(9,'C2 ring-coordinate/mask/source-RoPE publication','ANALYZED_CONDITIONAL_LOCAL',
        'Five-ringrows,used/live/skip,sourcepositions/fulltablegathers andexistingfusedbuilder;pool/index/longreaders.',
        ['M21-C2-metadata-fill-and-existing-fusion-contract-v39.json','F31-C2-existing-fused-builder-equivalence-v39.json'],
        'Fixedbanks/strides/valuegenerationconditionsremain;skipnotallzero/deadcallback.',['M08','M09','M21'])
    add(10,'Next-ID/rejection/seed/context/independentstep publication','ANALYZED_CONDITIONAL_INTERFACE',
        'Discard/backup/count-1,rejectselector,seedcontext/stepbuffers,raggedgathersandproposal-tailvalues.',
        ['M20-Markov-confidence-and-verification-contract-v36.json','M21-slot-rejection-state-and-packing-contract-v38.json','M21-positive-outside-selector-initialization-source-v41.json',gapfile],
        'Independentstepbuffersandoldread-before-writeorderingkept;actualgraph/aliaslifetimeproofisS15.',['M17','M18','M20','M21','M24'])
    add(11,'Dummy initialization andper-group PAD clearing','POSITIVE_GAP_POINT_REVIEWED_CAPTURE_EXTENT_PENDING',
        'Current16token-indexzero/16queryslotPAD/16contextslotPAD;dummyDPsyncandprofile/normalbranches.',
        ['M21-positive-outside-selector-initialization-source-v41.json',gapfile],
        'Allcapture/fullbankreadersandbankepochsneededforclearshortening;dummyreturnunusednotpermissiontoskipDPcollectives.',['M18','M19','M21','M22'])
    add(12,'Metadata producer/defer/frontier andRoPE scratch lifetime','LOCAL_PROTOCOL_ANALYZED_CALLER_BINDING_PENDING',
        'DeviceMetadataExecutorinput/reuse/wait/reset/release,_publish_taskdedup/capturedsyncandruntimeRoPEscratch.',
        ['M21-count-metadata-and-output-publication-contract-v37.json','M21-slot-coordinate-draft-cache-and-QLI-consumer-contract-v40.json'],
        'Fullmode/parameterbank/captured-producerdispatchandlastreadersneedM22review.',['M21','M22','M24'])
    add(13,'Adjacent attention/index/cache implementation branch ownership','NEIGHBOR_EVIDENCE_LINKED_GUARD_MATCH_PENDING',
        'DSAImplandCPImplprojection/indexselection/cache/forwardbranchesincludedbyfileassociation,butownedbyM07-M11/M19.',
        ['M21-original-scope-and-static-branch-index-v41.json','M19-embedding-HC-CP-row-and-consumer-contract-v35.json','M18-partition-bank-RoPE-and-slot-contract-v34.json'],
        'Afile-levelcrosslinkisnotguard-by-guardevidence;needfinitecase/neighbor-evidencecrosswalkbeforeclosingoriginalM21surface.',['M07','M08','M09','M10','M11','M19','M21'])
    add(14,'PCP alternate metadata/restore/empty-local case','PCP_DISPATCH_SCOPE_NOT_FULLY_BOUND',
        'OriginalPCPclasses,resizing,sourceRoPE,empty-localmetadata andpartitionrestoreguards.',
        ['M21-original-scope-and-static-branch-index-v41.json','M19-embedding-HC-CP-row-and-consumer-contract-v35.json'],
        'Establishrelevance/exclusionfromfixedcapturedconfiguration;activeCPclassproofdoesnotautomaticallyclosePCPalternatives.',['M00','M11','M21','M23'])
    add(15,'Graph/replay/stream/rank/iteration boundary ownership','UNANALYZED_FULL_GRAPH_BANK_AND_PHASE_BINDING',
        'M21metadata/slot/stepbanksasinputs/outputsacrosscapture,replay,requestreorderandrankpartitions.',
        ['M21-profile-chain-coverage-frontier-v40.json','M21-slot-rejection-state-and-packing-contract-v38.json'],
        'M22actualgraph/QLIbody,parameterbank/inputready/lastreaderreuse/16unboundgraphs;M23/M24fullboundaryaudit.',['M21','M22','M23','M24'])
    add(16,'Resource/cost andsame-denominator bounds','RESOURCE_BOUNDS_INCOMPLETE',
        'OriginalM21functional76943certain+80optional/529missingPMU;expanded209-chainnavigationisoverlappingmoduleassociation.',
        ['resource-coverage-summary-v20.json','module-theory-comparison-readiness-v20.json',gapfile],
        'Nohardwarepeak/causalcriticalpathfromlogicalbytes,durationsorPMUratio. Criticalsame-denominatorboundsstillincomplete.',['M21','M22'])
    add(17,'MoE prepare/padding/ID interface sharedwithM12','REUSED_M12_SOURCE_CONTRACT',
        'Originalentryhintsincludeprepare282/283/284andIDpad338;3200currentnativeTASKoutsidev37selector.',
        ['F04-precision-routing-and-row-contract-v29.json','F20-routing-ID-reuse-review-v29.json',gapfile],
        'RetainMC2padding/TPsplit/mask/ID/finalizationrows;notnewM21resourceworkornewstandalonegain.',['M12','M21','M23'])
    add(18,'Host input/output/copy/scalar/transport publication','HOST_FIELDS_ANALYZED_FULL_PROCESS_TRANSPORT_PENDING',
        '13utilitycopyfields,prepare/outsideH2D/D2H/queue/scalar,INT32sampleIDs/INT64counts andasyncoutputgeneration.',
        ['H02-field-and-synchronization-ledger-v13.json','M21-count-metadata-and-output-publication-contract-v37.json','H02-current-copy-payload-and-output-ready-review-v37.json'],
        'ActualcopyqueuebridgeandcompleteEngineCore/frontendtransport/producerCPUcostremainM03/M25.',['M03','M04','M21','M25'])
    raw=read(O/'M21-original-scope-and-static-branch-index-v41.json')
    def family(z):
        p=z['source'];owner=z['owner'];line=z['line']
        if p.endswith('/context_parallel/dsa_v41_cp.py'):
            if owner in['<module>','get_v41_cp_classes']:return 1
            if 'PCP'in owner:return 14
            if '_build_global_metadata'in owner or owner=='AscendDSAV41CPMetadataBuilder.build':return 5
            return 13
        if p.endswith('/spec_decode/dspark_proposer.py'):
            if owner.endswith('.set_inputs_first_pass'):return 10
            if owner.endswith('._clear_dummy_slot_mappings')or owner.endswith('.dummy_run'):return 11
            return 1
        assert p.endswith('/attention/dsa_v41.py'),p
        if owner in['_config_value','_request_counts']:return 2
        if 'AscendDSAV41MetadataBuilder'in owner:
            if '.build_c2_metadata'in owner:return 9
            if '.build_qli_metadata'in owner:return 8
            if '.build_smla_metadata'in owner or '.build_a5_smla_metadata'in owner:return 7
            if owner.endswith('.__init__'):return 1
            if owner.endswith('._build_batch_metadata'):return 3
            if owner.endswith('.prepare_source_rope')or owner.endswith('.defer_device_metadata')or owner.endswith('._publish_task'):return 12
            if owner.endswith('.build'):
                if line<975:return 3 if line>=955 else 1
                if line<1038:return 4
                if line<1048:return 3
                if line<1072:return 12
                if line<1124:return 6
                if line<1211:return 7
                if line<1287:return 8
                return 9
            raise AssertionError(owner)
        if owner=='AscendDSAV41Impl.__init__':return 1
        return 13
    static=[]
    for z in raw['existing_static_branch_sites']:
        k=family(z);r=branches[k-1]
        static.append(dict(z,semantic_family=r['id'],responsible_modules=r['responsible_modules'],audit_state='NAVIGATION_ONLY_SEMANTIC_GUARD_NOT_AUTOMATICALLY_PROVED',linked_evidence=r['evidence']))
    assert len(static)==268 and len({z['id']for z in static})==268
    hist2=Counter(z['semantic_family']for z in static)
    for r in branches:r['existing_static_site_count']=hist2[r['id']]
    write(O/'M21-semantic-branch-inventory-v41.json',dict(generated_utc=stamp,branches=branches,static_branch_sites=static,static_site_count=268,
        automatic_semantic_completion_forbidden=True,note='Everyoriginalstaticbranchsitehasanexplicitfamily/owner/evidencelink. Thatassignmentisnavigation,notanassertiontheguardhasbeenclosed.',whole_module_closed=False,overall_complete=False))
    original=raw['original_surface']['mandatory_gates']
    gates=[]
    links={'scope_and_dispatch':['M21-S01','M21-S14'],'all_relevant_branches':['M21-S01','M21-S13','M21-S14','M21-S15'],
        'value_and_all_consumers':['M21-S08','M21-S11','M21-S15','M21-S18'],'alias_and_state_lifetime':['M21-S10','M21-S12','M21-S15'],
        'cross_stream_rank_iteration_dependencies':['M21-S05','M21-S12','M21-S15'],'fusion_split_regroup_rules':['M21-S03','M21-S04','M21-S06','M21-S09','M21-S10','M21-S11'],
        'resource_performance_and_bounds':['M21-S16'],'evidence_and_literature_reanalysis':['M21-S01','M21-S13','M21-S16']}
    assert set(original)==set(links)
    for name,old in original.items():
        gates.append(dict(requirement=name,original_status=old['status'],result='INCOMPLETE_WORK'if name=='resource_performance_and_bounds'else'PARTIAL_PROOF',
            semantic_families=links[name],evidence=['M21-semantic-branch-inventory-v41.json',gapfile],
            completion_proved=False,note='Conditionalopportunityanalysisisnotanimplementationgate;remainingdispatch/guard/graph/process/boundsrangeisexplicit.'))
    write(O/'M21-original-requirement-completion-audit-v41.json',dict(generated_utc=stamp,requirements=gates,
        result='NOT_COMPLETE',source_scope=raw['original_surface']['source_selectors'],static_sites_assigned=268,semantic_families=18,
        native_navigation=dict(chains=209,TASKs=84297,APIs=64476),newly_point_reviewed_chains=9,newly_point_reviewed_TASKs=592,
        prioritized_next=['BindfixedconfigurationtoPCP/packed/group constructor alternatives;reconcileadjacentimplementationguardrecordswith existingM07-M11/M19analyses,notwholereporead.',
            'M22savedgraph/replay/capturemode/metadata-bankfirst-lastreader/QLIbodyaudit.','ThenM23-M25andall26modulefinitecompletionaudit.'],
        whole_module_closed=False,overall_complete=False))
    print(dict(expanded_native_chains=209,expanded_native_TASKs=84297,outside_categories=hist,semantic_families=18,static_sites=268,original_requirements=len(gates),whole_module_closed=False))


if __name__=='__main__':
    main()
