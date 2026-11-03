"""Operator-specific metadata fusion/copy/caching bounds, preserving legacy ABI fields."""
from pathlib import Path
from collections import Counter
import hashlib,math
from finalize_origin_coverage_v9 import N,O,W,read,write
from analyze_metadata_fields_v68 import family,line
S=W/'frozen-source-snapshot'

def wrap32(x):return (x+2**31)%2**32-2**31

def main():
    assert not (O/'metadata-field-fusion-costs-v68.json').exists()
    reg=read(O/'metadata-field-regime-models-v68.json')['regimes'];g=read(O/'metadata-field-native-chains-v68.json')
    bytask={tuple(k):r for r in reg for k in r['native_TASK_keys']}
    def known(row):
        vals=[bytask[(row['database_index'],t['TASK_rowid'])]['known_logical_value_bytes_excluding_unknown_control_scalars']for t in row['stages']]
        assert all(v is not None for v in vals)
        return sum(vals)
    for x in g['CP_length_groups']:assert known(x)==38*x['rows']+4
    for x in g['slot_coordinate_core_groups']:assert known(x)==140*x['rows']
    for x in g['C2_length_quotient_residual_groups']:assert known(x)==56*x['rows']
    for name in ['C2_cos_lookup_groups','C2_sin_lookup_groups']:
        for x in g[name]:assert known(x)==2056*x['rows']
    for x in g['SWA_recount_groups']:assert known(x)==7684*x['rows']
    config=N/'sampling-guard-source-v55/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/config.json'
    sha=hashlib.sha256(config.read_bytes()).hexdigest();assert sha=='43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7'
    cfg=read(config);cfg=cfg.get('text_config',cfg)
    assert set(cfg['compress_ratios'])=={0,1,2}and cfg['sliding_window']==128
    pins=[]
    for suffix,old in [('attention/dsa_v41.py',read(O/'M19-embedding-HC-CP-row-and-consumer-contract-v35.json')['source_files']),
                       ('attention/context_parallel/dsa_v41_cp.py',read(O/'M19-embedding-HC-CP-row-and-consumer-contract-v35.json')['source_files'])]:
        p=S/'vllm-ascend/vllm_ascend'/suffix;expected=old['/vllm-workspace/vllm-ascend/vllm_ascend/'+suffix]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
        pins.append(dict(path=str(p),sha256=expected,original166_hash_binding=True))
    existing_kernel=W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/triton/c2_ring_metadata.py'
    kernel_sha=read(O/'M21-C2-metadata-fill-and-existing-fusion-contract-v39.json')['source_pins']['/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/c2_ring_metadata.py']
    assert hashlib.sha256(existing_kernel.read_bytes()).hexdigest()==kernel_sha
    pins.append(dict(path=str(existing_kernel),sha256=kernel_sha,original166_hash_binding=True,existing_v39_equivalence_contract_reused=True))
    coords=[]
    for n in sorted({x['rows']for x in g['slot_coordinate_core_groups']}):
        coords.append(dict(N=n,typed_14TASK_core_known_value_bytes=140*n,three_scalar_Fill1_value_bytes=12,
            ordinary_nonempty_17TASK_known_value_reference_bytes=140*n+12,
            fused_raw_INT32_read_and_INT32_pairs_INT64_flat_write_bytes=20*n,
            interface_budget_excluding_unknown_control_scalars=120*n+12,
            required_coordinate_output_bytes=8*n,required_flat_output_bytes=8*n,
            legacy_uncompressed_or_ratio1_actual_end_mask_applies=False,
            legacy_ratio2_validity='physical-completion AND notskip AND t<min(QSL[actualB],actualT) AND (positionsabsentorodd)',
            contract='140Ncore+3scalarFill1conventionalknownvaluevaluation,notscalarNFill. Total2880Fills/lineversus2601coregroupsretainedwithoutinventedpairingof279extra. Int32pair+INT64flatABI,writelocalprefixonly/unchangedtail;sourceget_storage_block_sizeBsymbolicnotassumedconstantpower2. NewlegacyfusedABIneedsr1/uncompressed/positionsNonemode,notexistingpackedhelperdropin.'))
    cp=[]
    for b in sorted({x['rows']for x in g['CP_length_groups']}):
        cp.append(dict(B=b,source_5TASK_known_value_bytes=38*b+4,new_field_fused_per_lane_read_write_bytes=20*b,
            interface_budget_excluding_unknown_control_scalars=18*b+4,
            source_C2_quotient_residual_5TASK_known_value_bytes=56*b,new_quotient_residual_value_bytes=12*b,
            quotient_residual_interface_budget_bytes=44*b,
            combined_CP_plus_quotient_residual_source_known_value_bytes=94*b+4,
            combined_three_INT32_outputs_value_bytes=28*b,combined_interface_budget_bytes=66*b+4,
            contract='Lcp=where(QSLend>QSLstart,max(GPU_L-suffixCPU,0),0);suffixCPUqends-clamp(qends,actualstart,actualend),0innoncausal. GPUlengthauthoritativeafterrejection,notCPUupperbound. EachrequestreadsGPU_L/suffix/twoQSLINT32;new3outputsL/q/rreuseLregisteronlysamegeneration/ready/window/group. q=floor(L/2),r=L%2,preservecurrentINT32outputs. Existinglengths:c2batchcachealreadybaseline.'))
    rope=[]
    for n in sorted({x['rows']for x in g['C2_cos_lookup_groups']}):
        rope.append(dict(N=n,R=64,source_table_rows=1048576,source_FLOAT_C_or_S_table_bytes=268435456,
            source_two_3TASK_Broadcast64_Cast32_Gather_value_bytes=4112*n,
            per_row_INT64_to_INT32_cast_plus_two_row_gathers_value_bytes=1044*n,
            joint_row_lookup_same_conversion_error_value_bytes=1032*n,
            row_cast_then_two_gather_interface_budget_bytes=3068*n,
            joint_row_lookup_interface_budget_bytes=3080*n,
            required_two_FLOAT_output_banks_bytes=512*n,
            optional_skip_all_complete_false_unique_row0_table_read_bytes=512,
            contract='CurrentexpandindexN×64twiceisphysicalBroadcast/Cast/GatherElements,notzero-costview. Newrowgatherkeepnativeconversiononceperrow/exactgatherbounds-errorpolicy/source-tableversionandbothFLOATbanks(allconsumers). Sourcepos0forincompletecanreuseTABLErow0bitsexactly;donotreplacewithmathcos0/sin0orBF16M18cospolicy. Noentire268MBtablepercalltraffic.'))
    assert next(x for x in rope if x['N']==6400)['joint_row_lookup_interface_budget_bytes']==19712000
    recount=[]
    for n in sorted({x['rows']for x in g['SWA_recount_groups']}):
        recount.append(dict(local_query_rows=n,C=256,source_Cast_GE_Cast_Reduce_known_value_bytes=7684*n,
            forward_global_length_read_view_added_interface_bytes=0,
            if_new_fixed_bank_copy_needed_value_bytes=8*n,
            conditional_view_interface_budget_bytes=7684*n,conditional_new_bank_copy_interface_budget_bytes=7676*n,
            exact_count_formula='count_nonnegative=clamp(V,0,C);V=L-max(L-Q-W,0)',
            equality_condition='L>=0 AND V<=C,forexampleQ<=configuredSpec5,W128,C256',
            contract='GlobalproduceralreadyemitsINT32rawV;CPcurrentlydropslengthandlocalrecounts. Sliceidenticalorderedrowsandclamp(rawV,0,C)ifboundnotproved;forwardonlysameepoch/indexsemantics/normallogicalslotdomain. Existingphysical-slotfallbackseparate. Needlengthbanklifetimes/lastreader/input-ready/frontier/capture;newfixedbankcopyoptioncounts8N,notfreeaddressalias.'))
    ring=[]
    for b in [2,4]:
        ring.append(dict(B=b,source_header_known_vector_scalar_and_5plane_copy_value_bytes=156*b+8,
            source_5contiguous_copy_known_value_bytes=40*b,
            new_direct_header_per_lane_read_write_value_bytes=36*b,
            direct_header_interface_budget_bytes=120*b+8,
            required_INT32_ring5planes_output_bytes=20*b,
            skipTrue_additional_original_used_zero_write_bytes=4*b,
            skipTrue_new_header_without_BT_first_read_value_bytes=32*b,
            contract='Headerstarts/ends/Qlens,live=b<actualB,used=max(min(end,actualT)-start,0)theninactive/skip0;prev=max(GPU_L-Qlen,0);fiveplanes(prev,used,start,start,used>0?BTfirst:0). Five.copy_valuationsareSOURCEsemanticreads/writesoutsideComputeInfo,notinventednativeMemcpyTASK/physicaltraffic. KeepINT32wrap,copyplanes/layout,bounds/normalvalidBTshape/allinput-readbeforealiasedstores/ready/reusefrontiers.'))
    # Mathematical contracts and counterexamples, using no framework/device imports.
    for s in [-2**31,-2,-1,0,1,2,3,2**31-2,2**31-1]:
        old=s//2 if s>=0 and wrap32(s+1)%2==0 else -1
        new=s//2 if s>=0 and (s&1)==1 else -1
        assert old==new
    physical_only=(5>=0)and((5+1)%2==0);logical_odd=(4%2==1)
    assert physical_only and not logical_odd
    for bsz in [16,128]:
        for page in [0,3]:
            for pos in range(32):assert ((page*bsz+pos%bsz)%2)==(pos%2)
    # Packed helper's stronger actualEnd mask is not the legacy ratio1 law.
    legacy_ratio1_slot=5;packed_applies_rowmask=False
    assert legacy_ratio1_slot>=0 and not packed_applies_rowmask
    assert min(max(400,0),256)==256 and 400!=256
    for L in [0,1,127,128,32768]:
        for Q in range(6):
            V=L-max(L-Q-128,0);assert 0<=V<=133 and min(max(V,0),256)==V
    write(O/'metadata-field-fusion-costs-v68.json',dict(slot_coordinate_models=coords,CP_and_C2_lengths_models=cp,
        C2_cos_sin_row_lookup_models=rope,SWA_length_forward_models=recount,C2_ring_header_models=ring,
        C2_token_fields_model=dict(required_BOOL_complete_and_INT64_sourcepos_output_bytes='9*N',
            source_semantic_contiguous_complete_and_position_copy_value_bytes='18*N',
            new_semantics='complete=(t<valid_end AND pos%2==1 AND notskip);sourcepos=complete?pos-1:0',
            necessary_input_position_value_bytes='8*N',scalar_valid_end_and_skip_attrs='Knownsemanticinputs;physicalscalarGM/repeatedperCTAloadsnotpriced.',
            contract='FuseRange/mod/predicates/Sub/wherewithdirectpersistentout;preserveallsourcepos/C-S/pool/cache-update/readers/INT64errorpolicy. Cancomposenewrowlookupwithoutreloadingprivateexpandedindices;doNOTaddfullbudgetsdouble.'),
        cross_group_CSE_model=dict(source='2792CPfiveTASKgroupsobserved,nativecache-miss/hitcountnotfabricated.',
            conditional='Gcallswithsamecompletebatch/CP-window/GPUlength/QSL/inputgenerationandonecachemissneed1producer;G-1localtemporaryproducersdeadonlyallreadersusecachedbatch.',
            avoided_known_CP_value_bytes='(G-1)*(38*B+4)',
            contract='Existingv40cache-hitargumentunreadlemma;firstmiss/newglobal-localdomain/windowdifferentgroup/lastreaderfencesstilllive. Draftkwargscurrentlydropped,newconsumedinterfaceonlycompletefield-equivalence,not13groupsbyshape.'),
        reference_scope='Allbytebudgetsdeclaredsource/typedknown-valuevaluations,notphysicalHBM/L2/VectorINT64throughput/criticaltime. FlatbankINT64andpairINT32preserved. OpaqueSMLA/QLInativeplannersnotanalyzedinthissubset.',overall_complete=False))
    write(O/'metadata-fixed-parameter-reconciliation-v68.json',dict(config_path=str(config),config_sha256=sha,
        actual_compress_ratios=cfg['compress_ratios'],actual_distinct_ratios=[0,1,2],sliding_window=128,
        prior_imprecise_text='v67currentgaplistmentionedcompression0/4/32;thiswasanincorrectscopewording,notvaluesinthispinnedconfig.',
        correction='Useactual0/1/2compressionandseparatelynameanypoolwindow/tile/historydimensions. No4/32ratioinferredfromC2windoworotherdimensions.',
        prior_operator_cost_artifacts_changed=False,new_numeric_models_use_config_values=True,overall_complete=False))
    write(O/'metadata-field-operator-contract-v68.json',dict(source_files=pins,config_path=str(config),config_sha256=sha,
        local_clause_review='OnlySWA/CP/C2slots/lengths/RoPElookup/publishedbankdtypeandconsumers. 2callerfiles+existingv39fusedkernelhash/equivalencereused,3previous166pins. No framework/API/lifecycleaudit.',
        existing_C2_fused_builder=dict(path=str(existing_kernel),sha256=kernel_sha,already_implemented=True,
            prior_equivalence='F31-C2-existing-fused-builder-equivalence-v39.json',current_legacy_profiles_use_fallback=True,
            key_constraints='Existinghelperownsall5headerplanes/complete/sourcepos/C-Salready. Legacy910CconditionalreuseguardrequiresvalidINT32bounds/nonnegativepositions/sharedcos-sininputandoutputstridesoradaptseparatestrides/allbanks/frontiers. Notanewmissingkernel.'),
        facts=[
            'INT32[N,2]coordinatebankandINT64[N]flatbankareseparateoutputs;normalrawINT32slotinput+bothrequiredoutputs20Nvaluebytes.16384ViewCopybaseisbankextent,selectedNcopyonly.',
            'Legacyuncompressed/r1validitydoesNOTapplyactual-endmask. r2checksphysicalcompletedslotANDactualendANDoptionalpositionoddANDskip. Existingpackedhelperstrongermask/positionrequirementsnotdropinlegacy910CABI.',
            'Ifproducerprovesphysicalslot=blockNo*Blogical+position%BlogicalandBlogical%2==0,same-generationphysical/logicalparityagree. Withoutthisproof(s5,pos4)counterexampleinvalidatesdroppingonepredicate.',
            '2792CPfiveTASKgroupsuseauthoritativeGPUlengthminusCPquerysuffixthenclip/nonemptyselect. CPUnoncausalzeroSuffixdoesnotallowCPUlengthreplacementorremovinginactive-querymask.',
            '208C2quotient/residualgroupsalreadycachedunderlengths:c2. Fusedq/rordevicebitdiv2canavoidprivatecasts/broadcasts,notnewcacheclaim;jointCP/q/rneedscompletebatch/window/generation/readiness.',
            '104C2callbackspercos/sin3TASK eachmaterializeexpandedindicesN×64twice. Per-rowconvertedindices/jointtwo-rowlookupsavestheexpandedplaneswhilepreservingFLOATstoredvalues/nativecast-bounds-error/aliases/consumerlifetimes.',
            '69localSWA4TASKrecountsC256canuseglobalrawlengthonlyafterclamp/equalitydomainandorderedCPslice. Produceralreadystoreslengths;nativeSMLA/QLIopaqueplanformatNOTreplacedbythisproposal.',
            'C2ring5INT32planesandcompleteBOOL/sourceposINT64banksarepersistentpublishedoutputs. Somecontiguous.copy_arenotComputeInfo;sourceI/Omodelexplicitlyincludessemanticcopieswithoutinventingtraceidentities.',
            'SkipUpdategivesinvalidr2coordinate/flat,used0,completeFalse/sourcepos0,butpreviouslength/starts/ringheaderfieldsandC/Sbankreadersstillmustmatch. NoquerycannotdropglobalKV/metadatafrontiers.',
            'Sourcemetadataexecutorstage/groupwait/reuse/zero-queryfrontiersfromv37stillbaseline;newmultiout/CSE/outputaddressesmustpreserveall. Actualallgroup/cache-hit/valueequalitynotprovedbyB/shape.' ],
        witnesses=dict(physical_vs_position_parity=dict(slot=5,position=4,physical_completes=True,position_completes=False),
            legacy_vs_packed_rowmask=dict(rawslot=5,ratio=1,actualEnd=0,legacy_flat=5,packed_stronger_mask_flat=-1),
            raw_length_vs_count=dict(raw_visible=400,C=256,count=256),
            configured_safe_raw_count_domain=dict(W=128,maxQ=5,C=256,maxV=133),
            r2_parity_wrap32_checked=True),
        reused_contracts=['M21-count-metadata-and-output-publication-contract-v37.json','M21-slot-rejection-state-and-packing-contract-v38.json','M21-slot-coordinate-draft-cache-and-QLI-consumer-contract-v40.json','state-engram-operator-contract-v62.json'],
        new_remote_source_read=False,new_model_Torch_NPU_capture_health_process_implementation=False,overall_complete=False))
    vs=[]
    def v(i,c,s,b,g,d='CONDITIONAL'):
        vs.append(dict(id=i,candidate=c,choice=s,work_IO_effect=b,conditions=g,decision=d,conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,critical_path_gain_us=None))
    v('META-LEGACY-COORDINATE-FULLWRITE','F31','OnelegacykernelwritesINT32pairandINT64flat','140N+12known-valuebaseline→20N,120N+12budget','Exactr1/uncompressed/r2/Nonepositions/skip/actualEndmodes/INT32wrap/signednegative/B/error/stridedpair/prefix-tail/readiness/allreaders;notpackedhelperstrongermask.')
    v('META-PHYSICAL-COMPRESS-COORD','F31','Convertphysicalcompressedslotswhilegeneratingcoordinates','AvoidprivatecompressedINT32write/read8Ninterface','Preservephysicalcompletionforr2+logicalpositionparityunlessprovedequivalent,cachegroup/B/generation;flatoutputINT64notINT32.')
    v('META-R2-PARITY-BITS','F31','Replacevalid-slotmod2completionwithodd-bitpredicate','Sameboolforallsigned32includingAdd1wrap;avoidmod/addplanes','Actualr2/sourceINT32width,negativeinvalidandallbranchguards;integercapacity/otherreadersemanticsnotinferredfromBF16peak.')
    v('META-SLOT-POS-PARITY-REUSE','F31','Reuseonephysical/logicalparityonlyproducerinvariant','Onepredicate/optionalpositionreadbudgetconditional','Provephysical=page*Blogical+pos%Blogical,BlogicalevenandCProw/generationagreement;counterexamples5,pos4otherwise.', 'CONDITIONAL_STRONG_PRODUCER_INVARIANT')
    v('META-CONST-B-DIVMOD','F31','Sharedivmodandstrength-reduceifstorageBpower2','AvoidprivatescalarBroadcast/div/modplanes','ProveB=2^kandnonnegativephysical,nativeinput/outputwidth/rounding/prefix-tail;Bvalueunrecordednotassumed910Cspec.')
    v('META-BLOCK-SLOT-PRODUCER-OUT','F31','Extendexistingfusedgroupsproducerwithlegacytypedoutputs','Avoidrawslot/converted/coordinateintermediateinterfaces','Existing13pointeraddressarraysnotlogicalslots;pergroupmodes/strides/B/circular/CPmapping/protocolandINT32viaFLOATblock-IDpolicyv38mustmatch. Existinggroupfusionbaseline.')
    v('META-CP-LENGTH-FUSE','F31','GPUlength-suffix/clamp/nonemptyinonefieldkernel','38B+4knownbytes→20B,5TASKcandidate','GPUauthoritativeL/QSL/correctsuffix/window/noncausal/inactivequeries,allreaders/alias/ready;nofreeCPUupperboundreplacement.')
    v('META-CP-CACHEHIT-PRODUCER','F31','Avoidlocaltemporarylengthproducerwhenbatchcachehit','G-1timesknown38B+4onlyunderallreaderlemma','Firstlocal/globalmisslive,completebatch/window/valuegeneration/allcached-consumeridentityandfrontiers;Gnotcountedfromshape/13groups.')
    v('META-C2-QR-FUSE','F31','OneINT32quotient/residualproducer','56B→12Bknownvaluebytes','Currentr2andvalidlengthdomain/nativeintegerconversion;existinglengths:c2cachealreadyonce,allbanksready/lastreader.')
    v('META-CP-C2-QR-MULTIOUT','F31','EmitLlocal,q,rwhileLregisterlive','94B+4→28Bknownvaluebytes','SameGPUinput/QSL/window/batchgenerationandconsumerreadiness;differentphysicalslotgroupscanonlysharebatchfields,notallgroupbanks.')
    v('META-C2-RING-HEADER','F34','Header-onlydirectwritealternativetoexistingall-fieldfusedbuilder','156B+8sourcevaluation→36B,5copiesincludedsemantically','INT32sourcewrap/L-Qlen/used/starts/firstblock,normalBTshape/errors/alias/allreads-beforestores,stagefrontiers. Existingv39builderalreadyalsofusesheader;notnewmissingkernel.')
    v('META-C2-COMPLETE-SOURCEPOS','F34','DirectBOOLcompleteandINT64sourceposfrominputpositions','AvoidprivateRange/mod/and/Sub/whereand18Nsourcecopyvaluation','Allskip/actualEnd/nonfinite-notapplicable/negativeinteger/index-errorandpool/cache/C-Sfutureobservers;outputbanksretained.')
    v('META-C2-CS-ROW-INDEX','F31','ConvertoneindexperrowthentwoFLOATrowgathers','4112N→1044Nknownbytes,3068Nbudget','Nativeconversionandgathererrorpolicy/indexrange/exactstoredvalues,allbanks/lastreader/capture,notadvanced-indexnegativewrapreplacement.')
    v('META-C2-CS-JOINT-ROWLOOKUP','F31','Onejointcos/sinrowlookupwithconversioninline','4112N→1032Nknownbytes,3080Nbudget','Samecast/error/table/config/versionanddistinctFLOATdestinations;noBF16M18C-Sconversionorcomputingcos0frommath.')
    v('META-C2-TOKEN-CS-COMPOSE','F34','Reuseexistingv39all-fieldC2builderoutsidepacked-onlyguardconditionally','Alreadyfuses5headerplanes/complete/sourcepos/C-S;newnativecostbindingspricecurrentfallback','ValidINT32bounds/nonnegativepositiondomain,sharedcos-sinTABLE/OUTstridesoradaptseparatestrides,fullpersistentABI/oldpool-cacheallreaders/frontiers/capture;noexecuted910Ccompilationormissing-kernelclaim.')
    v('META-SKIPUPDATE-CONSTANT-OUT','F34','UnderSkipwriteinvalidslots/used0/completeFalse/sourcepos0directly','Avoidprivateslotinputs/predicates;requiredheaderandC-Soutputsremain','Normalvalidmetadata/sourceerrors,skiptrueALLaffectedstatewriters,starts/previouslen/futurebanks/lookuprow0exactbits;notdropallcompressorworkbyflag.')
    v('META-SWA-LENGTH-FORWARD','F31','ForwardglobalproducerlengthwithorderedCPslice','Recount4TASK7684N→viewor8Nnewbankcopy','Useclamp(rawV,0,C)unlessprovenV<=C;currentW128Q<=5C256sufficient,alllength/indexbankepochs/frontier/physicalfallback/lastreader/capture.')
    v('META-SWA-COUNT-UNCAPPED','F31','Substituterawvisible_lenforindexcountforallQ','raw400vscount256counterexample','OnlyconditionalQ/window/capacitydomainallowsidentity.', 'EXCLUDED_AS_UNCONDITIONAL')
    v('META-PRIVATE-SCALAR-FILL','F31','Inlineknown0/-1scalarsintofusedfieldkernel','AllFill1only4Bnot4N;279extra/linekeptunassigned','Sourcefieldconstant/readonly/currentwhereoutputs/allreaders,zero-queryeventfrontiersretained;notblanketzeroshapeelimination.')
    v('META-EMPTY-FIELD-FASTPATH','F31','Forphysicallyemptyfieldreturnemptyviewswithoutprivatefills','CanavoidscalarFillbutnotneededplanwrites/frontiers','ProveN0/allemptyfieldoutputs/bounds/errors/ready-reuse;actual0withcapturednonemptyquerymaystillneedopaqueplan.')
    v('META-FIELD-CSE-VERSION','F31','Reusepurebatch/slot/ropefieldswithcompletekeys','Valuegenerationmaintenance/metadataandbankreadinesscost','Inputvalues/actualcounts/CPwindow/skip/ratio/B/tablegeneration/allaliases/lastreader;existingbatch/slot/lengthscachesbaseline,newdraftinterfaceconsumedexplicitly.')
    v('META-PACKED-HELPER-DROPIN','F31','Replacelegacycoordinatecodewithcurrentpackedhelperunconditionally','StrongeractualEndandpositions/modecounterexample','NeednewcompatiblelegacyABIorproveallstrongermasksredundant.', 'EXCLUDED_AS_UNCONDITIONAL')
    write(O/'metadata-field-fusion-variants-v68.json',dict(variants=vs,variant_count=len(vs),interaction_rules=[
        'Coordinate/core/compressedproducer/parity/divmodchoicesoverlap;countsameintermediateonce. INT64flatbank8Nmandatorywhilepair8Nalsooutput.',
        'CPfieldfuse/cachehitCSE/CP-qrmultioutarealternativegranularities;existingcachehitsnotnew13xgain.',
        'Rowcast-two-gathersandjointC-Slookupalternativebudgets. C2complete/header/C-Scompositionmayreuseinputbutcannotaddfullindependentbudgets.',
        'SWAglobalrawlengthalreadyproduced;forwardingneedsdomainclamp/CProws/lastreader. C256countnotallrawV.',
        'SOURCEsemanticMemcpyvaluationnotexacttracejoin/physicaltraffic. NativeINT64div/mod/index/Vectorcapacityandlatencyunbound.',
        '586regimes67631execscopedconditions;18opaqueplanregimes529exec+3plan-zero-regimes79andotherM21fieldsstillpending,notcoveredbythisreport.'],whole_modules_closed=0,overall_complete=False))
    print(dict(regimes=586,exec=67631,source_pins=3,variants=len(vs),coordinate_shapes=len(coords),CP_shapes=len(cp),C2_CS_shapes=len(rope),recount_shapes=len(recount),
        C2N6400_joint_row_budget=19712000,flat_dtype='INT64',actual_compression_ratios=[0,1,2]))

if __name__=='__main__':main()
