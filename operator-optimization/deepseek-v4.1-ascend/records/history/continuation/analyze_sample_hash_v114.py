"""Selected-token/history factorization for Engram; exact CPU/source analysis only."""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib,json,random
from finalize_origin_coverage_v9 import N,O,W,read,write

COMMON=W/'frozen-source-snapshot/vllm/vllm/models/deepseek_v41/common/engram.py'
ASC=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/hash_state.py'
HASH=N/'state-engram-operator-source-v62/vllm_ascend/ops/triton/engram_hash.py'
M=(1<<63)-1

def pin(p):
    b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def part(p,a,b):return dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b])
def baseline(codes,blocks,mults,pad):
    result=[]
    for row in mults:
        rolling=0;blocked=False;out=[]
        for d in range(4):
            blocked|=blocks[d];value=pad if blocked else codes[d]
            rolling^=value*row[d]
            if d:out.append(rolling)
        result.append(out)
    return result
def history_summary(history,blocks,mults,pad):
    result=[]
    for row in mults:
        rolling=0;blocked=False;out=[]
        for d in range(1,4):
            blocked|=blocks[d-1]
            rolling^=(pad if blocked else history[d-1])*row[d];out.append(rolling)
        result.append(out)
    return result
def finalize(current,blocked,summary,mults,pad):
    if blocked:return baseline([pad]*4,[False]*4,mults,pad)
    return [[current*row[0]^v for v in hh]for row,hh in zip(mults,summary)]

def reference():
    rng=random.Random(114);cases=0;ids=0;max_product=0;dig=hashlib.sha256()
    for v in [1,2,3,17,129280,1<<31]:
        bound=max(1,(M//v)//2)
        choices=[1,2*bound-1]
        for trial in range(128):
            mults=[[choices[(trial+i+d)%2]if trial<4 else 2*rng.randrange(bound)+1 for d in range(4)]for i in range(2)]
            codes=[rng.randrange(v)for _ in range(4)];pad=rng.randrange(v)
            h=history_summary(codes[1:],[False]*3,mults,pad)
            for mask in range(16):
                blocks=[bool(mask&(1<<d))for d in range(4)]
                summary=history_summary(codes[1:],blocks[1:],mults,pad)
                expected=baseline(codes,blocks,mults,pad)
                got=finalize(codes[0],blocks[0],summary,mults,pad)
                assert got==expected and all(0<=x<=M for row in got for x in row)
                for row in mults:
                    for code,mm in zip(codes,row):max_product=max(max_product,code*mm);assert code*mm<=M
                for li,row in enumerate(got):
                    for d,x in enumerate(row):
                        for head in range(8):
                            # Equality is checked before remainder;anysame positive divisor/offset is preserved.
                            prime=1000003+2*(li*24+d*8+head);offset=(d*8+head)*2000000
                            assert x%prime+offset==expected[li][d]%prime+offset;ids+=1
                cases+=1;dig.update(repr((v,codes,mask,mults,got)).encode())
    return dict(cases=cases,hash_ID_comparisons=ids,fixture_sha256=dig.hexdigest(),max_product_observed=max_product,
        vocab_domains=[1,2,3,17,129280,1<<31],all16_current_history_block_patterns=True,
        CPU_integer_reference_not_native_execution=True)

def main():
    assert not(O/'sample-hash-inputs-v114.json').exists()
    paths=[O/'REVIEW-CHECKPOINT-v113.json',O/'coverage-current-obligations-v113.json',O/'candidate-relation-clauses-v113.json',
        O/'candidate-pair-navigation-v113.json',O/'state-engram-operator-contract-v62.json',O/'state-engram-regime-models-v62.json',
        O/'state-engram-fusion-variants-v62.json',O/'Engram-hash-lookup-publication-and-gate-contract-v30.json',
        O/'Engram-hash-UVA-and-gate-costs-v62.json',O/'recovery-fallback-source-and-applicability-v101.json',
        O/'penalty-state-and-composition-v110.json',COMMON,ASC,HASH]
    parts=[part(COMMON,94,165),part(COMMON,167,245),part(COMMON,382,462),part(ASC,83,172),part(HASH,86,167)]
    write(O/'sample-hash-inputs-v114.json',dict(inputs=[pin(p)for p in paths],
        operator_questions=['What normal constructor domain excludes INT64overflow andnegative remainder?',
            'What exact history state can be prepared before finaltoken selection?',
            'How can F35publish orselect that state without committing rejected alternatives or multiplyingcrosscard/UVAwork?'],
        no_new_remote_source_or_device_work=True,overall_complete=False))
    rows=read(O/'state-engram-regime-models-v62.json')['regimes']
    names=['_engram_req_index_kernel','_write_hash_cache_kernel','_hash_ids_kernel','_engram_host_uva_gather_dequant_kernel']
    selected=[r for r in rows if r['operator']in names]
    native=[dict(regime_id=r['regime_id'],operator=r['operator'],executions=r['executions'],signature=r['signature'],TASK_keys=r['native_TASK_keys'])for r in selected]
    counts=Counter()
    for r in native:counts[r['operator']]+=r['executions']
    assert dict(counts)==dict(zip(names,[88,88,88,176]))
    write(O/'sample-hash-profile-scope-v114.json',dict(existing_regimes=native,existing_execution_counts=dict(counts),
        prior_v30_triples_and_shared_alllayer_hash_reused=True,new_native_attribution_promotions=0,
        observed_tokenmap_descriptor_rows=129280,descriptor_rows_are_not_compressed_vocab_size=True,
        legacy_F24_template_CPU_resource_labels_not_current_selected_execution_resources=True,
        selected_hash_and_UVAlookup_are_accelerator_TASKs_not_CPUzero_time=True,
        actual_crossiteration_request_suffix_or_parameter_values_bound=False,overall_complete=False))
    write(O/'sample-hash-source-contract-v114.json',dict(parts=parts,facts=[
        'Default token_map construction produces codes0..V-1 andassertsVmatcheslayout. pad_id isone suchmappedcode;DEAD_ID=-1. Cacheinitis0 andnormalcachewriterstores onlymappedcodes orDEAD.',
        'Default multiplierbuilderuses B=max(1,floor(floor((2^63-1)/V)/2)),rng u∈[0,B),m=2u+1. Active registeredmultiplier contents/overrides are not captured here.',
        'Ascend hash resolves in_batch first,then knownnonnegative windowtoken,then optionalslotcache/defaultpad. Unknownwindow -1 isnot DEAD;negativeposition orDEAD source blocks that andallolderngram terms.',
        'Cachewrite precedes hashfallback reads;earlyhistoryreuse mustmatch theresolved effective history at the eventualcall,includingalias/read/write ordering,not only the physicaladdress.',
        'One hashcall alreadycovers bothlayers andreqsearch isalreadyoncepertoken. The source documentscurrentbackendfailure when directlymerging reqsearch/hash;no suchretryornewkernel isperformed.',
        'F35emitted prefix includesaccepted proposed IDs andthe recovered/bonus token underitsactualbranch. Dense outputsentinels/unaccepted proposals arenot validrawtoken_map indices. Existing penalty/statecommit semantics are reused.',
        'LegacyF24 templateCPU labels do not classify the selectednative hash/UVA tasks. Hostmapped lookup data andaccelerator compute/storage/transport remain different resource domains.'],
        loaded_initializer_parameter_contents_or_actual_state_epochs_attested=False,overall_complete=False))
    write(O/'Engram-normal-INT64-range-proof-v114.json',dict(status='CONDITIONAL_CEILING',
        domain='Defaultsourceconstructor,1<=V<=2^31,validraw tokenIDs andmappedcodes,DEADmaskedbeforemultiply,normalcacheinitialization andonlyspecifiedwriters;registeredbuffers notoverridden.',
        derivation=['M=2^63-1;B=floor(floor(M/V)/2)>=1 inthisdomain;m=2u+1<=2B-1<=floor(M/V)-1.',
            '0<=code<=V-1 implies0<=code*m< V*floor(M/V)<=M;pad isalso avalidcode.',
            'Each multipliedterm hasbits0..62 only. XOR ofsuchterms retainsbits0..62,so0<=R<=M.',
            'Thus signed/unsignedpositive-divisor remainder agree onR;no overflowwrap premise isneeded insidethisnormaldomain. Genericbufferoverride/outofrange/corruption cases remain outsideproof.'],
        cache_invariant='Initial0 liesin[0,V-1];validwriter stores token_map[id]orDEAD. A blockedDEAD isreplacedbypad beforemultiply. Unresolvednegativewindow entries takefallback,not directnegative multiply.',
        actual_native_param_values_bound=False,division_or_mul_capacity_upper_bound_proved=False,
        whole_hash_or_MFU_closed=False,overall_complete=False))
    write(O/'Engram-history-factorization-v114.json',dict(
        domain='Two sourcehashlayers,ngramlengths2/3/4,eightheads each;exactresolvedmappedhistory/blockflags,unchangedmultipliers/primes/offsets andnumericdomain.',
        definitions=dict(current='x isvalidselectedcompressed currentcode;bc is currentDEAD/negative-position predicate',
            history='z_d,d=1..3 areeffectivehistorycodes; bh_d is prefix OR of historyDEAD/negativeposition throughd',
            H='H_l,s = XOR_{d=1..s} ((bh_d?pad:z_d)*m_l,d),s=1..3',
            P='P_l,s = XOR_{d=0..s}(pad*m_l,d),s=1..3'),
        final=dict(current_alive='R_l,s=(x*m_l,0) XOR H_l,s',current_blocked='R_l,s=P_l,s',
            hash_ID='Keep unchanged R_l,s % prime_l,s,h + offset_l,s,h,then sameINT32store.'),
        correctness='Forbcfalse,the sourceblockedprefixfromd1onwards equalshistoryprefixOR. XORassociativity splits the currentterm fromthat exacthistoryprefix. Forbctrueallsourcevalues arepad,independentofhistory.',
        history_packet_bytes=48,hash_ID_output_bytes=192,
        history_packet_scope='SixINT64values foroneposition/request/history/parameter epoch,plusowner/position/validity/state tags. Not a universalminimalstatistic.',
        early_preparation_premise='Resolvedhistory mustbeindependentoftheunselectedcurrentID/deadpredicate andstablethroughbaselinefallbackread. Ifcurrentcachewritescanchangeitsfallbacksources,usecandidate-specificH orretainlatecomputation.',
        rejected_or_bonus_ID_handling='Choose the correct accepted-prefix history beforethe recovered/bonus ID. Do not reuse a summary keyedonlybyscheduler row,slotaddress ornumberaccepted.',
        prior_hash_outputs='Already-reduced INT32hash IDs are not automatically an Hpacket. Retainingpre-remainder R/H requires newproduceroutput/readiness/lifetime ABI.',
        allblocked_current_constant='The48P%prime+offset IDs areconstantperparameter/padgeneration. This doesnotdelete lookup,deadmask publication orcachehistorywrites.',
        actual_prefix_to_nextcall_bindings_proved=False,overall_complete=False))
    validation=reference();witnesses=[]
    mm=[[1,3,5,7],[9,11,13,15]];hist=[2,3,4];pad=0
    wrong_hist=baseline([1]+hist,[True,False,False,False],mm,pad)
    wrong=[[v^(1*row[0])for v in vals]for row,vals in zip(mm,wrong_hist)]
    correct=history_summary(hist,[False]*3,mm,pad)
    assert wrong!=correct
    witnesses.append(dict(id='CANDIDATE_DEAD_BREAKS_XOR_OUT',proposal_current=1,proposal_blocked=True,history=hist,
        proposed_R=wrong_hist,xor_out_proposal_gives=wrong,true_H_for_alive_recovery=correct,
        consequence='A deadproposalRusedpadforallolderterms;removingitscurrentproductcannotrecoveralivehistory. Retain avalidHpacket orrebuild exacthistory.'))
    collision={};found=None
    for suffix in product(range(8),repeat=3):
        h=suffix[0]*3^suffix[1]*5^suffix[2]*7;after=suffix[0]*5^suffix[1]*7
        if h in collision and collision[h][1]!=after:
            found=dict(a=list(collision[h][0]),b=list(suffix),same_longest_H=h,next_after_zero_a=collision[h][1],next_after_zero_b=after);break
        collision[h]=(suffix,after)
    assert found
    witnesses.append(dict(id='ONE_OLD_LONGEST_ACCUMULATOR_NOT_SHIFT_STATE',**found,
        consequence='Independent lookbackmultipliers do notpermit a generic one-accumulator XOR-out/XOR-in shift. This doesnotclaimall48hashIDsareuninvertible.'))
    witnesses.extend([dict(id='UNKNOWN_WINDOW_NOT_DEAD',window_raw=-1,known=False,fallback_code=7,
        consequence='Treatingunknownwindowasblockedchangesngram;resolve fallback atcorrectepoch.'),
        dict(id='PROPOSAL_NOT_COMMIT',proposals=[10,11,12],accepted_count=1,recovered=20,
            committed_suffix=[10,20],wrong_committed_suffix=[10,11,12],consequence='Use actualemitted sequence;speculative alternativesstayprivate.'),
        dict(id='SENTINEL_NOT_RAW_TOKEN',physical_output=[10,-1,-1],valid_emitted_count=1,
            consequence='Do notindex token_map withpadding/unacceptedplaceholder. EngramDEADis aresolvedhistorysentinel,notpermissionforrawnegativeindexing.'),
        dict(id='FALLBACK_CACHE_EPOCH',slot=7,before_code=2,after_current_writer_code=9,
            consequence='Aprecomputedsummarybasedonbeforecodecannotreplacebaselinefallbackreadingaftercurrentwrite withoutreadonlysnapshot/aliasproof.')])
    write(O/'sample-hash-integer-validation-v114.json',dict(reference=validation,invalid_rewrites=witnesses,
        no_native_compiler_or_device_test=True,overall_complete=False))
    prefix=[]
    for d in [0,1,5,16]:
        alternatives=d+1
        prefix.append(dict(proposed_depth=d,prefix_alternatives=alternatives,
            Hbank_payload_bytes=48*alternatives,
            independently_prepared_history_mul_count=6*alternatives,history_nontrivial_XOR_count=4*alternatives,
            selected_finalize_mul_count=2,selected_finalize_XOR_count=6,selected_remainder_count=48,
            baseline_single_position_mul_count=8,baseline_nontrivial_XOR_count=6,baseline_remainder_count=48,
            total_extra_mul_vs_one_late_fullhash=6*d,total_extra_XOR_vs_one_late_fullhash=4*d+4,
            reference_only_not_expected_work_or_gain=True))
    lookup=read(O/'Engram-hash-UVA-and-gate-costs-v62.json')['lookup_model']['formats']
    write(O/'sample-hash-joint-work-and-capacity-v114.json',dict(status='SYMBOLIC_CEILING',prefix_alternative_models=prefix,
        retained_target_packet_case='Acurrentalivehashbody couldexposeH fromits pre-remainderR andkept currentproduct:6additionalXOR+48Bpacketstore perretainedposition,pluslifetime/tags/publication. Deadcurrentcannotuse thatshortcut. Last allaccepted/bonusposition maynotexistinoldhashoutput;actualposition maprequired.',
        selected_tail='GivenvalidprecomputedH forselectedhistory,2currentterm multiplies+6XOR+48remainders;originalbaselinealreadyhashesbothlayers inonecall. EarlierHconstruction isnotfree andremaindersarenotremoved.',
        allblocked_current='Per-generation constant48IDpacketcanreplacearithmetic atblockedcurrent,butfullhashoutput/lookup/cache/mask observers remain. No workloadhitfraction isknown.',
        transport=[dict(raw_ID_bytes=4,full_hash_packet_bytes=192,payload_ratio=48),dict(raw_ID_bytes=8,full_hash_packet_bytes=192,payload_ratio=24)],
        cross_card='Compare ownercomputedhashpublication withreceiverlocalhashafteractual4/8BIDfanout. IfrawIDfanoutremainsneededbyembedding/otherreaders,the192Bhashpacketisadditional,not a replacement. Only anexplicitclosedreceiverABIcanreplaceIDpublication;ratiosarenotspeedups.',
        cold_local_lookup_payload_per_token=dict(BF16=6*512,INT8_GROUP32=6*(256+32),MXFP8_GROUP32=6*(256+8)),
        lookup_scope='Two layers×3localheads×256width inoldTP8reference. Actualformat/ownership/uniquehead-ID/weight-scaleepochs knownbeforeanypayloadbound;logicalcoldhostmappedpayload,notHBM orphysicalwirebytes.',
        speculative_lookup='ForAownedlogicalprefetches andUdistinctcoldkeys(layer,head,hashID,table/scalepoch),encodedreadpayloadreferenceisU*formatbytes. Compareactualdemandkeys/waste andimmutablebanklifetimes;prefetchingmanyalternatives can increasework.',
        bound='Forselecteddeclaredinstructionwork,T>=max(W_mul/U_mul,W_XOR/U_XOR,W_mod/U_mod,B_UVA/U_UVA,B_publish/U_publish,othercuts),onlywhenmatchingvalidcapacityupperbounds. No INTEGER/CPU/UVA/collective capacities establishedhere.',
        critical_path='Movinghistorybefore selectiondoesnotproveoverlap;sharedAIV/scalarresources,launches,banks andnextcallreadinessmustbepriced. Producer-sideF35fusionmustnotdelayremaininghead/rejection outputs.',
        no_MFU_physical_numeric_ceiling_or_speedup_proved=True,overall_complete=False))
    write(O/'sample-hash-state-and-publication-v114.json',dict(
        permitted_graphs=['AfterfinalvalidID/ownerselection,preparehashpacketwithselectedhistoryandpublishfornext matchingEngramcall.',
            'BuildprivateH_k candidatesforacceptedprefixalternatives,thenchooseoneafterrejection andcomputeonlyactualvalidterminalIDs.',
            'Readonlylookup/prefetch ofalreadyknowncandidate hashes underimmutabletableepoch;commitonlyactualemittedtokens andmatchingnextcallmetadata.'],
        demand_key=['requestidentity/sequenceepoch','actualposition andorderedmappedlast3 history/deadflags','tokenmap/multiplier/prime/offset/pad generation',
            'validemitted-count andaccepted/recovered/bonusbranch','normal/sourcepriority/cache snapshotepoch','DP/TProwowner andnextslot/group/COWgeneration','lookuphead/table/scaleepoch'],
        commit='Preserve rawemittedID outputs andrequiredhost/user/embedding/penalty/logprob readers. PrivatespeculativeH/hash/lookupbuffers are notcanonical slot-cache writes. Keep originalnormalcachewrite effects/order unless newallreader/alias/state proof.',
        empty_no_draft_and_failure='K0/emptyemittedbranches perform no newdemandwork. Scheduled/proposed IDsnotemitted cannot updatecanonicalhistory. Paddedunknown/sentinel branchremainsdistinctfrom DEADhistory.',
        original_history_write='Currenthashcachewriter canincludeunacceptedtargetproposalrows,thenrewritethem afterrollback peroriginalslotprotocol. The newlogicalcommittedsuffix helper mustnot pretendthatbaselinephysicalprotocol neverwritesproposals.',
        fields_not_in_profile=['actualacceptedlength/outputvalidity','exactnextscheduledinputpositionmap','resolvedhistory/window/cachecontents','registeredparameterarrays','rowowners/epochs/allreaders','UVAready/CPUlookupcriticalpath'],
        numeric_and_state_legality_proved_for_actual_fixedload=False,overall_complete=False))
    evidence=['sample-hash-profile-scope-v114.json','sample-hash-source-contract-v114.json','Engram-normal-INT64-range-proof-v114.json',
        'Engram-history-factorization-v114.json','sample-hash-integer-validation-v114.json','sample-hash-joint-work-and-capacity-v114.json','sample-hash-state-and-publication-v114.json']
    write(O/'sample-hash-composition-v114.json',dict(clause=dict(id='REL114-01',a='F24',b='F35',
        relation='VALID_SELECTED_TOKEN_AND_PREFIX_HISTORY_TO_NEXT_HASH_LOOKUP',
        operator_claim='Precomputeorretain exacthistory H fornext matchingposition,chooseactualaccepted-prefix/recovered-IDbranch,thenfinalize hash andoptionallyprefetch;keepcanonicalcachecommit separate andpricecrossrankpublication.',
        applicable_domain='Same actualresolvedhistory/parameter/request/row/position epoch;validselectedrawID/deadpredicate andunchangedINT64/remainderpolicy. Allnormalcachewrites/readers,DP/TPpublication andextraH/lookup workpriced.',
        evidence=evidence,conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None),
        group=dict(id='HG114-SELECT-HISTORY-HASH',members=['F24','F31','F35'],
            contract='Choose onevalidnextcall ID/history/owner/slot epoch. Prefixalternatives remainprivate;preserveactualemittedrawIDs andcanonicalcacheprotocol. PriceHretention,192Bhashvs4/8BIDtransport,lookupwaste andallpublication once.',
            scope_state='SCOPED_COMPOSITION_CONSTRAINTS_RECORDED;ALL_VARIANTS_AND_HIGHER_ORDER_CLOSURE_OPEN',all_subsets_analyzed=False,legal_joint_fusion_proved=False),
        new_pairs=1,new_groups=1,new_main_candidates=0,whole_pair_all_variants_closed=False,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',CPU_factor_cases=validation['cases'],hash_ID_comparisons=validation['hash_ID_comparisons'],
        reused_hash_executions=88,reused_lookup_executions=176,new_pairs=1,new_groups=1,new_native_promotions=0,new_device_work=False,overall_complete=False)))

if __name__=='__main__':main()
