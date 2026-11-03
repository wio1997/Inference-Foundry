"""Whole-scope retrieval/coverage crosswalk, distinct from branch-proof closure.

Every existing family is preserved. Only M13 selector/layout/operator clauses
are newly examined in this turn; no index is promoted to semantic proof.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import O,T,read,write


DOMAINS={
 'M00':('SETUP_AND_OPERATOR_SELECTION','Startup/weight preparation and implementation selection determine dtype/layout/parameter generation;outside steady-state timing is not zero work.','M21-hardware-and-DSACP-dispatch-contract-v47.json','Exact selected native/weight-layout ABI only when it changes an operator conclusion;do not inspect unrelated initialization internals.'),
 'M01':('OPERATOR_INPUT_OUTPUT_CONTRACT','Frozen request/sampling fields control full-logit readers,greedy/RNG and masks;request representation alone does not identify all runtime postprocessors.','request-guard-saved-profile-v54.json','Reconcile only request branches that change necessary operator outputs/RNG/precision;transport latency is separate.'),
 'M02':('STATE_AND_STORAGE_CONTRACT','Prefix/cache allocation and SWA/compressor-state ownership affect initial-state reuse;counter0 is not a global branch exclusion.','C2-state-opportunity-contract-v24.json','Keep rollback/allocation/physical-page and parameter generations in reusable-state legality,not framework exploration.'),
 'M03':('OPERATOR_SCHEDULING_BOUNDARY','Admission/queue/batch changes affect input generations and overlap;sampled CPU location is not a native launch or operator bound.','CPU-process-coverage-contract-v17.json','Only publication/ordering evidence needed by operator graphs;do not expand engine scheduling research.'),
 'M04':('OPERATOR_COMPUTE_AND_PUBLICATION','Input-ID/position/index updates include overwrite conflicts,sanitizeprefix and SP/CP rowplans;capacityB/N differs from live validcount.','embedding-initialization-operator-contract-v67.json','Retain whole-prefix/old-state readers and rowmap sourceguards;dataflow fusion is not arbitrary in-place parallelization.'),
 'M05':('OPERATOR_COMPUTE','Standard vocab lookup/maskedzero/SP/embeddingbank paths;TopK visionbias presence does not prove imageencoder execution.','embedding-initialization-operator-contract-v67.json','BOOL80field/readers pending;MM/inputembeds/alias and lookup-vs-communication alternatives remain explicit.'),
 'M06':('OPERATOR_COMPUTE','HC4 pre/post/premix/weightedY/residual have distinct BF16/FP32 stage order;next-layer fullresidual remains live.','HC-operator-numerical-and-dependency-contract-v59.json','All fanout/finite Vector/Cube/typed capacity and larger fusion output-readiness conditions retained.'),
 'M07':('OPERATOR_COMPUTE','Frozen V4.1 QA→QRnorm→quant→QB→partialRoPE;legacy post-QB RMS512 is excluded,including draftV41.','V41-query-operator-path-contract-v75.json','Retain exact target/draft/graphphase/localquery/globalKV and nativeprecision;do not reintroduce legacyRMS or same-shape identity.'),
 'M08':('OPERATOR_COMPUTE_AND_STATE','C1/C2 pool,RMS,Ring andkeypreRoPE branches include sparsewrites,oddstart/oldstate and tokenaligned rows.','state-engram-operator-contract-v62.json','Completion masks/page/history/rollback and precision remain;cache pointer orratio alone is not reusablevalue proof.'),
 'M09':('OPERATOR_COMPUTE_AND_STATE','QLI modes1/2/3,candidateproducer/fourconsumers,fullhead quantizers andkey/index banks retain distinct inputs.','index-context-operator-contract-v64.json','Key/head quantizer FP16scale/RINT/NaN andselectedvisibility generation;no blind candidate-restricted scan replacement.'),
 'M10':('OPERATOR_COMPUTE','SWA/CSA C1/C2/draft sparse branches;local noquery still allows KVpublication/collectives;metadataNone is a separate constantoutput domain.','attention-indexer-native-numerical-contract-v60.json','Sink/causal/selectedlength/rounding and full output/caches remain;Vector/matrix work and new splitsoftmax join must be repriced.'),
 'M11':('OPERATOR_COMPUTE','G1 WA→BF16rank1024→WB→collective,unevenvalid/global/padded rowplans;current G1 doesnot prove G>1/950packed applicability.','projection-operator-numerical-contract-v61.json','All BF16 panelrounds and A2A/RS precision/ownership;streaming/pipeline capacity andlive copies remain conditional.'),
 'M12':('OPERATOR_COMPUTE','Current nohash/biaspresent TopK andalreadyfusedMC2;explicit800→816 padding is not every M816 call.','padding-route-semantic-proof-v77.json','7MemSet1272globalpartlyhere/typed capacities/nativeparams/fullEP receiverprotocol;constanttail question conditionallyreviewed.'),
 'M13':('OPERATOR_COMPUTE','New v78:56quantselector/8controllercases,full replicatedSPweights and observed INT8→INT32→DSQ→BF16 chain;no flag-only exactTP rewrite.','shared-layout-guard-cases-v78.json','GlobalIamax/rawINT32beforedequant reductions,new ownership/ABI andtiled kernel-HCCL callbudget,readiness/alias/nativeexactness.'),
 'M14':('OPERATOR_COMPUTE_AND_HOST_MEMORY','Engram current/lookbackhash/dead/keep/UVA/hostrow-cache andgate;router sentinel mask is distinct.','state-engram-operator-contract-v62.json','Host/UVA cache generation and physicalmemory/rowtraffic unbound;distinct dead/keep/history predicates retained.'),
 'M15':('OPERATOR_COMPUTE_AND_STATE','Final HCcollapse/norm plus raw/normalized MTP consumers;rawhidden cannot disappear when confidence/aux still reads.','heads-sampling-operator-contract-v65.json','Keep fullaux/normalized/raw fanout and percheckpoint BF16rounds/rows,not just LMhead input.'),
 'M16':('OPERATOR_COMPUTE','Sharded target/draft vocabhead andfullreplicatedMarkovW2:globalID-only versusfullscores/logprob readers are separate domains.','heads-sampling-operator-contract-v65.json','Greedy summary ties/postprocessors/LoRA/vocab mapping;fullscores preserved in random/logprob domains;sharding recurrences repriced.'),
 'M17':('OPERATOR_COMPUTE_AND_RNG','Greedyacceptedprefix/reject/bonus versusrandomrecovery uses different fullV/RNG/outputsuffix obligations.','heads-sampling-operator-contract-v65.json','Fullrow rawscore regeneration/penalties/entropy and draworder;profileallgreedy doesnot close counterfactual randomdomain.'),
 'M18':('OPERATOR_COMPUTE_AND_STATE','Three ordered targetauxcheckpoints→maincontextprojection→RMS→three layerKVbanks,positions/slots andnativeC/S casts.','index-context-operator-contract-v64.json','Deferring/gatherbatching must retain distinct checkpointvalues/banks;layercacheweights/readers andacceptedprefix/slot epochs remain.'),
 'M19':('OPERATOR_COMPUTE','DraftHC/SPlocal2/3query versusglobal10/20KV,initialpadding/premix anddummyNone paths;queryshape doesnot setallmetadataN.','pad-operator-contract-v72.json','4MemSet312exactpadowners butdest/length unknown;providedmask/localinitializer/emptyquery/cachepublication semantics retained.'),
 'M20':('OPERATOR_COMPUTE_AND_RECURRENCE','FiveMarkovsteps have serialIDdependency andpostprocessedBF16 bias;confidence/globaldynamicbudget changes pruning legality.','heads-sampling-operator-contract-v65.json','Memo key/row/nativegeneration,certifiedprune numericbounds,fullRNG/logits readers andglobalbudget ordering remain.'),
 'M21':('OPERATOR_METADATA','SMLA/QLI plan algorithm/opaque output keys,slots/count/RoPE/banks differ from fullframework metadata construction.','planner-source-algorithm-contract-v71.json','Actualbankowner/phase/epoch/argument values andtyped AICPU capacities;all828observed unambiguousregimes alreadyconditionalreviewed.'),
 'M22':('OPERATOR_CAPTURE_AND_LIFETIME_CONTRACT','Graph/staticbanks/workspace andstreampublication matter only as operator observation/liveness/rewrite constraints;16graphs phaseunbound.','M22-mode-padding-and-DP-contract-v46.json','Reconcile duplicated adjacent-plane obligations into existing operator/domain records;do not treat all capture framework branches as new kernels.'),
 'M23':('OPERATOR_COMMUNICATION','14720recordedHCCLcalls include3192nonComputeInfo;count conventions/perpeer/perrank differ;physicalMC2 topology not inferred fromTP8groups.','communication-operator-contract-v66.json','Exact datatype/reduction/ownership/arrival dependencies plus actualnetworkcapacity;newsharedglobalamax/rawINT32RS domain now explicit.'),
 'M24':('OPERATOR_CROSS_ITERATION_STATE','Accepted/rejectedstate updates needoldvaluesnapshot beforeallwrites;parameter/input/history/position epochs govern crossiterationreuse.','heads-sampling-operator-contract-v65.json','State/address equality alone insufficient;allstoragewriters/rollback/requestremap and full output publication remain.'),
 'M25':('OPERATOR_OUTPUT_LIFETIME_CONTRACT','Fullscore/token/logprob/stream/terminal readers control pruning/lastuse;sourceoutputcontrol is a semantic boundary,not an NPU utilization denominator.','M25-parent-and-sentinel-contract-v54.json','Read only missing outputreader/terminal obligations that can change necessary kerneloutputs;do not extend HTTP/SSE framework analysis.')}


def main():
    assert not(O/'all-module-fixed-case-domain-audit-v78.json').exists()
    matrix=read(O/'module-phase-branch-matrix-v77.json');cov=read(O/'operator-analysis-coverage-ledger-v77.json')
    case=read(T/'cases.json')['cases']['c8-input32k-output1k-prefix90-166']
    modules=[]
    for m in matrix['modules']:
        mid=m['id'];scope,anchors,evidence,nextq=DOMAINS[mid]
        assert(O/evidence).exists()
        rr=[r for r in cov['regimes']if mid in r['module_candidates']]
        sole=[r for r in rr if r['module_candidates']==[mid]]
        pending=[r for r in rr if r['state'].startswith('REGIME_')]
        branches=[]
        for i,b in enumerate(m['branches'],1):
            branches.append(dict(record_key=b.get('id',f'{mid}-LEGACY-RECORD-{i:02d}'),canonical_semantic_ID_present='id'in b,
                branch=b['branch'],recorded_state=b['state'],recorded_gap=b.get('gap'),recorded_evidence=deepcopy(b.get('evidence')),
                this_audit_is_new_per_branch_semantics_proof=False))
        modules.append(dict(module_id=mid,operator_scope=scope,input_branch_schema=m.get('branch_schema','CHRONOLOGICAL_EVIDENCE_RECORDS'),
            input_family_record_count=len(branches),existing_families=branches,fixed_case_applicability_anchors=anchors,primary_existing_evidence=evidence,
            newly_rechecked_operator_guard_domain=mid=='M13',next_operator_evidence_question=nextq,
            current_observed_regime_candidates=len(rr),current_observed_execution_candidates=sum(r['executions']for r in rr),
            unambiguous_regimes=len(sole),unambiguous_executions=sum(r['executions']for r in sole),
            unresolved_operand_regime_candidates=len(pending),unresolved_operand_execution_candidates=sum(r['executions']for r in pending),
            typed_operator_counts=dict(Counter(r['operator']for r in sole)),
            observed_all_branches=False,whole_module_closed=False))
    assert len(modules)==26 and sum(m['input_family_record_count']for m in modules)==335
    assert sum(m['input_branch_schema']=='SEMANTIC_BRANCHES_V1'for m in modules)==22
    assert [m['module_id']for m in modules if m['input_branch_schema']!='SEMANTIC_BRANCHES_V1']==['M00','M02','M03','M13']
    write(O/'all-module-fixed-case-domain-audit-v78.json',dict(source_matrix='module-phase-branch-matrix-v77.json',source_matrix_sha256=hashlib.sha256((O/'module-phase-branch-matrix-v77.json').read_bytes()).hexdigest(),
        fixed_case=case,modules=modules,module_count=26,input_family_record_count=335,input_semantic_modules=22,
        legacy_chronological_modules=['M00','M02','M03','M13'],newly_rechecked_operator_guard_modules=['M13'],
        all_existing_records_preserved=True,all_model_guard_conditions_newly_proved=False,
        arithmetic_note='Per-module candidate-attributed counts overlap when module_candidates has multiple entries;do not sum them as a new execution denominator. 9pendingoperandregimes are not all remainingfusion/ceilingquestions.',
        audit_findings=['335records do not mean335disjoint operatorbranches orproofs.22semanticmodules/4chronologicalmodules;M13hasoperatoranalysisevidence but lackedsemanticfamilyreconciliation.',
            'Currenttyped42M13regimes22360executions alreadyconditionalreviewed in v57;no regimepromotion or newnativeexecution claimed. Newsourceguards/modelconstruction+precision proof changes legal TP-sharding comparison.',
            'Control/setup/transport modules are retained as input/output/lifetime/selection constraints. Their unmeasured CPUinternals are not additional NPUkernels and are not a reason to research the entire framework.',
            'M22adjacent-plane repeated obligations should reference actual operator/state records;their table wording does not show missing operatorsemantics by itself.',
            'PhysicalVector/nonlinear/Sort/typedmemory/AICPU/network capacities and compatible composedgraph lowerbounds remain incomplete. Existingroof references do not establish moduleMFU ranking.'],
        next_work=[dict(id='M22-ADJACENT-OPERATOR-CONTRACT-DEDUP',state='OPEN',question='Reconcile M22S06–S09/S12–S13 adjacentstate obligations with currentoperatorfamilies;identify which exact input/output/storage generation evidence is genuinely missing,without broad frameworkread.'),
                   dict(id='M02-STATE-STORAGE-SCOPE-RECONCILIATION',state='OPEN',question='Convert chronologicalcacheallocation/state evidence to operatorinitial-state/physicalstorage/rollback families with existingC2/Engram/selectedindex contracts;do not infer half-capacity or warmreuse fromratios.'),
                   dict(id='TYPED-CEILING-AND-MODULE-RANKING',state='OPEN',question='Use primaryhardware/SDK operator evidence to bind typed capacities or preserve explicitsymbolic bounds. Reconcile existing module-resource denominators;do not rank mixed peak-reference fractions as MFU.'),
                   dict(id='RELATION-AND-HIGHER-ORDER',state='OPEN',question='Reconcile remaining550centralpairslots with earliercandidategraphclause evidence. Need actualcomposition/dependency/resource alternatives,notoneemptytableentryperturn.')],
        global_uncertainty=cov['unchanged_global_uncertainty'],whole_modules_closed=0,overall_complete=False))
    print(dict(modules=26,input_records=335,semantic_modules=22,chronological_modules=['M00','M02','M03','M13'],
        newly_examined_guard_module='M13',all_model_guard_conditions_proved=False,whole_modules_closed=0))


if __name__=='__main__':main()
