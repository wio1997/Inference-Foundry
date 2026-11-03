from copy import deepcopy
from datetime import datetime, timezone
from finalize_origin_coverage_v9 import N, O, read, write
from finalize_Engram_whole_frontier_v30 import packet, graph
from persist_profile_refinement_v2 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v36.json').exists()
    stamp = datetime.now(timezone.utc).isoformat()
    specs = [
        ('profile','pass','M20-profile166-v36','query_M20_confidence_and_steps_v36_166.py',False,
         'All32 saved fullcallers/typed TASKs/COMM for merged control/head/Markov and next-ID boundary; heavy decoder/context scopes excluded.'),
        ('review','error','M20-source166-v36','read_M20_confidence_and_next_ID_v36_166.py',False,
         'Initial reader stopped on a too-small merged-tail span guard; rc1 retained, no result/target mutation.'),
        ('review','pass','M20-source166-v36-v2','read_M20_confidence_and_next_ID_v36_166_v2.py',True,
         'Actual merged end/returns/caller tail and next-ID method captured; source tail178lines is not the whole327-line merged method.'),
        ('profile','pass','M20-step-audit166-v36','audit_M20_steps_and_next_ID_v36_166.py',False,
         '104 complete five-step source-order patterns,520 exact11-TASK steps and88next-ID surfaces; no unmatched step TASK.'),
        ('review','pass','M20-scheduler166-v36','read_M20_dynamic_scheduler_and_handoff_v36_166.py',True,
         'Exact imported DynamicSpecScheduler update/closed direct self-call graph, model confidence wrappers and runner next-ID/count consumers; no model imports.'),
        ('review','pass','M20-publication166-v36','read_M20_confidence_head_and_budget_publication_v36_166.py',True,
         'Exact FP32 confidence head, seed-inclusive Markov relation and worker asynchronous count-copy/event protocol pinned.'),
        ('review','pass','M20-verify-consumer166-v36','read_M20_verify_prefix_consumer_v36_166.py',True,
         'take_draft_token_ids2205-2226 trims already-generated output lists with per-request device lengths; not generation-time early exit.'),
        ('profile','pass','M20-exact-profile166-v36','audit_M20_exact_dynamic_presence_v36_166.py',False,
         'All127920 saved native API records queried at exact scheduler/confidence/verify-consumer/backup-clone source sites:0ComputeInfo-linked matches, not false flags/no transfers.'),
    ]
    packets={}
    for _,status,label,_,embedded,_ in specs:
        if status=='error':assert read(N/(label+'.json'))['returncode']==1
        else:packets[label]=packet(label,embedded)
    p=packets['M20-profile166-v36'];a=packets['M20-step-audit166-v36'];ep=packets['M20-exact-profile166-v36']
    ps,au=p['summary'],a['summary']
    assert (ps['native_APIs'],ps['native_TASKs'],ps['native_chains'],ps['source_bound_collectives'])==(4464,7520,28,104)
    assert (au['complete_five_step_patterns'],au['steps'],au['pattern_issues'],au['step_pattern_match_count'],au['unassigned_step_TASKs'])==(104,520,0,520,0)
    assert (au['covered_seed_and_step_TASKs'],au['outside_step_pattern_TASKs'],au['next_ID_TASKs'])==(5824,1696,1144)
    assert ep['summary']['all_native_API_records_checked']==127920 and ep['summary']['matched_APIs']==0
    refs=[dict(label=k,receipt=str(N/(k+'.json')),full_result=v['full_result'],sha256=v['result_sha256'])for k,v in packets.items()]
    source_files={}
    for _,status,label,_,embedded,_ in specs:
        if status!='pass' or not embedded:continue
        for f in packets[label]['result']['files']:
            prev=source_files.setdefault(f['path'],f['sha256']);assert prev==f['sha256']

    profile_file='M20-current-Markov-next-ID-and-dynamic-visibility-v36.json'
    contract_file='M20-Markov-confidence-and-verification-contract-v36.json'
    next_file='F31-next-ID-and-acceptance-summary-review-v36.json'
    budget_file='F30-dynamic-verify-budget-and-prefix-bounds-v36.json'
    framework_file='H02-H06-verification-length-publication-review-v36.json'
    variants_file='M20-variants-and-condition-ledger-v36.json'
    branches_file='M20-phase-and-branch-ledger-v36.json'
    graphs_file='M20-source-and-conditional-budget-graphs-v36.json'
    write(O/profile_file,dict(generated_utc=stamp,evidence_grade='MEASURED_SAVED_PROFILE',profile=ps,source_order_audit=au,
        exact_dynamic_presence=ep['summary'],native_chain_headers=p['native_chain_headers'],evidence_refs=refs,
        scope_notes=[
            '7520TASK=5824seed/step+552base-index/head/reconstruction+1144next-ID helper. This is an ancestry boundary selector, not a replacement of v20 M20 functional PMU=5304.',
            '104patterns are16dummy B4,32proposal B2,56proposal B4. Each has five11-TASK steps. Source-order validation rejects unfinished/extra steps; no request/pointer/root epoch pairing.',
            '520steps=440proposal+80dummy; each has4Slice,1GatherV2,1MatMulV2,1Add,2ViewCopy,1ArgMaxV2,1Cast. Positive greedy branch does not establish target all_greedy or all future configs.',
            '88next-ID helpers have13ComputeInfoTASK each:2NotEqual,4Cast,and one each Less/LogicalAnd/ReduceSum/Sub/Clip/GatherElements/Select. Shapes[B,K] are[2,1]x16,[2,6]x16,[4,6]x56.',
            '104complete head AllGather COMM records differ from32ComputeInfo hcom tasks. All selected tasks/COMM have no unique target prepare-root binding; no ordinal/nearest-time repair.',
            'Confidence-name and exact source-location queries have0nativeComputeInfo-linked matches. No runtime dynamic flag or absent CPU/memcpy/D2H/synchronization is inferred.',
            'Backup copy2217, clone/discard2222-2228 and int(k) host consumption are source facts outside this ComputeInfo census; do not invent88 measured copies orB measured synchronization calls.'
        ],whole_module_closed=False,overall_complete=False))

    facts=[
        ('DOCUMENTED_PLUS_MEASURED','DSpark1584 writes the known seed to T[:,0]. Step i uses markov_embed(T[:,i]), Markov bias, BF16 in-place base+bias add, greedy argmax/remap and copies prediction into T[:,i+1]. Observed five steps do not mean the seed is a discarded prediction. Early-return DSpark1637/1643 returns T[:,1:].'),
        ('DOCUMENTED','Merged1454-1780 also contains probability/draft_probs, selector, non-DSpark and autoregressive-continuation branches. Current trace positives bind the greedy Markov path; actual parallel/dynamic configs remain unrecorded. _propose also has a num_speculative_tokens0 return. Do not promote count/absence to all flags.'),
        ('DOCUMENTED','DSpark1611-1617 calls dynamic_spec.update after the Markov loop with model,last_hidden_states,draft_token_ids,num_reqs. DynamicSpecScheduler in spec_decode/utils.py is the actual imported receiver; method-name-only search for confidence missed receiver.update.'),
        ('DOCUMENTED','DSpark confidence uses raw last_hidden_states.reshape(B*D,H), and model.markov_embed(T[:B,:D]) including the seed and previous predictions. This is the same ordered embedding-input sequence as the loop under one parameter/input generation, not the just-produced T[:,1:D+1] sequence.'),
        ('DOCUMENTED','Confidence head is ReplicatedLinear(H+R,1,params_dtype=float32,bias=False in this DSpark constructor). It concatenates raw hidden and embedding after embedding-to-hidden-dtype conversion, widens toFP32, projects once and applies sigmoid. It does not apply the base LM-head RMSNorm to hidden.'),
        ('DOCUMENTED','Scheduler copies probability into a persistentFP32[B,D] bank and clamps to[1e-6,1]. It computes FP32cumprod survival, updates shared budget, then allocates keep lengths. NaN is not repaired by clamp; finite-domain monotonic arguments need their own condition.'),
        ('DOCUMENTED','Periodic budget increments its counter on each update; at interval resets the counter before checking B0. For B>0 it counts survival>=threshold via .sum().item(), divides byB,ceil, increments even budgets belowD, and clamps to[min_k,D]. The newly updated budget is used by allocation in that same call. Defaults are configuration defaults, not actual captured values.'),
        ('DOCUMENTED','Allocator fills min_k per request, globally TopKs B*max(budget_k-min_k,0) candidates from flattened survival[:,min_k:] (bounded by candidate count), converts positions to request IDs by floor-div and scatter-adds counts. It is not independent per-request thresholding or a uniform fixedbudget per row.'),
        ('DOCUMENTED','Global TopK is sorted=False. Flat order, boundary ties, finite/NaN/underflow behavior and per-request count outcomes matter; a new selection/histogram protocol cannot arbitrarily reassign equal-score winners.'),
        ('DOCUMENTED','take_draft_token_ids2205-2226 retrieves out, returns it unchanged if no dynamic scheduler/result, otherwise converts each device k with int(k), clamps[0,self.num_spec_tokens], then tokens[:k] through zip. It trims already-generated lists at retrieval; it does not stop Markov generation. Vector/order/generation and current zip behavior must be preserved.'),
        ('DOCUMENTED','Next-ID helper2192-2252 unconditionally constructs backup IDs from requests.get_token_id(num_tokens_no_spec-1) and copies the backup bank toGPU. It clones sampled IDs, fills discard rows with-1, counts literal(x!=-1)&(x<V), gathers count-1 clamped to0, and chooses backup when count0. Original sampled IDs remain unmodified.'),
        ('DOCUMENTED_PLUS_MEASURED','The helper returns INT32 next IDs and INT64 counts for current typed domains. Native BOOL->FLOATsum->INT64 counts at K1/6 are exactly0..K. A fused integer count can preserve these bounded results; do not generalize unbounded shape/overflow behavior.'),
        ('DOCUMENTED','worker2033 passes both next/count to _copy_valid_sampled_token_count and to input preparation/proposal. The copy helper returns immediately if eventNone; otherwise a named copy stream waits on default, copies counts toCPU non-blocking and records an event. Async mode retains the GPU count reference; input_batch.prev_sampled_token_ids then receives next IDs.'),
        ('DOCUMENTED','Count CPU copy, GPU correction reference and next-ID state are distinct consumers with readiness/last-reader requirements. Direct sampler summaries must preserve discard/vocab filtering and both returned dtypes; accepted_count+1 alone is not this contract.'),
        ('UNKNOWN','Actual confidence/dynamic/parallel flags, ID/mask/slot values, repeated-ID hits, host scalar/copy costs, request-generation envelopes, native tie/precision details and causal gain remain unrecorded by this selector.')]
    write(O/contract_file,dict(generated_utc=stamp,source_files=source_files,evidence_refs=refs,facts=[dict(grade=g,statement=s)for g,s in facts],
        reuse=['F30-proposer-review-v9.json','M16-head-partition-postprocess-and-layout-contract-v32.json','F30-sharded-vs-owner-Markov-review-v32.json',
            'M17-probability-consumer-and-RNG-contract-v33.json','M18-partition-bank-RoPE-and-slot-contract-v34.json','M19-embedding-HC-CP-row-and-consumer-contract-v35.json'],
        local_questions_answered=['Markov seed/prediction column relation','Actual confidence head/raw hidden/embedding input and scheduler receiver',
            'Global verification allocation and periodic state ordering','Retrieval-time prefix trim versus generation-time early exit',
            'Exact next-ID predicate/count-1/discard/backup behavior and count publication consumers'],
        stop_rule='These direct local questions are answered. Continue M21 whole metadata coverage; actual-value/native numerical/performance conditions do not require SDK/kernel implementation merely for conditional discovery.',
        implementation=False,whole_module_closed=False,overall_complete=False))

    # Local functional projection of the exact next-ID helper, including its unusual hole/negative-ID behavior.
    write(O/next_file,dict(generated_utc=stamp,candidate='F31',related=['F35','H02'],evidence_grade='DERIVED_FROM_PINNED_SOURCE_PLUS_MEASURED_BOUNDED_DOMAINS',
        formula=dict(virtual_discard='Lprime[b,k]=-1 if b is discarded, else L[b,k]',predicate='p[b,k]=(Lprime[b,k]!=-1) and (Lprime[b,k]<V)',
            count='c[b]=sum_k int(p[b,k])',index='j[b]=max(c[b]-1,0)',next_ID='Lprime[b,j[b]] if c[b]>0 else backup[b]'),
        proof='Only next_ID and count escape the helper. In the normal K>=1/valid-index domain, directly reading the original matrix through a virtual discard mask gives every same Lprime read and predicate. Computing the same integer count and then the same count-1 lookup gives exactly the same two outputs without a private clone/masked matrix or separately materialized bool/count/index planes. Default variant preserves original backup construction/publication and all copy/event effects.',
        counterexamples=[dict(input=[7,-1,8],V=100,count=2,next_ID=-1,not_equivalent='Searching rightmost true position would return8.'),
            dict(input=[-2,-1],V=100,count=1,next_ID=-2,not_equivalent='Replacing predicate by0<=x<V would instead choose backup.'),
            dict(input=[7,8],discarded=True,count=0,next_ID='backup',not_equivalent='Reusing producer accepted count without discard handling is wrong.')],
        exact_current_sum_domain='K in{1,6},boolean counts in[0,6],exact inFP32 andINT64. Preserve output countINT64 and nextIDINT32.',
        other_domains=['K0 would make original gather(index0) invalid; retain fallback/error behavior or limit new path toK>=1.',
            'Do not mutate source sampled IDs or delete their response/logprob/acceptance readers.',
            'Sampler-emitted summaries need identical discard and ID-range filtering, not only an acceptance boundary Rmap.',
            'Backing bank and count-copy/GPU-reference lifetimes are not removed by this local proof.'],
        no_native_kernel_implemented=True,no_measured_gain=True,whole_module_closed=False,overall_complete=False))

    budget=dict(generated_utc=stamp,candidate='F30',related=['H02','H06'],evidence_grade='DOCUMENTED_PLUS_DERIVED_CONDITIONAL',
        current_policy=dict(probability='sigmoid(FP32_linear(float(cat(raw_hidden,cast(markov_embed,hidden_dtype))))) then clamp[1e-6,1]',
            survival='cumprod per request in original FP32 numerical contract',periodic_budget='ceil(count(survival>=threshold)/B), odd increment if even and belowD, clamp[min_k,D]',
            allocation='min_k plus request histogram of globalTopK over survival[:,min_k:]',publication='INT32 device keep_lens -> retrieval-time host ints/caps/zip/prefix lists'),
        same_embedding_proof='T[:,0]=seed and prediction i writesT[:,i+1]. Loop input embedding E_i=Embed(T[:,i]); scheduler later embeds exactlyT[:,:D]. Under same input/parameter generation and readonly retained outputs, E_0..E_(D-1) can be retained or streamed rather than gathered again. Preserve embedding-to-hidden-dtype conversion and row reshape order.',
        confidence_fusion_boundary='Read raw BF16 hidden and the same markov embedding directly, widen as original, perform one5376D FP32 projection for exampleH5120+R256 and sigmoid. Do not replace raw hidden by normalized base-head hidden, or split into separately rounded h/m scalar dots without numerical proof.',
        global_allocation_example=dict(B=2,D=3,min_k=1,budget_k=2,survival=[[.9,.8,.7],[.9,.6,.5]],keep_lens=[3,1],meaning='Two extra candidates go to row0 globally; this is not uniform[2,2]. Example is a policy identity, not captured values.'),
        bounded_prefix_lemma=[
            'Assume finite clamped probabilities, a validated/conservative upper bound U_i on every future published survival of a request, already-generated mandatory min_k prefix, and a fixed known final extra-budget L for this update.',
            'Once at least L evaluated candidate values exist, let tau be their Lth-largest value. If U_i<tau, at least L already-evaluated candidates outrank every uncomputed tail value of request i. Adding other candidates can only strengthen exclusion. Preserve a unique boundary or the original tie contract.',
            'On non-budget-update steps L is the known existing budget. On periodic recompute steps do not use old L: resolve the final budget first (for example threshold-count upper/lower bounds yield the same rounded/clamped budget), or disable this pruning for that update.',
            'If threshold statistics still need accounting, also prove the uncomputed tail contributes no unknown threshold count, or use an exact count bound. U_i<threshold alone does not fix a global budget that other requests can still change.',
            'Only post-decoder Markov/confidence and optionally lazily evaluated head rows may be pruned when every tail observer is dead. The noncausal draft decoder still needs its original future query/KV rows; verify-prefix exclusion does not make those decoder inputs dead.',
            'Preserve all RNG calls/state or restrict to the corresponding deterministic greedy domain, score-write/ID-bank observers, error behavior and metadata generations. A mathematical bound is not proof of native precision or speedup.'
        ],
        current_API_boundary='Python budget_k drives torch.topk k. Keeping budget on device requires a new compatible selection/publication protocol; merely replacing the int with a Tensor does not establish removal of synchronization.',
        no_current_flag_or_gain_claim=True,implementation=False,whole_module_closed=False,overall_complete=False)
    write(O/budget_file,budget)
    write(O/framework_file,dict(generated_utc=stamp,framework_candidates=['H02','H06'],evidence_refs=refs,
        H02=[
            'Next-ID helper CPU backup list/get_token_id and copy_to_gpu are unconditional source work. Reusing a device backup bank requires exact request/position/no-spec/accepted-rollback generation; never add count.all().item() merely to skip fallback publication.',
            'take_draft_token_ids converts each device keep length viaint(k). This proves host scalar consumption, not B measured synchronize calls. A batched asynchronous length copy can share a generation envelope and readiness event with draft IDs.',
            'Envelope must preserve out.req_ids order, length shape, producer generation, consumer max/min cap and zip behavior; persistent length bank can be overwritten by later updates.',
            'Existing valid-count eventNone early return, named-stream wait/copy/event, GPU count reference and prev_sampled token update are preserved. Readiness is not pointer equality.'
        ],H06=[
            'Scheduler budget comment cites past Qwen3-8B/A3 behavior; it is not a measured910C/CANN9.2 cost curve. Current actual dynamic params/branch time are unrecorded.',
            'Verification length is a logical prefix count; physical target padding, graph bucket, DP/TP/MC2 uniform domains and batch completion can leave cost unchanged.',
            'Cost-policy tuning is a separate semantic counterfactual from preserving the existing policy in a device pipeline; retain target/speculative correctness and RNG/metadata contracts.',
            'Regroup physical buckets only after preserving the original global allocation cohort and request map. Changing batch membership before globalTopK changes that policy.'
        ],saved_profile_boundary='No new on-CPU or transfer timing was measured. Existing H02 prepare-only utility-field census does not prove this outside-prepare backup handoff free or absent.',
        implementation=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))

    variants=[
        ('F30-v36-SELECTED-BASE','F30','Feed base norm/head from ordered selected hidden rows without a private Index materialization where legal.','Current104Index/Check entries are positive; same row count is not identity. Keep raw hidden for confidence, ordered duplicates and BF16/head layout.'),
        ('F30-v36-MARKOV-EMBED','F30','Reuse or stream seed-inclusive Markov input embeddings into confidence.','SameE_i=Embed(T[:,i]) for loop and schedulerT[:,:D]; preserve parameter/input generation, dtype conversion, row order and readonly lifetime. Confidence-active branch only.'),
        ('F30-v36-CONFIDENCE-PIPE','F30','Direct two-input FP32 confidence projection/sigmoid and compatible probability/prefix publication.','Keep one FP32 dot boundary from raw hidden plus embedding; Cat/widen planes may be private. No normalized-hidden substitution or unproved split-dot sum.'),
        ('F30-v36-DEVICE-BUDGET','F30','Fuse clamp/cumprod/statistics/global selection-to-counts and keep budget state on device under a new selection ABI.','Preserve native arithmetic, topK boundary/ties, periodic counter/zeroB/odd-round ordering, outputINT32 bank and host publication. Not a flag-only change to torch.topk.'),
        ('F30-v36-BOUNDED-PREFIX','F30','Use rigorous future-survival bounds to avoid post-decoder tail evaluation when it cannot affect final prefix allocation.','Fixed finalL, enough evaluated candidates, strict upper-bound separation, native/tie/RNG/observer conditions. Periodic budget may need interval resolution; retain full noncausal decoder input domain.'),
        ('F31-v36-NEXT-ID-FUSE','F31','Fuse virtual discard, exact predicate/count, count-1 gather and backup select.','Local two-output proof in F31 review; preserve negative-other-than-1 behavior, holes, output dtypes, backup production, no mutation of original IDs and existing event consumers.'),
        ('F31-v36-PRODUCER-SUMMARY','F31','Have the sampling/output producer publish the same next-ID and valid-count summary.','Producer must apply this consumer discard/vocab rule or pass sufficient information for exact correction. R+1 and rightmost-valid shortcuts need extra prefix/valid-ID proofs; preserve original sampled matrix observers.'),
        ('H02-v36-BACKUP-GEN','H02','Maintain/reuse a device backup-token bank under exact request-state generations, or skip dead fallback production under an already-proved host invariant.','Do not infer all counts positive from shapes. Preserve prompt onboarding/async accepted state/rollback/request moves and normal error domain; no new blocking .item guard.'),
        ('H02-v36-LENGTH-BULK','H02','Batch asynchronous verification-length D2H with draft-ID publication and consume one matching generation envelope.','Preserve req_id order, k caps/zip, producer/readiness/last-reader and buffer overwrite epochs. Source scalar consumption is not a measured synchronization count.'),
        ('H06-v36-COST-POLICY','H06','Calibrate verification budget to actual fixed-model/hardware/workload cost geometry.','A changed policy needs speculative correctness/RNG contracts and real padding/communication cost evidence; oldA3 odd-budget comment is not current910C evidence.'),
        ('H06-v36-GEOMETRY','H06','Couple logical verify prefixes to compatible physical buckets/packing after preserving global allocation.','Keep original global cohort and request identities, prefix cache/slot/state and DP/TP/MC2 participation. Logical length reduction is not automatically less physical work.')
    ]
    variant_rows=[dict(id=i,parent=c,transformation=t,conditions=k,implemented=False,native_validation=False,critical_path_gain_us=None)for i,c,t,k in variants]
    requirements=[
        ('VISIBILITY','Actual dynamic/parallel flags and branch costs remain UNKNOWN; exact0ComputeInfo hits are not absent CPU/transfers or disabled flags.'),
        ('SUMROUND','Keep BF16 base/bias output and BF16 addition before argmax, existing ties/NaN/vocab postprocessing and writeback observers.'),
        ('RNG','Preserve all probability/draft_probs/full selected residual/RNG consumers; greedy Markov trace is not target all_greedy.'),
        ('TOKENSPACE','Embedding/cache keys use the actual post-remap Markov input token space and model/weight namespace, not raw draft columns.'),
        ('HIDDENROWS','Confidence uses raw last_hidden_states row order; base-head Index/normalization is a distinct consumer plane, even for equal shapes.'),
        ('SEED','T0is seed, loopE_i consumesT_i and writesT_(i+1); confidence readsT[:D]. Do not shift reused embeddings to just-produced outputs.'),
        ('EMBEDGEN','Retained embedding values readonly through confidence, with matching input/parameter/dtype/generation; actual repeat-hit rates not inferred.'),
        ('CONFIDENCEFP32','Preserve embedding-to-hidden-dtype then concatenation/widening, one FP32 projection and sigmoid; split dots/other norm need separate numerical proof.'),
        ('PROBPREFIX','Preserve clamping/NaN and FP32 cumprod association/underflow/threshold values; log-domain replacement is not exact by real algebra alone.'),
        ('TOPKTIE','Global flattened candidate window, sortedFalse boundary ties and request-count outcomes retained or explicit strict boundary conditions used.'),
        ('GLOBALBUDGET','Allocation is global acrossB requests with min_k and bounded extra budget, not independent per-request uniform caps.'),
        ('PERIODICSTATE','Counter increment/reset, B0 behavior, ceil/odd adjustment/clamp and same-update use of new budget preserved.'),
        ('DEVICEABI','Device budget requires a compatible dynamic-budget selection/publication interface; old Python topk-k consumption cannot be assumed removed.'),
        ('TAKEOUTPUT','Retrieval caps each int(k), preserves out.req_ids order and current zip/prefix-list behavior; it is post-generation output trimming.'),
        ('PUBGEN','Length and draft-ID envelopes refer to the same request/producer generation; reusable GPU banks and async copies respect all last readers.'),
        ('SCALARCOST','Host scalar consumption and .item are source facts, not measured Bsync calls/latency; retain profile cost uncertainty.'),
        ('BOUNDDOMAIN','Prefix pruning needs finite-domain validated/conservative bounds on actual published native survival and strict separation/unique selection or original ties.'),
        ('BOUNDBUDGET','Final extraL must be fixed/known. On periodic updates resolve new-budget bounds first or disable pruning; oldL and U<threshold alone are insufficient.'),
        ('DECODERDOMAIN','Prune only dead post-decoder work; original noncausal draft decoder future query/KV inputs still influence prefix hidden states.'),
        ('TAILREADERS','All omitted tail IDs/scores/probabilities/bank writes/RNG/error observers preserved or independently proved dead.'),
        ('VALIDPRED','Literal(x!=-1)&(x<V), including negative values other than-1, preserved; do not substitute nonnegative validity.'),
        ('DISCARD','Virtual discard reproduces original clone+index_fill normal-domain reads and leaves original sampled matrix unchanged.'),
        ('COUNTINDEX','Selection uses max(count-1,0), not max true position. Hole behavior remains unless a valid-prefix invariant is separately established.'),
        ('COUNTDTYPE','Current K1/6 boolean count exactly0..6; output countINT64 and nextIDINT32 retained, no unbounded overflow/dtype generalization.'),
        ('ERRORDOMAIN','K>=1 and valid discard indices chosen for new normal path; retain fallback/errors for empty width/out-of-range/other domains.'),
        ('BACKUP','Zero-count fallback and authoritative request.get_token_id generation retained; default fusion keeps original backup publication.'),
        ('EVENTNONE','valid_sampled_token_count_eventNone returns before later GPU/reference/prev-ID updates; preserve this branch.'),
        ('COUNTREADY','Default-to-copy-stream wait, nonblocking CPU count copy/event, async GPU count reference and nextID prev-state readiness/last-read preserved.'),
        ('SUMMARYINPUT','Sampler summary applies exact discard and vocab filtering; accepted boundary+1 is not automatically this count.'),
        ('BACKUPGEN','Device backup reuse includes request/position/no-spec/accept-reject/rollback/permutation/prompt-onboarding generations, not address/shape.'),
        ('NOBLOCKGUARD','Do not introduce count.all().item() or other new blocking device test to avoid backup/packing work.'),
        ('COSTPOLICY','Policy changes are distinct from semantic-preserving fusion; actual910C/CANN9.2 padded-bucket cost and speculative/RNG correctness required.'),
        ('COHORT','Regroup after preserving global allocation cohort and request map; changing batch membership changes TopK allocation.'),
        ('PHYSICALWORK','Logical rows/bytes/TASK sums are not transport/peak-memory/request speedup; new launch/packing/communication/interference costs counted jointly.'),
        ('BRANCHUNIVERSE','Zero-spec/one-spec/parallel/selector/probabilistic/nonDSpark/autoregressive/empty/config-error branches remain explicit and not closed by current counts.')]
    assert len(variant_rows)==11 and len(requirements)==35
    write(O/variants_file,dict(generated_utc=stamp,variants=variant_rows,conditions=[dict(id='M20-v36-'+k,requirement=v,actual_current_legality_proved=False)for k,v in requirements],
        relationships=['F31 standalone fusion and producer summary remove overlapping private scans/copies; count the joint graph once.',
            'Embedding CSE and streaming confidence are composable but reuse the same256D values once; no cache-hit speedup inferred.',
            'Device scheduler and bulk host publication are dependent protocols; preserving current policy and changing cost policy are different counterfactuals.',
            'Bounded post-decoder pruning does not remove model-forward work; lazily splitting batched base heads has a separate throughput/launch tradeoff.',
            'H06 physical bucketing preserves the policy global cohort and fixed-group collective ABI; do not add nominal logical token savings as hardware gain.'],
        implementation=False,whole_module_closed=False,overall_complete=False))
    branches=[
        ('Current greedy Markov','MEASURED','104seed-to-five-step patterns,520exact11TASK steps;440proposal/80dummy. Seed is not first prediction.'),
        ('Next-ID normal domains','MEASURED_PLUS_DOCUMENTED','88helpers:K1/B2x16,K6/B2x16,K6/B4x56; exact discard/count-1/backup semantics pinned.'),
        ('Dynamic DSpark','DOCUMENTED_ACTUAL_CONFIG_UNKNOWN','Raw hidden+seed-inclusive embeddings→FP32confidence→survival→global budget→device lengths; exact native ComputeInfo locations0.'),
        ('DFlash probability proxy','DOCUMENTED_OTHER_BRANCH','max softmax(logits.float) confidence proxy differs from DSpark confidence head; do not conflate algorithms.'),
        ('Periodic versus nonperiodic/empty','DOCUMENTED','Counter/zeroB/new-budget use order pinned; actual interval/threshold params unknown.'),
        ('Proposal retrieval','DOCUMENTED','take_draft_token_ids trims already-generated lists; method has only super.take direct same-file match, not a complete EngineCore RPC/queue trace.'),
        ('Selector/probability/early-return/AR continuation','DOCUMENTED_OUTSIDE_SELECTED_POSITIVES','Merged source retains these paths, _propose has zero-spec return; actual flags not supplied by source alone.'),
        ('Discard/invalid/hole/backup cases','DOCUMENTED','Private discard clone, !=-1 &<V, count-1, zero backup; observed IndexFill/clone/H2D absence in ComputeInfo not absence of those effects.'),
        ('Count-copy event/async branches','DOCUMENTED','eventNone early return and otherwise separate CPU event/GPU count reference/prev-ID consumers preserved.'),
        ('Sharded/owner Markov and score consumers','REUSED_V32_V33_CONDITIONAL','Confidence does not require full summed logits, but probability/raw-score/draft_probs and owner publication/RNG contracts remain.'),
        ('Cost-policy/geometry change','CONDITIONAL_NOT_MEASURED','A3/Qwen heuristic not current910C evidence; fixed-case graph/DP/TP/MC2 physical domains remain.')]
    write(O/branches_file,dict(generated_utc=stamp,branches=[dict(branch=b,state=s,evidence=e,whole_branch_closed=False)for b,s,e in branches],
        stop_rule='Confidence/scheduler/head/retrieval/next-ID local contracts answered. Move to whole M21 input metadata and then M22-M25; no whole SDK audit or native implementation needed just to record conditional discovery.',whole_module_closed=False,overall_complete=False))

    nodes=['raw_hidden','base_norm_head','seed_T0'];edges=[('raw_hidden','base_norm_head','ORDERED_SELECTED_ROWS')]
    for i in range(5):
        ns=[f'T{i}',f'E{i}',f'Bias{i}',f'SumBF16_{i}',f'ID{i+1}'];nodes+=ns
        if i==0:edges.append(('seed_T0','T0','KNOWN_SEED'))
        else:edges.append((f'ID{i}',f'T{i}','PREVIOUS_PREDICTION'))
        edges += [(f'T{i}',f'E{i}','EMBEDDING_INPUT'),(f'E{i}',f'Bias{i}','MARKOV_W2'),('base_norm_head',f'SumBF16_{i}','POSITION_BASE'),(f'Bias{i}',f'SumBF16_{i}','BF16_ADD'),(f'SumBF16_{i}',f'ID{i+1}','ARGMAX_MAP_PUBLISH')]
    nodes+=['confidence_reembed_seed_prefix','confidenceFP32','survival','global_budget_keep','take_output_prefix']
    edges += [('raw_hidden','confidenceFP32','RAW_NOT_NORMED'),('seed_T0','confidence_reembed_seed_prefix','SEED_PREFIX'),('confidence_reembed_seed_prefix','confidenceFP32','MARKOV_INPUTS'),('confidenceFP32','survival','CLAMP_CUMPROD'),('survival','global_budget_keep','GLOBAL_SELECTION'),('global_budget_keep','take_output_prefix','HOST_LENGTHS')]
    for i in range(1,6):edges.append((f'ID{i}','take_output_prefix','ALREADY_GENERATED_IDS'))
    for i in range(1,5):edges.append((f'ID{i}','confidence_reembed_seed_prefix','PREVIOUS_PREDICTIONS_ONLY'))
    baseline=graph(sorted(set(nodes)),edges)
    summary_graph=graph(['sampled_IDs','discard_set','backup_bank','virtual_discard_count_select','next_ID','count_I64','CPU_count_copy_event','GPU_correction_reference','prev_ID_state'],
        [('sampled_IDs','virtual_discard_count_select','READONLY'),('discard_set','virtual_discard_count_select','VIRTUAL_MINUS1'),('backup_bank','virtual_discard_count_select','ZERO_COUNT_FALLBACK'),('virtual_discard_count_select','next_ID','INT32'),('virtual_discard_count_select','count_I64','INT64'),('count_I64','CPU_count_copy_event','COPY_STREAM_READY'),('count_I64','GPU_correction_reference','ASYNC_LAST_READER'),('next_ID','prev_ID_state','SAME_GENERATION')])
    device_graph=graph(['retained_markov_E','raw_hidden','confidence_direct_FP32','prefix_scan','device_budget_counter','global_selection_counts','generation_envelope','async_host_copy','take_matching_prefix'],
        [('retained_markov_E','confidence_direct_FP32','SAME_SEED_INPUT_VALUES'),('raw_hidden','confidence_direct_FP32','RAW_ROW_ORDER'),('confidence_direct_FP32','prefix_scan','CLAMPED_PROB'),('prefix_scan','device_budget_counter','PERIODIC_THRESHOLD_COUNT'),('device_budget_counter','global_selection_counts','SAME_UPDATE_BUDGET'),('prefix_scan','global_selection_counts','SURVIVAL_CANDIDATES'),('global_selection_counts','generation_envelope','INT32_LENGTHS'),('generation_envelope','async_host_copy','SNAPSHOT_READY'),('async_host_copy','take_matching_prefix','REQUEST_ID_EPOCH')])
    write(O/graphs_file,dict(generated_utc=stamp,source_consistent_greedy_and_conditional_dynamic_branch=baseline,conditional_next_ID_summary=summary_graph,conditional_device_budget_publication=device_graph,
        notes='Baseline combines source branches to show consumers, not a claim dynamic executed in current profile. Early-return/probability/AR/dummy scopes remain separate. Graphs are not implemented schedules.',graphs_acyclic=True,implementation=False,whole_module_closed=False,overall_complete=False))

    evidence=[profile_file,contract_file,next_file,budget_file,framework_file,variants_file,branches_file,graphs_file]
    common=dict(contract=str(O/contract_file),next_ID=str(O/next_file),budget=str(O/budget_file),framework=str(O/framework_file),conditions=str(O/variants_file),graphs=str(O/graphs_file),implementation=False,new_performance_result=False)
    candidate_updates={}
    for cid in ['F30','F31']:
        vv=[v for v in variant_rows if v['parent']==cid]
        candidate_updates[cid]=dict(review_v36=dict(common,variants=vv),variants_v36=vv,target_source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v36.json'),
            profile_review_state='CURRENT_STEP_PATTERNS_AND_CONFIDENCE_NEXTID_CONSUMERS_REVIEWED;CONDITIONAL_UNPRICED',
            conclusion_changing_gap='Current actual flags/values/RNG/tie/numerical/alias/transport-generation and causal benefit remain conditional. Direct seed/confidence/budget/prefix/next-ID questions answered.',
            next_profile_or_contract_action='Advance whole M21 metadata coverage; do not repeat direct scheduler/head/next-ID readers or read SDK merely for conditional registration.',all_consumers_closed=False)
    candidate_updates['F26']=dict(review_v36=dict(common,relation='Raw collapsed hidden is a separate confidence consumer; do not replace it by base-head normalized hidden or remove it through head fusion.'))
    candidate_updates['F35']=dict(review_v36=dict(common,relation='Sampler-emitted next/count summary must reproduce downstream discard/vocab/count-1/backup rule and count-copy events; Rmap+1 is insufficient without extra invariants.'))
    framework_updates={cid:dict(review_v36=dict(common,variants=[v for v in variant_rows if v['parent']==cid]),new_performance_result=False,implementation=False,
        next_profile_or_contract_action='Retain exact request/generation/event/physical-geometry conditions; continue M21 complete saved-copy/input metadata coverage.')for cid in ['H02','H06']}
    findings={
        'M01':'Retrieval-time verify prefix truncation preserves request IDs/order/caps/zip; policy counterfactual needs speculative/RNG correctness beyond current greedy-step positives.',
        'M03':'Per-request device-length host int consumption and asynchronous generation envelope are source contracts; actual onCPU/sync cost not measured by ComputeInfo.',
        'M04':'Exact next-ID discard/predicate/count-1/backup and count copy/GPU-reference/prev-ID generation pinned; fused normal-domain two-output projection derived.',
        'M15':'Confidence reads raw collapsed hidden, not base-head RMSNorm output; selected head rows and raw hidden fanout remain distinct.',
        'M16':'Base norm/head and full-vocab reconstruction reusedv32; confidence5376D FP32 scalar head does not require full summed-vocab logits.',
        'M17':'Producer summaries can avoid downstream scan only with exact discard/vocab/count-1/RNG/output contracts; holes and negative-other-than-1 behavior retained.',
        'M20':'104five-step patterns/520exact11TASKsteps plus conditional confidence/global-budget/retrieval pipeline reviewed;11variants across F30/F31/H02/H06 with35conditions, no gain/implementation.',
        'M21':'88next-ID preparations/1144ComputeInfoTASK; CPU backup/clone/indexfill/D2H absent in selector is not no work. Complete metadata/copy coverage remains next frontier.',
        'M22':'Persistent token_probs/survival/length banks, sampler outputs and count references need snapshot/readiness/last-reader generations; no address-only reuse.',
        'M23':'Owner/sharded Markov protocols retain full distribution consumers and global budget cohort; no independent per-rank prefix allocation shortcut.',
        'M24':'Periodic budget counter/reset/zeroB/same-update ordering and seed-inclusive embedding reuse key pinned; bounded-prefix proof needs fixed finalL and native upper bounds.',
        'M25':'take_draft_token_ids cuts already-generated proposal lists with device length ints; complete EngineCore/RPC/response lifecycle remains outside this local consumer method.'}
    module_updates={mid:dict(finding=text,evidence=evidence,candidates=['F30','F31']+(['F26']if mid in ['M15','M16']else[])+(['F35']if mid in ['M04','M17','M21']else[]),gap='Direct local contracts answered; full metadata/transport/branch universe and actual values/native math/causal benefit remain separate obligations.')for mid,text in findings.items()}
    remaining=[s for s in deepcopy(read(O/'module-coverage-gaps-v35.json')['remaining'])if not s.startswith('NextwholecoveragefrontierM20')]
    remaining.insert(14,'M20v36directcontractsanswered:104five-step/52011TASKpatterns,seedT0→embeddingTi→predictionT(i+1),confidence reusesT[:D]rawhidden,FP32head/sigmoid/globalcumprod/periodicbudget/topKrequesthist and take_draft outputtrim.0exactComputeInfo matches does not prove disabledflags/noCPUcopies. F31exactnextIDvirtualdiscard/count-1/backup proof and asynccountconsumersbound;35conditions/11variantsrefineF30/F31/H02/H06. Do not rereadtheseaswhollyunknown. Actualflags/values/tie/nativeprecision/generation/gain remainconditional.')
    remaining.insert(15,'NextwholecoveragefrontierM21/F31/H02/H06 metadata/inputpublication across target/draft/accept-reject/state. Startallranknativefullcallers/typedTASK and saved host-copy/queue/prepare-profile surfaces; include outside-prepare backup2217/clone/indexfill/countD2H if present, without treating ComputeInfo0 asno work. Reusev12/v13CP-field proof andv28-v36row/slot/count/dynamic contracts; focus previouslyunreviewed per-field producers/consumers andshared publication, notwholeproposer/SDK. ThenM22-M25andfiniteglobalcoverageaudit;35candidateinventorynotwholegoalcompletion.')
    summary=dict(filename='M20-confidence-and-next-ID-summary-v36.json',previous_turn_classification='PROGRESS:v35decoderCP/group/HCcontracts',
        native_APIs=4464,native_TASKs=7520,native_chains=28,source_bound_COMM=104,five_step_patterns=104,steps=520,step_TASKs=11,pattern_issues=0,next_ID_helpers=88,next_ID_TASKs=1144,
        exact_dynamic_native_API_matches=0,actual_dynamic_flags='UNKNOWN_NOT_RECORDED',conditional_variants=len(variant_rows),condition_rows=len(requirements),
        refined_model_candidates=['F30','F31'],related_model_candidates=['F26','F35'],refined_framework_candidates=['H02','H06'],new_model_candidates=[],
        remote_jobs=8,remote_pass=7,remote_retained_errors=1,
        remote_job_status='8remoteCPUjobs terminal:7pass/1retained first-reader span guard error, corrected in explicitv2. Main semantic/math review pass. No model/torch/NPU/service/new-health work.',
        evidence_refs=refs,no_new_model_NPU_service_or_health_work=True)
    report=f'''本轮按保存profile→必要定点源码推进M20及下一ID交界。覆盖4464原生API、7520 TASK、28条链、104条完整COMM；104组seed到五步发布的源码顺序模式全部匹配，每步11TASK，共520步。16组dummy B4、32组proposal B2、56组proposal B4。5824个seed/step TASK之外保留552个base index/head/reconstruction和1144个下一ID TASK，没有按相同计数或最近时间绑定target请求/root。全体127920原生API补查确切scheduler/confidence/取回/backup位置后仍0个ComputeInfo-linked命中；不据此认定配置关闭或没有CPU、memcpy、H2D/D2H和同步。

Markov与confidence的输入关系已经说清：T[:,0]是已知seed，步骤i读Embed(T[:,i])并把预测写入T[:,i+1]；early-return分支返回T[:,1:]。第一列不是“多算后丢弃的第一个预测”。confidence再次读取的正是T[:,:D]，因而可条件地保留/流送循环中的embedding，不能错移成当前新预测。confidence使用raw last_hidden_states，将同dtype的hidden与Markov embedding拼接再float，经过FP32的5376→1投影和sigmoid；它不使用base LM-head的RMSNorm输出。可研究直接双源读取、查表复用和pipeline，但必须保留原dtype/单次dot/row关系。

实际DynamicSpecScheduler在spec_decode/utils.py。它clamp概率、FP32 cumprod，再周期性以threshold计数的.item()更新共享budget，然后从整批survival[:,min_k:]选globalTopK，按request计数得到INT32长度。不是每请求独立固定k或阈值截断。周期counter、B0、ceil/奇数调整和“本次更新立即用于本次分配”的顺序都有影响。旧注释的Qwen3-8B/A3成本观察不证明910C/CANN9.2最优预算。device预算重组需要新的选择/发布ABI，不能只把Pythonint换成Tensor。

长度在take_draft_token_ids取回阶段被逐个int(k)、clamp，再通过zip对已生成列表tokens[:k]裁剪。这证明它是输出取回时的prefix，不是当前Markov生成时的提前退出。批量异步回传长度、与draft IDs按request/generation一起发布，是H02条件变体；不能把源码中的B个标量消费说成测得B次同步。prefix上界剪枝也只能在最终extra预算L已固定、有足够已评估候选、未来native survival上界严格低于边界且尾部无其他观察者时考虑。周期更新不能拿旧L套用；非因果draft decoder仍需要其未来query/KV行。这里只给条件证明，没有实现或测速。

下一ID辅助函数88组，各13个ComputeInfo TASK，输入K1或6。它先生成CPU请求状态中的backup并拷到GPU，克隆sampled IDs、把discard行填-1，然后使用字面谓词(x!=-1)&(x<V)计数，再取max(count-1,0)列，count0回退backup。并非寻找任意位置最后一个有效token：[7,-1,8]得到count2且选-1；[-2,-1]得到count1且选-2。单核可用虚拟discard读原矩阵，精确计算同样两项输出，省私有clone/mask/index平面；不能擅自增加x>=0判断或把count-1改成max-true-index。默认方案保持backup构造/发布，另将sampler直接发布摘要与device backup代际复用列为独立条件。

count还有异步消费者：eventNone会提前返回；否则copy stream等待default，非阻塞D2H并记event，async模式保留GPU count引用，随后更新prev_sampled IDs。融合/复用不能只保持指针，还要保持CPU拷贝、GPU修正和next-ID状态的ready/last-reader代际。producer的R+1只有在相同discard/ID有效性/prefix不变量下才可能代替这里的计数。

本轮细化F30/F31并关联F26/F35，同时更新H02/H06，记录{len(variant_rows)}个条件变体、{len(requirements)}项约束、分支台账与3张关系图。8次远端CPU作业7pass、首版tail跨度guard失败1次保留并已修正；主审通过。持久化在新helper v2中增加框架台账细化，原v35已校验helper保持字节不变。没有导入模型/torch、启动NPU、改服务或做新健康检查，最后健康仍v12post。35模型候选、7框架候选、0整模块关闭；26项理论比较INCOMPLETE、3136candidate资源行、849PMU缺失和16个phase-unbound图保持。下一批M21完整metadata/输入发布，继续用profile找未审字段，再定点补代码。整体Goal active。
'''
    jobs=[dict(kind=kind,status=status,description=desc,artifacts=[N/script,N/(label+'.json')])for kind,status,label,script,_,desc in specs]
    jobs.append(dict(kind='review',status='pass',description='Main semantic/math review: exactnextIDvirtualdiscard proof/counterexamples,seed-inclusiveconfidenceCSE/globalbudget and fixed-Lprefix-bound conditions;11variants/35conditions and framework ledger refinement,no implementation/gain.',artifacts=[O/q for q in evidence]+[N/'persist_profile_refinement_v2.py']))
    action=('继续Loop004，v36实质进展。先读M20-confidence-and-next-ID-summary-v36.json、M20-Markov-confidence-and-verification-contract-v36.json、F31-next-ID-and-acceptance-summary-review-v36.json、F30-dynamic-verify-budget-and-prefix-bounds-v36.json、module-coverage-gaps-v36.json。'
        'Profile4464API7520TASK28chains104COMM；104seed→5step模式520步每11TASK，无unmatched；5824seedstep+552basehead/index+1144nexthelper。88helper K1B2x16/K6B2x16/K6B4x56，不按count/nearesttime配target。127920exactsourcequery0ComputeInfo不推dynFalse/noCPUcopy/sync。'
        'T0seed，i读Embed(T_i)写T_i+1，返回T[:,1:]early branch；confidence再Embed(T[:D])可同代际reuse，raw last_hidden不是LMheadnorm，concat后F32单5376→1sigmoid，不能错shift或splitdot误舍入。'
        'DynamicSchedulerutils94a569fa... updateclamp→F32cumprod→periodicthreshold.item+ceil/odd/clamp→globaltopKrequesthistINT32，非每req独立。counter/reset/B0/same-update新budget顺序保留。take_draft2205-2226逐int(k)/cap/zip截已生成proposal；bulkasyncD2H要绑定reqID+draft/lengthgen，非B已测sync。'
        'Boundprefix仅固定known最终L、足够已算candidate、validated futureNative U<tau+ties/RNG/tailobserver条件；periodic需先解预算bounds或禁prune，U<T alone不够。只postdecoder，可lazyhead有launchtradeoff；保留原非因果decoder未来KV/query行。'
        'NextID2192-2252 CPUbackupget_token_id(numtokensnospec-1)copyGPU，clone+discard-1，谓词x!=-1 &&x<V，count-1clamp0gather，c0backup；[7,-1,8]选-1，[-2,-1]选-2。K1/6数0..6nativeF32sum与int精确；输出nINT32/countINT64，虚拟discard可去私有clone/mask/index平面但保原IDs/backup及错误domain。producerR+1需同discard/vocab/prefix条件。'
        'worker copycount eventNone先return；否则copyStreamwaitdefault→nonblockingCPUcountcopy/event；asyncGPUcountref和prevSampled n独立lastreaders；bankgen别pointer-only。H02backupgen/bulklen，H06实际910Ccost+physicalbucket域，A3/Qwen旧注释非现成本。'
        '8remote7pass1spanGuarderror保留，mainpass共9runs8pass1error；35model7framework0wholemodule，11variants35conditions，F30/F31 refine+F26/F35 relation,H02/H06 frameworkreviewv36。新persist_profile_refinement_v2.py支持framework_updates，原helper未改；无model/torch/NPU/service/newhealth,lastv12post；26theoryINCOMPLETE/3136candidate资源/849missingPMU/16phaseunbound保持。'
        '下一批M21全metadata/inputpublication，先allranknative/typedTASK+savedhostcopy/queue/prepare；特别outside-preparebackup/clone/indexfill/countD2H可见性，复用v12/v13以及v28-v36，查未审字段/producerconsumer共享而非整proposer/SDK。随后M22-M25/globalfinitecoverageaudit，Goalactive。')
    persist_refinement(version=36,previous=35,title='v36 M20 confidence budgets and exact next-ID handoff',summary=summary,
        changes=['104five-step patterns/52011TASKsteps and88next-ID13TASKsurfaces fully retained; exact dynamic visibility bounded without false flags.',
            'Seed-inclusive confidence/raw-hidden/F32head/globalbudget/retrieval-prefix and next-ID/count-copy contracts point-bound; local virtual-discard proof and hole/negative-ID counterexamples added.',
            '11conditional variants/35conditions refineF30/F31/H02/H06 with native/tie/RNG/fixed-budget/generation obligations; no implementation or measured gain.'],
        candidate_updates=candidate_updates,framework_updates=framework_updates,module_updates=module_updates,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=[contract_file,next_file,budget_file,framework_file],publication_rows={
            'F30':'| F30 | DSpark Markov、confidence 与验证预算重组 | 条件 | v36核对104五步/520步，绑定seed-inclusive embedding复用、raw hidden FP32confidence、global生存概率/预算和取回时prefix裁剪。device预算/批量长度发布与严格上界post-decoder剪枝为条件；当前dynamic flag/耗时未知，不把逻辑长度当物理收益。 |',
            'F31':'| F31 | Attention/DSpark metadata 与输入摘要发布 | 条件 | v36补88组next-ID的精确discard、!=−1且<V计数、count−1选择和backup合同；正常K1/6可研究虚拟discard单核或sampler直接摘要，保留原矩阵、INT32 next/INT64 count、backup和CPU事件/GPU计数代际。空洞/负数例外禁止rightmost或R+1无条件替代。 |'},
        gate_updates=dict(status='PROFILE_AND_NECESSARY_POINT_CONTRACTS_REVIEWED',answered_by=evidence,
            closed_scope='Current step pattern/seed/actual confidence/scheduler/retrieval-prefix/next-ID formula and count publication local contracts.',
            not_closed='Actual flags/values/whole metadata and EngineCore transport/all aliases/RNG/native tie-precision/physical causal gain or whole module.',
            stop_rule='Advance whole M21 saved-profile metadata coverage. Do not read SDK/native implementations merely to record these conditional opportunities.'),
        scripts=[N/script for _,_,_,script,_,_ in specs]+[N/'persist_profile_refinement_v2.py'],executor_prefix='zcode-M20-',main_script=N/'finalize_M20_confidence_v36.py')


if __name__=='__main__':
    main()
