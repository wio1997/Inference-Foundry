from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json, hashlib

N=Path(__file__).parent
O=N/'profile-first-20261031'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
W=R/'whole-scope-20261030'
S=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend'
DBSHA='084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4'


def receipt(name):
    r=json.loads((N/name).read_text())
    assert r['returncode']==0,(name,r['returncode'])
    return json.loads(r['stdout'])


def main():
    moe=json.loads((O/'moe-profile-summary-v7.json').read_text())
    assert moe['count_hist']=={'1':1280}
    raw=receipt('archived-ep3-dspark-moe-evidence166-v7-v2.json')
    audit=receipt('archived-ep3-dspark-pipeline-audit166-v7-v2.json')
    observed=receipt('observed-moe-counts-evidence166-v7.json')
    communication=receipt('archived-ep3-dspark-communications166-v7.json')
    ancestry=receipt('archived-ep3-dspark-phase-ancestry166-v7.json')
    embedding=receipt('archived-ep3-embedding-frames166-v7.json')
    assert raw['result_sha256']=='e806fc67206b77a7d90d1d70f289eab84656e78fb813c1636509c901d51c4a98'
    assert audit['result_sha256']=='7a6bd748cf829df459eb7f4d865aa4afd9c25786f84d7ff89e968ef149310727'
    assert communication['result_sha256']=='efc552f7a34c0cf5434ab2a0901cfce3beba8617e43e68f50843d22c59048bc9'
    assert ancestry['result_sha256']=='5212d636baeb3da0a15d9e4a23af2604abdcb4d3d30487c37f013ed204321116'
    assert raw['database_sha256']==DBSHA and raw['database_stat_unchanged'] and raw['task_count']==3307
    assert observed['source_stats_unchanged'] and observed['cohorts']==296 and observed['snapshots']==4736
    assert communication['summary']['communication_ops']==communication['summary']['host_HCCL_APIs']==165
    for role in communication['summary']['per_role'].values():assert role['matched_COMM_count_hist']=={'1':role['api_count']}
    a=audit['summary']
    assert a['F28']['repeat_records']==13 and a['F28']['hcpre_count_per_ordered_repeat_group']=={'6':13}
    assert a['F29']['context_projection_records']==39 and a['F29']['pipeline_scatter_count_hist']=={'1':39}
    phases={(r['role'],r['phase']):r['count'] for r in ancestry['summary']['counts']}
    assert phases[('repeat','DUMMY_ANCESTRY_CONFIRMED')]==1 and phases[('repeat','PROPOSAL_ANCESTRY_CONFIRMED')]==12
    assert phases[('context_projection','DUMMY_ANCESTRY_CONFIRMED')]==3 and phases[('context_projection','PROPOSAL_ANCESTRY_CONFIRMED')]==36
    unique_moe={(x['task']['taskId'],x['task']['startNs'],x['task']['endNs']) for x in raw['moe_anchors']}
    assert len(raw['moe_anchors'])==8 and len(unique_moe)==6
    for x in raw['moe_anchors']:
        assert len(x['api'])==1 and x['api'][0]['name']=='aclnnDispatchFFNCombine'
        assert any('moe_comm_method.py(591)' in f['frame'] for f in x['api'][0]['selected_frames'])
    formal={g['role']:g for g in observed['groups'] if g['phase']=='formal_C8_measure'}
    assert formal['target']['cohorts']==80 and formal['draft']['cohorts']==7
    repeat_model=[dict(global_rows=t,TP=8,local_rows=(t+7)//8,
                       original_repeat_output_bytes=t*4*5120*2,
                       shard_then_repeat_output_bytes=((t+7)//8)*4*5120*2,
                       original_pad_output_bytes=((t+7)//8)*8*4*5120*2,
                       pad_single_channel_output_bytes=((t+7)//8)*8*5120*2) for t in [20,15]]
    source_specs=[
        (S/'models/deepseek_v41/dspark.py',[(55,68),(223,276),(284,310)],'fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601'),
        (S/'models/deepseek_v41/model.py',[(884,899)],'101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91'),
        (S/'ops/fused_moe/moe_comm_method.py',[(573,615)],'7040f691355bd9acb3c668e698de3f7d3854edbc22e3f979f9b03d99800364d7'),
        (S/'ops/rope_dsv4.py',[(382,402)],'9cbcd2cb2a1123dcbf5afc3999025035d155dfd3b160f186c36aebb5aa7c706b'),
        (S/'models/common/ops/sequence_parallel.py',[(32,64),(104,152)],'73bcf07050bce8b049dd0bbdb0a7504f6eba157a1e97a5033849ef31ef05fcb0'),
        (W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py',[(407,436)],None),
    ]
    sources=[]
    for p,ranges,pin in source_specs:
        b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();lines=b.decode().splitlines()
        assert pin is None or sha==pin
        assert all(1<=start<=end<=len(lines) for start,end in ranges)
        sources.append(dict(path=str(p),sha256=sha,reviewed_ranges=ranges,
                            excerpts=[dict(start=start,end=end,lines=[f'{i+1}: {lines[i]}' for i in range(start-1,end)]) for start,end in ranges],
                            historical_loaded_native_binary_attested=False))
    contracts={
      'F21':dict(profile=dict(target=moe,draft=a['F21_draft'],selected_anchor_checks=8,distinct_selected_tasks=6),
                 recorded_branch='Selected six distinct169tasks submit atmoe_comm_method591; in pinned code this is nonMega enable_fused_mc2=1 native dispatch_ffn_combine. Count output is preallocated self.expert_token_nums.',
                 observed166_formal_counts=formal,
                 domains='169task timing and166same-call counts remain separate; formal166target M400/1023 differs from old169M643/1023.',
                 candidates=['Reorder existing compound internals: dispatch/GMM1 overlap already exists; investigate activation/GMM2 readiness, cumsum publication and microbatch pipelines only under proper participant/memory protocols.',
                             'Use actual expert/rank row distributions for row/group scheduling or placement candidates; preserve logical expert weights/IDs, routing probabilities, capacity and collective agreement.'],
                 constraints=['Keep GMM1/activation/GMM2 shared-workspace last-use ordering and UB/event ownership from existing native contract.',
                              'Quantized activation row and its entire per-token scale must both be ready; partial2304D maxima cannot replace full row scale.',
                              'Observed zero expert rows do not make Set/Wait/SyncAll removable; all producers/consumers must agree on any new protocol.',
                              'Count buffer is reused across up to40target layers in166; delayed consumers require correct version/lifetime.'],
                 conditional_work_bound=dict(status='CONDITIONAL_SYMBOLIC_CEILING',domain='Direct dense independent-assignment twoGMM algorithm, no sparsity/result reuse or alternative algorithm; matchingINT8 MAC resource domain only.',
                     MAC_per_assignment=5120*4608+2304*5120,
                     expression='T_compute >= max_r(n_receiver_rows_r * 35389440 / P_MAC_upper_r)',
                     comparison='UNKNOWN:166same-call stage time and matching hardware capacity cap not supplied; cannot divide old169time by this bound.',
                     ratios='Receiver max/mean count is a workload lower-bound ratio in equal-capacity model, not measured efficiency or attainable speedup.'),
                 open=['Same-call native dispatch/GMM/activation/combine and flag timing; old task-based capture has pcSampling off.',
                       'Actual loaded native build/tiling/scale and full all-iteration branch coverage; no attribution of lowMAC proxy to one cause.']),
      'F28':dict(profile=a['F28'],phase_ancestry=ancestry['summary'],embedding_collective=communication['summary']['per_role']['F28_embedding'],
                 value_mapping_review='Pinned sp_shard zero-pads ONLY dim0 then slices thisTP rank; sp_padding_mask depends on global token count only, padsTrue, same row partition. Therefore shard(repeat(E)) and repeat(shard(E)) have the same token/channel values under these helper contracts.',
                 primary_variant='Keep global-token mask preparation, SP-shard single-channel embedding, then repeat only local rows; keep matching input_ids shard, premix and initial residual for firstHCpost.',
                 copy_model=repeat_model,copy_model_limits='Logical tensor output bytes for selected shapes, not observed HBM traffic/peak-live allocation/time or a necessary lower bound.',
                 crosscard_variant='For the observed standard masked-vocabulary SUM path followed bySP, investigate coupled ReduceScatter producingSP-local embedding instead of AllReduce(full rows) then shard; keep compatible reduction/SP groups, row order/padding and numerical contract. ExistingPCP exchange is a separate source branch and not a new saving.',
                 constraints=['Target already shards before HC repeat; do not count the draft reorder as a target gain.',
                              'SP helper clones no-pad slices for functional custom-op alias rules; preserve equivalent alias/contiguity/lifetime and graph output/address contract.',
                              'Full initial four-channel residual is still needed by HCpost; do not equate eliminating global repeat with eliminating all residual storage.',
                              'Onehot y bypass and folded coefficient weights remain separate numerical candidates requiring HF32/FP32/BF16/order/special-value proof.'],
                 open=['Exact reduction-group versusSP-group ownership for collective replacement, and collective numerical equivalence.',
                       'Historical storage/graph event generations and empty/no-SP branches; proposal ancestry is not formal client-step identity.']),
      'F29':dict(correct_scope='Context KV projection/norm/RoPE/cache and request-index reuse, NOT logits/Markov.',profile=a['F29'],phase_ancestry=ancestry['summary'],
                 observed_pipeline='39bound chains: BF16 [T,5120]x[512,5120] -> BF16kv512 -> full512D RMSNorm -> nativeRotaryPositionEmbeddingV2 on64tail withBF16x/cos/sin -> Cat448+64 -> INT32pair ScatterNdUpdateSk intoBF16cache.',
                 candidates=['Keep full512D norm/rounding and interleaved64D rotation, then form actual cache row/slot directly instead of private cat/contiguous intermediate where unobserved.',
                             'Reuse context-position cos/sin query results across this call’s draft layers under one immutable position/table version;156Index records exist in39projection groups,4each. Filter to requiredconfig/layers only after mapping proof.',
                             'Pack independent per-layer KV projections into wider output channels OR pipeline per-layer preparation with downstream draft work; their whole-output-ready and resource effects overlap.',
                             'Per-layerNone/empty slot returns from store after projection; a well-formed inference-only skip can be considered if no required side effects/other consumer, while preserving dummy/profile allocation purposes.'],
                 constraints=['Each layer keeps its own weights,512D norm and cache. Do not normalize1536combined features or share distinct per-layer KV values.',
                              'NativeRoPE branch differs from target InplacePartialRotaryMul; preserve actualBF16cos/sin and norm output boundaries and in-place y.copy_ view effects or prove private.',
                              'Use true block_size, cache page stride, slot/position versions, negative-skip and duplicate-write contract; same shape/group size does not identify storage.',
                              'Context and current-query writes meet at SWA cache. Noncausal same-request reads can require later query KV, so causal-prefix readiness is insufficient.',
                              'Dummy context3 and proposal context36 are distinct populations; a no-op cache write does not justify deleting dummy memory/profile work.'],
                 open=['Actual slots, layer-group alias mapping, position/config identities and cross-iteration cache versions.',
                       'Loaded native numeric/ready protocol and complete request/phase/resource coverage; no measured fusion or communication gain.']),
      'F30':dict(profile_model_calls=a['F30'],phase_ancestry=ancestry['summary'],
                 base_head_collective=communication['summary']['per_role']['F30_base_logits'],
                 association_correction='TASK/COMPUTE subset contains1base-headAllGather but directCOMM linking proves13. Do not turn missing task metadata into collective absence.',
                 retained='Existing collapse/norm, Markov/logits/argmax, RNG/probability/confidence and vocabulary ownership contracts remain. Base head1dummy+12proposal; Markov65total includes mixed ancestry not separately phase-counted here.',
                 open=['Proposer add/argmax/rejection and dynamic confidence branches are outside this DSpark-model-frame-only task selection; previous gap evidence stays scoped.',
                       'Formal client/iteration, complete active consumers and numerical equivalence remain separate; no F30full module closure.'])
    }
    now=datetime.now(timezone.utc).isoformat()
    data=dict(generated_utc=now,contracts=contracts,source_provenance=sources,source_gate=str(O/'MOE-DRAFT-CODE-QUESTION-GATE-v7.json'),
              profile_input_paths=[str(N/n) for n in ['archived-ep3-dspark-moe-evidence166-v7-v2.json','archived-ep3-dspark-pipeline-audit166-v7-v2.json','observed-moe-counts-evidence166-v7.json','archived-ep3-dspark-communications166-v7.json','archived-ep3-dspark-phase-ancestry166-v7.json','archived-ep3-embedding-frames166-v7.json']],
              full_server_results=[raw['full_result'],audit['full_result'],observed['full_result'],communication['full_result'],ancestry['full_result']],
              error_recovery=['Initialsource-chain guard118287>50000 stopped before device association;v2countsallhost and retains onlydevice-linkedAPI; rawDBunchanged.',
                              'Firstsaved-JSON parser assumedTileline258; actualTileispartofline249. Replaced guessed line with operator+observedforwardstack selection; noDBrequery for parser retry.'],
              F29_scope_erratum='v6pending prose incorrectly calledF29draftlogits/Markov. Canonical inventory andv7correctly separateF29contextKV andF30logits/Markov.',
              new_device_or_inference_work=False,repository_wide_source_audit=False,whole_modules_closed=0,overall_complete=False)
    (O/'moe-draft-reconciliation-v7.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    ledger=json.loads((O/'candidate-profile-ledger-v6.json').read_text())
    states=dict(F21='ALL_RETAINED_COMPOSITES_SELECTED_NATIVE_BRANCH_AND_SEPARATE166COUNTS_REVIEWED',F28='ACTUAL_DRAFT_REPEAT_SP_INIT_HC_AND_DUMMY_PROPOSAL_ANCESTRY_BOUND',
                F29='ACTUAL_CONTEXT_KV_NATIVE_ROPE_CACHE_PIPELINES_AND_PHASES_REVIEWED',F30='MODEL_HEAD_MARKOV_AND_COMM_BOUND_PROPOSER_ENDPOINTS_STILL_PARTIAL')
    for c in ledger['candidates']:
        if c['id'] in contracts:
            c.update(profile_review_state=states[c['id']],scoped_detail=contracts[c['id']],conclusion_changing_gap='; '.join(contracts[c['id']]['open']),
                     target_source_gate=str(O/'MOE-DRAFT-CODE-QUESTION-GATE-v7.json'),candidate_closed=False,legal_fusion_proved=False,
                     next_profile_or_contract_action='Only explicitly open/phase/consumer/resource questions require further evidence. No broad code audit, no inference of legality from adjacency or time savings from task gaps.')
            c.setdefault('evidence_files',[]).append('moe-draft-reconciliation-v7.json')
    ledger.update(generated_utc=now,review_v7_utc=now,new_source_read=True,new_target_source_read=True,overall_complete=False,
                  intake_statement='All32previouslylistedcandidates now have scoped profile review; this is not the complete model/framework opportunity universe or branch closure.')
    (O/'candidate-profile-ledger-v7.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    modules=json.loads((O/'module-profile-ledger-v6.json').read_text())
    for m in modules['modules']:
        if m['id'] in ['M05','M06','M12','M13','M15','M16','M17','M18','M19','M20','M21','M22','M23','M24']:
            m['profile_entry_points'].append('moe-draft-reconciliation-v7.json')
            m['review_state']='PARTIAL_TARGET_AND_DRAFT_CALLSITE_PHASE_COMM_AND_COUNT_EVIDENCE; module/phase/branch/resource closure open'
    modules.update(generated_utc=now,new_source_read=True,new_target_source_read=True,overall_complete=False)
    (O/'module-profile-ledger-v7.json').write_text(json.dumps(modules,ensure_ascii=False,indent=2)+'\n')
    gaps=dict(generated_utc=now,module_universe_status='PROVISIONAL26; must verify completeness before declaring allmodule coverage',
              modules_without_candidate_links_in_current_ledger=[dict(id=m['id'],name=m['name'],profile_entry_points=m['profile_entry_points']) for m in modules['modules'] if not m['candidate_links']],
              interpretation='No link in current ledger is neither proof of no optimization nor proof no historical analysis; reconcile existing framework evidence before new source reads.',
              remaining_requirements=['Per-module phase and conditional-branch matrix, including active/inactive/unsupported/unknown with evidence.',
                                      'Cross-operator/layer/card/iteration edge and consumer coverage; derive finite required variants from the declared frozen workload.',
                                      'Same-domain resource bounds/rankings and critical-path reasons; preserve unresolved causal alternatives and identify conclusion-changing evidence.',
                                      'Profile all-type/callsite coverage and framework/prefix/KV/state/response paths beyond existing32model candidates.'],
              candidate32_review_not_completion_percentage=True,whole_modules_closed=0,overall_complete=False)
    (O/'module-coverage-gaps-v7.json').write_text(json.dumps(gaps,ensure_ascii=False,indent=2)+'\n')
    summary=dict(generated_utc=now,target_composites=1280,draft_composites=39,dspark_linked_tasks=3307,linked_native_APIs=raw['dspark_api_count'],
                 selected_MC2_checks=8,distinct_MC2_tasks=6,dspark_context_chains=39,context_Index_records=156,
                 phase_counts=ancestry['summary']['counts'],explicit_HCCL_COMM_links=165,formal166_cohorts=dict(target=80,draft=7),
                 reviewed_candidates=['F21','F28','F29','F30'],pending_family_only_current_pass=[],
                 prior_candidate_count=len(ledger['candidates']),modules_without_candidate_links=[m['id'] for m in gaps['modules_without_candidate_links_in_current_ledger']],
                 targeted_source_files=len(sources),repeat_logical_model=repeat_model,
                 whole_modules_closed=0,overall_complete=False)
    (O/'moe-draft-summary-v7.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
