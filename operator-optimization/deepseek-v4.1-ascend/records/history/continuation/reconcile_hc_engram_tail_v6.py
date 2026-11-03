from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json, hashlib

N = Path(__file__).parent
O = N/'profile-first-20261031'
R = Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
S = R/'whole-scope-20261030/frozen-source-snapshot/vllm-ascend/vllm_ascend'
DBSHA = '084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4'


def receipt(name):
    r = json.loads((N/name).read_text())
    assert r['returncode'] == 0, (name, r['returncode'])
    d = json.loads(r['stdout'])
    assert d['database_sha256'] == DBSHA
    return d


def union(intervals):
    end = None
    total = 0
    for a,b in sorted(intervals):
        if end is None or a > end:
            total += b-a
            end = b
        elif b > end:
            total += b-end
            end = b
    return total


def main():
    hc = json.loads((O/'hc-boundaries-summary.json').read_text())
    assert hc['retained_scopes'] == 1280 and hc['crosslayer']['matched_pairs'] == 1248
    saved = receipt('archived-ep3-hc-boundaries-saved166.json')
    assert saved['source_file_stat_unchanged'] and saved['source_file_sha256'] == 'cc35da2333fb92a982bacfacac4f6323567fc6b5f40e08c3ce7366944a53df05'
    tail = receipt('archived-ep3-hc-tail-saved166.json')
    assert tail['source_sha256'] == '60951d2cc63c75e901a9e4e8b1b88d3bea1cfa75807de4b5234b040d707cecd9'
    comm = receipt('archived-ep3-hc-comm-links166.json')
    assert comm['database_stat_unchanged'] and len(comm['items']) == 4
    for x in comm['items']:
        assert len(x['api_candidates']) == 1 and x['api_candidates'][0]['name'] == 'HcclAllGather'
        assert x['communication']['rankSize'] == 8
        assert x['api_candidates'][0]['connectionId'] in [l['id'] for l in x['links']]
    selected = next(x for x in saved['display'] if x['label'] == 'CROSS_db852bff_L0_L1')
    rows = selected['compute_rows']
    assert len(rows) == 40 and all(len(r['api']) == 1 for r in rows)
    role_counts = Counter()
    for r in rows:
        frames = [f['frame'] for a in r['api'] for f in a['source_frames']]
        role = ('ordinary_gate' if any('/engram/common.py(' in f for f in frames)
                else 'WKV' if any('/engram/layer.py(41)' in f for f in frames)
                else 'parameter_cast_product' if any('/engram/layer.py(53)' in f for f in frames)
                else 'model_copy')
        role_counts[role] += 1
    assert role_counts == dict(ordinary_gate=34, WKV=1, parameter_cast_product=3, model_copy=2)
    selected_account = dict(compute_record_count=40, roles=dict(role_counts),
                            compute_duration_sum_ns=sum(r['endNs']-r['startNs'] for r in rows),
                            compute_interval_union_ns=union([(r['startNs'],r['endNs']) for r in rows]),
                            boundary_interval_ns=selected['interval_ns'][1]-selected['interval_ns'][0],
                            interpretation='Exported compute-record accounting in one interval; residual time is not proved idle/launch overhead or removable request time.')
    cases = []
    for x in tail['items']:
        rs = {r['taskId']:r for r in x['compute_rows']}
        ids = (41605,41610,41613,41614) if 'db852bff' in x['label'] else (45301,45306,45309,45310)
        copy,norm,index,head = [rs[i] for i in ids]
        assert index['op_type'] == 'Index' and norm['op_type'] == 'RmsNorm' and head['op_type'] == 'MatMulV2'
        assert any('model_runner_v1.py(2634)' in f['frame'] for a in index['api'] for f in a['frames'])
        cases.append(dict(label=x['label'],full_HC_copy=copy, final_norm=norm, selected_hidden=index, LM_head=head))
    source_specs = [
        ('models/deepseek_v41/model.py',[(1544,1608),(1798,1802)],'101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91'),
        ('models/deepseek_v41/engram/common.py',[(54,72)],'4797b8962264d137cdb142a9988825288283092fb797ddbd334c6ab2223eb58d'),
        ('models/deepseek_v41/engram/layer.py',[(34,55)],'dc9457c88aa3c90133115bc608447e037b0693f8db71bb8da37f1ead44021f80'),
        ('worker/model_runner_v1.py',[(2048,2110),(2610,2680),(2772,2803),(3129,3135)],None),
        ('models/deepseek_v41/dspark.py',[(272,283)],None),
        ('models/deepseek_v41/vl_model.py',[(281,284)],None),
    ]
    sources = []
    for suffix,ranges,pin in source_specs:
        p = S/suffix
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert pin is None or sha == pin
        lines = raw.decode().splitlines()
        sources.append(dict(path=str(p), sha256=sha, reviewed_ranges=ranges,
                            excerpts=[dict(start=a,end=b,lines=[f'{i+1}: {lines[i]}' for i in range(a-1,b)]) for a,b in ranges],
                            historical_loaded_binary_attested=False))
    contracts = {
      'F02': dict(profile='All32 first-target HC signatures; selected target repeat39646 at1551 and premix initialization39647..39650 at1552-1553, plus exact selected HC role stacks.',
                  variants=['External pre_mix y branch separated from current coefficient branch; retain all coefficients before their consumers.',
                            'Target initial logical HC broadcast/load mapping; target already shards before repeat.',
                            'Initial onehot y bypass/weight-prefold only conditional on numerical contract; ordinary real-number identities do not establish bit equivalence.'],
                  open=['Embedding value identity/first-pre contents and actual binary numerical contract','Draft first-layer entry remains distinct and unbound','Residual post consumer and alias lifetime; no blanket removal of all HC materialization']),
      'F03': dict(profile=hc, selected_role_proof='Two selected EP3 L0 attention-post902 to FFN-pre905 exact TASK/API stacks; configured Engram and observed aux boundaries have intervening work outside layer sidecar.',
                  graph='HCpost R -> optional aux(R) -> optional Engram(R)->Rprime -> next HCpre(Rprime, earlier premix); Rprime remains the later HCpost residual.',
                  constraints=['Preserve BF16 intermediate and original FP32 HC accumulation/normalization order','Retain all residual/coeff outputs and real consumers','Native existing whole-kernel completion does not establish tile-ready visibility','Do not erase observed Engram or aux/gather from crosslayer edges'],
                  open=['Real tensor/alias/event generations','Loaded native HC tiling/binary','Resource and whole-request critical-path impact']),
      'F24': dict(profile='Four selected boundary intervals each40 compute records, including lookup SP row copy; selected leading transpose at embedding354 and fixed buffer copy at model1221. Prior exact four named Engram kernel classes and ready/stride audits reused.',
                  constraints=['Host-UVA gather+dequant already combined','DP/hash and TP/head exchange plus per-layer/mask ready must remain where required','Direct joint lookup must not postpone a needed per-layer publication without equivalent interface','Stride-aware ID view requires source lifetime/offset/producer-reader ordering; snapshot clone may isolate a changing producer'],
                  open=['Exact historical stride/alias/generations absent in checked raw metadata','Whole lookup/preparation module traffic and critical path not given by four named kernel classes']),
      'F25': dict(profile=selected_account, observed_variant='Selected ordinary rotated gate is decomposed into34 exported compute records; parameter2casts+product3, WKV1, modelcopies2 separately accounted.',
                  math='original=unrotate(hidden.float()); original/key full-feature RMS and weighted dot feed signed-sqrt/sigmoid/mask; residual adds ORIGINAL hidden.float(), not unrotated original.',
                  variants=['Group gate casts/reductions/scalar nonlinearities/residual while preserving precision/rounding and full5120D dependence',
                            'WKV output tiles into key norms/dot and shared value consumers; preserve BF16 WKV output semantics',
                            'Reuse FP32 q*k and rotation conversion under immutable/versioned weights; repeated physical parameter casts/product are present in selected invocation'],
                  constraints=['Preserve rotation direction and original-vs-residual basis','Keep both key and hidden norms; dot alone is insufficient','Preserve multiply order, FP32 intermediate rounding, epsilon, signedzero/NaN, sigmoid, mask0*Inf semantics','Shared value has4HC consumers; do not overwrite after first'],
                  open=['Native numerical/compiler contract and tile interface','Actual immutable parameter lifecycle across graph/weight updates','Unexported activity and causal benefit;88ms boundary is not an obtainable saving']),
      'F26': dict(profile_cases=cases, collective_proof=[x for x in comm['items'] if x['label'].startswith('MTP_')],
                  observed='Selected full HC copies and collapse→BF16→finalRMSNorm bound directly; full norm8184/4760rows followed by logits selection3/14rows and bound LMhead.',
                  reject_unconditional_logits_only='Full normalized hidden is saved in ExecuteModelState; prompt logprobs and non-aux drafter branches can read it. Stored copy alone is not a proof of live use either.',
                  conditional_variants=[
                    'Push row selection before collapse/norm for the UNION of all active consumer rows, only when unselected dense outputs/aliases are unobservable; preserve BF16 boundary, indices, duplicates, order, padding and graph contract.',
                    'If raw MTP buffer is a live required consumer, retain full HC values; copying and collapse can be separate reading branches with proper readiness/lifetime.',
                    'Same selected tail uses auxiliary3way DSpark context while raw MTP buffer is copied. Under aux branch the inspected proposer uses aux context; if the raw getter/API and all other readers are unobserved, conditionally omit raw-MTP production and collapse locally before gathering collapsed hidden. No actual dead-consumer proof is claimed.'],
                  logical_payload_domains=dict(bytes_per_row=dict(full_BF16_HC=40960, FP32_mix=16, collapsed_BF16=10240),
                     raw_HC_live_original=40976, raw_HC_live_replace_mix_with_collapsed=51200,
                     raw_HC_unobserved_local_collapse_then_gather=10240,
                     limits='Conditional dense logical row payload only, not link traffic, necessary lower bound or gain; variants overlap and cannot be added.'),
                  open=['All effective normalized-hidden and raw-MTP/getter consumers for exact workload','Prompt-logprob and non-aux branch conditions and required row identities','Numerical lowering across local/global shape, events and graph output contract']),
      'F27': dict(profile='Selected same heavy29 tail concat41639, contextFC41641 atdspark282, TP8 AllGather COMM264 connection103954→API431557, transpose41644 and full5120D norm41645.',
                  physical_shapes=dict(aux_concat='3x[8180,5120] -> [8180,15360]', local_projection='[8180,15360] -> [8180,640]', gathered_norm='[8180,5120] -> [8180,5120]'),
                  collective=next(x for x in comm['items'] if x['label']=='DSPARK_CONTEXT_FC'),
                  constraints=['main_norm(main_proj(aux)) requires the complete feature row in existing interface','Local640-column normalization is not full5120-column norm','Preserve auxiliary row/version, projection dtype rounding, TP ordering and all needed context outputs','Splitting the15360D projection into3 separately rounded GEMMs is not automatically equivalent'],
                  open=['All-rank tensor/group ownership and timing/resource closure','Tile readiness and numerical contract for any distributed norm or reordered projection'])
    }
    now = datetime.now(timezone.utc).isoformat()
    data = dict(generated_utc=now, database_sha256=DBSHA, hc_profile=hc, contracts=contracts, source_provenance=sources,
                source_gate=str(O/'HC-ENGRAM-CODE-QUESTION-GATE.json'),
                boundary_source_json_sha256=saved['source_file_sha256'], tail_source_json_sha256=tail['source_sha256'],
                controller_recovery='Initial ZCode controller timed out124; independent rc0 status and saved-result read proved remote query completed. No16-window requery performed; v2 full-range task prepared but NOT executed.',
                further_profile_extension='Only two3ms tail windows extended to20ms; four exact collective connections queried.',
                new_device_or_inference_work=False, whole_modules_closed=0, overall_complete=False,
                limits=['Selected submission sites plus frozen-source semantics do not attest historical loaded code, storage or tensor/event identity.',
                        'Candidate profile review maturity is not legal implementation, full-module coverage or completion percentage.',
                        'Conditional variants do not yield speedup or hardware utilization from task sums/gaps.'])
    (O/'hc-engram-tail-reconciliation-v6.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    ledger = json.loads((O/'candidate-profile-ledger-v5.json').read_text())
    states = dict(F02='FIRST_TARGET_REPEAT_AND_HC_ROLE_STACKS_REVIEWED_DRAFT_SEPARATE',F03='ALL_RETAINED_HC_BOUNDARIES_AND_SELECTED_INTERVENING_CONSUMERS_REVIEWED',
                  F24='SELECTED_PREP_ROW_COPY_AND_PRIOR_LOOKUP_READY_STRIDE_EVIDENCE_RECONCILED',F25='SELECTED_ROTATED_ORDINARY_GATE34TASKS_AND_WKV_STACKS_BOUND',
                  F26='SELECTED_FINAL_NORM_ROW_SELECTION_MTP_AND_CONSUMER_BRANCHES_REVIEWED',F27='SELECTED_AUX_CONCAT_CONTEXT_FC_TP8_COLLECTIVE_AND_NORM_BOUND')
    for c in ledger['candidates']:
        if c['id'] in contracts:
            c['profile_review_state'] = states[c['id']]
            c['scoped_detail'] = contracts[c['id']]
            c['conclusion_changing_gap'] = '; '.join(contracts[c['id']].get('open', []))
            c['evidence_files'].append('hc-engram-tail-reconciliation-v6.json') if 'evidence_files' in c else c.update(evidence_files=['hc-engram-tail-reconciliation-v6.json'])
            c['target_source_gate'] = str(O/'HC-ENGRAM-CODE-QUESTION-GATE.json')
            c['next_profile_or_contract_action'] = 'Only the stated open items require further evidence; no broad source audit. Preserve branch-specific conditional variants and do not count timing gaps as gain.'
            c['candidate_closed'] = False
            c['legal_fusion_proved'] = False
    assert len(ledger['candidates']) == 32
    ledger.update(generated_utc=now, review_v6_utc=now, new_source_read=True, new_target_source_read=True, repository_wide_audit=False, overall_complete=False)
    (O/'candidate-profile-ledger-v6.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    modules = json.loads((O/'module-profile-ledger-v5.json').read_text())
    for m in modules['modules']:
        if m['id'] in ['M05','M06','M14','M15','M16','M18','M19','M22','M23','M24']:
            m['profile_entry_points'].append('hc-engram-tail-reconciliation-v6.json')
            m['review_state'] = 'PARTIAL_HC_ENGRAM_TAIL_AND_AUX_CONTEXT_BINDINGS; branch/resource/module closure remains open'
    modules.update(generated_utc=now,new_source_read=True,new_target_source_read=True,repository_wide_audit=False,overall_complete=False)
    (O/'module-profile-ledger-v6.json').write_text(json.dumps(modules,ensure_ascii=False,indent=2)+'\n')
    summary = dict(generated_utc=now,retained_scopes=1280,intra_HC_pairs=1280,crosslayer_HC_pairs=1248,
                   selected_Engram_boundary_windows=4,selected_ordinary_gate_records=34,selected_Engram_account=selected_account,
                   selected_tail_windows=2, tail_rows=[dict(full_norm_rows=n,logits_rows=k) for n,k in [(8184,3),(4760,14)]],
                   explicit_collective_links=4,reconciled_candidates=list(contracts),pending_family_only_current_pass=['F21','F28','F29'],
                   new_source_files_targeted=len(sources),repository_wide_audit=False,controller_timeout_recovered=True,
                   whole_modules_closed=0,overall_complete=False)
    (O/'hc-engram-tail-summary-v6.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
