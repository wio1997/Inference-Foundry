#!/usr/bin/env python3
"""Pin terminal readers, build a non-deployed dead-bank patch and reprice archives."""
import ast
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OUT=Path('audit/execution-search')
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
FROZEN=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot')
SUPPLEMENT=FROZEN.parent/'sampler-source-supplement/frozen-source-snapshot'
sys.path.insert(0,str(BASE/'tools'))
PINS={}

def read(p):
    raw=Path(p).read_bytes();PINS[str(p)]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def patch_model(p):
    original=p.read_text();candidate=original
    assert '_gather_hc_before_collapse' not in original
    replacements=[
        ('        self._mtp_hidden_buffer = (\n',
         '        # Keep the original full-row collapse geometry independently of storage.\n'
         '        self._gather_hc_before_collapse = get_pp_group().is_last_rank and needs_mtp_hidden_states\n'
         '        self._mtp_hidden_buffer = (\n'),
        ('            if get_pp_group().is_last_rank and needs_mtp_hidden_states\n',
         '            if self._gather_hc_before_collapse and spec_config.method != "dspark"\n'),
        ('        # MTP needs full HC states\n        if self._mtp_hidden_buffer is not None:\n',
         '        # Preserve original collectives and numerical geometry for DSpark.\n'
         '        if self._gather_hc_before_collapse:\n'),
        ('            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))\n',
         '            if self._mtp_hidden_buffer is not None:\n'
         '                self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))\n'),
        ('        if use_sequence_parallel and self._mtp_hidden_buffer is None:\n',
         '        if use_sequence_parallel and not self._gather_hc_before_collapse:\n')]
    for a,b in replacements:
        assert candidate.count(a)==1,a
        candidate=candidate.replace(a,b)
    before=ast.parse(original);after=ast.parse(candidate)
    def methods(tree):
        return {(c.name,f.name):ast.dump(f,include_attributes=False) for c in tree.body if isinstance(c,ast.ClassDef) for f in c.body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
    bm,am=methods(before),methods(after)
    changed=[list(k) for k in bm if bm[k]!=am[k]]
    assert changed==[['DeepseekV41Model','__init__'],['DeepseekV41Model','forward']],changed
    rel='vllm_ascend/models/deepseek_v41/model.py'
    artifact=OUT/'terminal-dspark-dead-bank.patch'
    artifact.write_text(''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel)))
    return dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),base_sha256=hashlib.sha256(original.encode()).hexdigest(),changed_methods=changed,parse_valid=True,deployed=False,new_NPU_bit_gate=False,
                preserved=['old full HC AllGather and pre_mix AllGather, order and shapes','original full-M collapse: FP32 products/reduction then BF16','original full-M RMSNorm and selected target head','all DSpark aux/context/cache paths','all non-DSpark allocation/forward branches'],
                removed=['only DSpark bank allocation and dead copy'],
                qualification='Source-level dead-storage removal. Not a runtime bit-exact/graph/memory-lifetime acceptance; no service changes.')

def target_score_price():
    from price_vocab_selection_fusion import records,RECEIPT,cached,ORDINARY,GUARDS
    from price_attention_send_epilogue import ENDPOINTS
    from price_draft_movable_arrivals import latest_change
    waves,hist,schedule,_,_=records()
    receipt=read(BASE/RECEIPT)
    cache=Path(receipt['cache']['path']);cache=cache if cache.is_absolute() else BASE/cache
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==receipt['cache']['sha256']
    source={};groups=Counter()
    for rank,p in enumerate(cached(RECEIPT)):
        assert rank==p['rank']
        for role,stream,m,rows,g in p['calls']:
            if role!='target':continue
            h=rows[0];sig=p['regimes'][h[3]]
            assert sig==['MatMulV2',f'"{m},5120;16160,5120"',f'"{m},16160"','DT_BF16;DT_BF16','DT_BF16']
            source[rank,rows[-1][2]]=dict(full_us=(rows[-1][2]-h[1])/1000,head_us=(h[2]-h[1])/1000,first_task=h[0],last_task=rows[-1][0],M=m)
            groups[rank//8,m]+=1
    native=read(BASE/ENDPOINTS)['endpoint_ms'];offsets=hist['clock_contract']['offset_upper_ns']
    prices={};samples={}
    for route in ['whole_target_head_verify_free','head_GEMM_free','all_after_head_free']:
        per={metric:[[0.]*8,[0.]*8] for metric in native}
        for n,ranks in waves.items():
            if not any(w.get('scheduled') for w in ranks.values()):continue
            timing=[w['entry'] for w in ranks.values() if w.get('scheduled')]
            credits=[]
            for rank,w in sorted(ranks.items()):
                if 'target' not in w['calls']:
                    credits.append(0.);continue
                c=source[rank,w['calls']['target']['end']]
                value={'whole_target_head_verify_free':c['full_us'],'head_GEMM_free':c['head_us'],'all_after_head_free':c['full_us']-c['head_us']}[route]
                credits.append(value*1000)
                if route=='whole_target_head_verify_free' and w['calls']['target']['stats']['b']==1:
                    samples.setdefault(c['M'],[]).append(c['full_us'])
            if n+1 in waves:
                starts=[waves[n+1][i]['start'] for i in range(16)]
                changes=latest_change(starts,[round(s-v) for s,v in zip(starts,credits)],offsets)
            else:changes=[min(credits)/1e6,max(credits)/1e6]
            for metric in native:
                for i,client in enumerate(schedule['client']):
                    deadline=client['end_ns'] if metric=='makespan' else client['nonempty_chunk_ns'][0]
                    if min(timing)>deadline:continue
                    for j in [0,1]:per[metric][j][i]+=changes[j]
        prices[route]={}
        for metric,gains in per.items():
            saving=[max(native[metric])-max(old-g for old,g in zip(native[metric],v)) for v in gains]
            individual=[[100*g/old for g,old in zip(v,native[metric])] for v in gains]
            prices[route][metric]=dict(saved_ms=saving,percent=[100*x/max(native[metric]) for x in saving],per_request_ms=gains,maximum_request_percent=max(x for v in individual for x in v))
    import statistics
    median={m:statistics.median(v) for m,v in samples.items()}
    assert 1 in median and 6 in median
    ordinary=[];guards=read(BASE/GUARDS)
    for r in read(BASE/ORDINARY)['ordinary_requests']:
        ctrl,=[x for x in guards['stages'][r['stage']]['requests'] if x['name']==r['request']]
        faster=min(ctrl['stock_elapsed_ms'],ctrl['restored_stock_elapsed_ms'])
        ms=(median[1]+r['draft_verifications']*median[6])/1000
        ordinary.append(dict(request=r['request'],verification_steps=r['draft_verifications'],conditional_response_ms=ms,percent_of_faster_reference=100*ms/faster))
    return dict(C8=prices,target_calls_by_DP_M=[[dp,m,n] for (dp,m),n in sorted(groups.items())],B1_shape_matched_full_us_medians=median,six_ordinary=ordinary,
                qualification='All head GEMM, collective, score artifacts, argmax/reject and internal post-head gaps free; original schedule/next-target frontier/clock bounds. Excludes separate valid-count/seed/async-copy/next-input/host plan. Conditional search budget, not realizable expected speedup.',source_cache=dict(path=str(cache),sha256=receipt['cache']['sha256']))

def ordinary_terminal_free():
    from collections import defaultdict
    from price_terminal_head_delivery import load_windows,ORDINARY,GUARDS
    from price_oa_ob_fusion import interpolate
    import statistics
    waves,_,_=load_windows();eager=defaultdict(list);graph=defaultdict(list)
    for ranks in waves.values():
        for dp in [0,1]:
            first=ranks[dp*8]
            if not first.get('scheduled'):continue
            ts=[ranks[i]['terminal'] for i in range(dp*8,dp*8+8)]
            values=[min(t.post_to_head_us for t in ts),max(t.post_to_head_us for t in ts)]
            if first['kind']=='E':eager[dp,ts[0].rows].append(values)
            elif first['calls']['target']['stats']['b']==1:graph[dp].append(values)
    guards=read(BASE/GUARDS)['stages'];out=[]
    for r in read(BASE/ORDINARY)['ordinary_requests']:
        dp=r['dp'];q=r['eager_rank_rows']
        refs=sorted((n,[statistics.median(v[i] for v in vals) for i in [0,1]]) for (d,n),vals in eager.items() if d==dp)
        xs=[n for n,_ in refs];prompt=[interpolate(q,xs,[v[i] for _,v in refs]) for i in [0,1]]
        draws=graph[dp] or graph[1-dp];decode=[statistics.median(v[i] for v in draws) for i in [0,1]]
        ctrl,=[v for v in guards[r['stage']]['requests'] if v['name']==r['request']]
        result={}
        for label,field in [('first_nonempty','ttft'),('response_end','elapsed')]:
            n=ctrl['observed_target_graph_starts_since_request_arrival'][label]['maximum_on_one_rank']
            ms=[(a+n*b)/1000 for a,b in zip(prompt,decode)]
            faster=min(ctrl['stock_'+field+'_ms'],ctrl['restored_stock_'+field+'_ms'])
            result[label]=dict(ms=ms,percent_of_faster_reference=[100*x/faster for x in ms],observed_target_starts=n)
        out.append(dict(request=r['request'],dp=dp,local_prompt_rows=q,entire_post_to_head_free=result))
    return dict(cases=out,maximum_percent=max(x for r in out for v in r['entire_post_to_head_free'].values() for x in v['percent_of_faster_reference']),qualification='Conditional prompt interpolation within same-DP archived eager rows; B1 graph brackets and original observed start counts since arrival, zero replacement. Not observed ordinary kernel timings or a changed-schedule bound.')

def main():
    identity=read(OUT/'terminal-execution-identity.json')
    inventory=dict(read(identity['inventory_path']))
    source=[]
    specifications=[
      (FROZEN,'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py',[[969,989],[1555,1608],[1798,1802]]),
      (FROZEN,'vllm-ascend/vllm_ascend/models/deepseek_v41/vl_model.py',[[283,284]]),
      (FROZEN,'vllm-ascend/vllm_ascend/worker/model_runner_v1.py',[[198,203],[999,1057],[1320,1395],[1430,1578],[1578,1660],[1848,1899],[2038,2120],[2610,2678],[2763,2819],[2854,2908],[2926,2970],[2973,2994],[3028,3134],[6520,6557]]),
      (FROZEN,'vllm/vllm/config/speculative.py',[[1909,1913]]),
      (FROZEN,'vllm/vllm/v1/worker/gpu_model_runner.py',[[289,395],[1358,1438],[1775,1924],[5643,5750]]),
      (FROZEN,'vllm/vllm/v1/worker/gpu_input_batch.py',[[1026,1108]]),
      (FROZEN,'vllm/vllm/v1/core/sched/async_scheduler.py',[[1,70]]),
      (FROZEN,'vllm/vllm/v1/core/sched/scheduler.py',[[1937,2046],[2347,2399]]),
      (FROZEN,'vllm/vllm/v1/engine/core.py',[[630,736]]),
      (FROZEN,'vllm/vllm/v1/engine/output_processor.py',[[286,350],[629,772]]),
      (FROZEN,'vllm/vllm/v1/engine/detokenizer.py',[[69,167],[310,362]]),
      (FROZEN,'vllm/vllm/v1/core/sched/utils.py',[[94,135]]),
      (FROZEN,'vllm-ascend/vllm_ascend/spec_decode/utils.py',[[61,128]]),
      (SUPPLEMENT,'vllm-ascend/vllm_ascend/sample/rejection_sampler.py',[[1,60],[378,685]]),
      (SUPPLEMENT,'vllm-ascend/vllm_ascend/ops/triton/reject_sample.py',[[24,309],[343,471]])]
    for root,name,ranges in specifications:
        p=root/name;digest=hashlib.sha256(p.read_bytes()).hexdigest()
        runtime='/vllm-workspace/'+name
        assert inventory[runtime]==digest,(runtime,digest,inventory.get(runtime))
        source.append(dict(local_path=str(p),runtime_path=runtime,sha256=digest,read_ranges=ranges))
    source[7]['read_ranges'].extend([[2540,2609],[2998,3033]])
    # Newly retrieved missing executor source is pinned to the same runtime scan.
    for row in read(OUT/'terminal-runtime-source/manifest.json'):
        if row['path'].endswith('/worker_base.py'):continue
        p=Path(row['local_path']);digest=hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest==row['sha256']==inventory[row['path']]
        ranges=([[968,1039]] if p.name=='multiproc_executor.py' else [[50,114]] if p.name=='kv_connector_model_runner_mixin.py' else [])
        source.append(dict(local_path=str(p),runtime_path=row['path'],sha256=digest,read_ranges=ranges,reviewed_scope='AST override names only; output thread methods are inherited' if p.name=='patch_multiproc_executor.py' else 'listed ranges'))
    patch=patch_model(FROZEN/specifications[0][1])
    # Re-run exact task-ID, shape, stream and scheduler joins, without writing old evidence.
    from price_terminal_sparse_ffn import sections,analyze
    from price_terminal_head_delivery import build
    sparse=analyze()
    terminal=build()
    target=target_score_price()
    ordinary_free=ordinary_terminal_free()
    saved=read(BASE/'full-review/TERMINAL-HEAD-DELIVERY-ESTIMATES.json')
    assert terminal['C8_saved_ms']==saved['C8_saved_ms']
    assert terminal['screen']==saved['screen']
    # Save bounded rederived prices, with archived input pins rather than old conclusions alone.
    (OUT/'terminal-cost-replay.json').write_text(json.dumps(dict(C8_reference_ms=terminal['C8_reference_ms'],C8_saved_ms=terminal['C8_saved_ms'],ordinary=terminal['ordinary'],ordinary_whole_post_to_head_free=ordinary_free,replacement=terminal['replacement'],screen=terminal['screen'],limits=terminal['limits'],source_pins=terminal['sources'],target_score=target,sparse_FFN=dict(row_totals=sparse['row_totals'],request_bounds=sparse['request_bounds'],stage_sensitivity=sparse['stage_sensitivity'],semantic_limits=sparse['semantic_limits'],source_pins=sparse['sources']),new_NPU_work=False,new_profile_exports=0,whole_chain_complete=False),ensure_ascii=False,indent=2)+'\n')
    native={};qualifications=[]
    for name in ['HC-TERMINAL-DELIVERY.json','TERMINAL-COMPACT-FFN-SHADOW.json','TERMINAL-DENSE-COEFFICIENT-SHADOW.json','TERMINAL-NUMERIC-REPAIR-BUDGET.json']:
        d=read(BASE/'full-review'/name)
        native[name]={k:d[k] for k in ['status','native_comparisons','post_rounding_counterexample','stock_substitutions','branches','divergent_calls','per_branch','conclusion','limits','source_pins','sources'] if k in d}
        if name=='HC-TERMINAL-DELIVERY.json':
            current={s['runtime_path']:s['sha256'] for s in source}
            same={k:v for k,v in d['current171_source_hashes'].items() if k in current}
            assert len(same)>=2 and all(current[k]==v for k,v in same.items())
            native[name]['current_source_SHA_matches']=same
        if name=='TERMINAL-COMPACT-FFN-SHADOW.json':
            native[name]['cases']={k:{field:v[field] for field in ['shadow_equal','stock_full_exact_outputs','native_worker_steps','differences_summed_over_ranks'] if field in v} for k,v in d['cases'].items()}
        for pin in d.get('source_pins',[]) if isinstance(d.get('source_pins'),list) else []:
            p=Path(pin['path']);p=p if p.is_absolute() else BASE/p
            if p.is_file():
                actual=hashlib.sha256(p.read_bytes()).hexdigest()
                q=dict(report=name,path=str(p),expected_sha256=pin['sha256'],current_sha256=actual,current_matches=actual==pin['sha256'])
                if actual==pin['sha256']:PINS[str(p)]=actual
                else:
                    # Analysis tools can evolve after an archived result; never relabel
                    # the current bytes as the frozen reader. Resolve its exact Git blob.
                    rel=str(p.relative_to(BASE))
                    commits=subprocess.check_output(['git','log','--all','--format=%H','--',rel],cwd=BASE,text=True).splitlines()
                    found=None
                    for commit in commits[:100]:
                        r=subprocess.run(['git','show',commit+':'+rel],cwd=BASE,capture_output=True)
                        if r.returncode==0 and hashlib.sha256(r.stdout).hexdigest()==pin['sha256']:
                            found=commit;frozen=OUT/'terminal-archive-source'/p.name;frozen.parent.mkdir(exist_ok=True);frozen.write_bytes(r.stdout)
                            q.update(historical_commit=commit,historical_path=rel,local_frozen_path=str(frozen),historical_matches=True)
                            PINS[str(frozen)]=pin['sha256'];break
                    assert found is not None,('unresolved historical source pin',q)
                qualifications.append(q)
    (OUT/'terminal-native-archive-qualification.json').write_text(json.dumps(dict(status='HISTORICAL_READER_RESOLVED_WITH_CURRENT_MISMATCH_PRESERVED',pins=qualifications,raw_failure_run='run-20261109T145803Z',qualification='Raw native receipts remain pinned. Changed current reader is not substituted; exact historical blob recovered from Git without executing it.',new_NPU_work=False),ensure_ascii=False,indent=2)+'\n')
    (OUT/'terminal-native-evidence-reuse.json').write_text(json.dumps(native,ensure_ascii=False,indent=2)+'\n')
    d=dict(status='PARTIAL_TERMINAL_READER_AND_VALUE_REVIEW_LOOP014_ACTIVE',introduction='Full HC -> bank vs normalized rows -> head/prompt scores/DSpark aux -> verification -> async output -> next iteration',
      precision='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',source_pins=source,reference_inventory=dict(files=identity['scanned_file_count'],matched_files=len(identity['hits']),path=identity['inventory_path'],sha256=identity['inventory_sha256']),
      bank=dict(status='待确认',source_feasibility='Dead DSpark allocation/copy removal is feasible while preserving full numerical geometry. Patch concrete and parse checked, not device accepted.',bytes_per_rank=8192*4*5120*2,MiB_per_rank=320,GiB_16_ranks=5,capacity_kind='source allocation; not allocator-measured released memory or capacity increase',patch=patch,
          active_readers=['V1 propose calls getter but consumes return only for method=mtp; frozen DSpark takes aux'],inactive_other_readers=['upstream GPUModelRunner.propose is overridden by NPU V1','new GPU runner/speculator and NPU V2 are different runners','other architecture/model implementations and tests do not consume this active model instance'],
          gaps=['Isolated runtime qualification of patch against current native graph addresses/allocator/lifetime; no additional changed-M math needed','current service memory high-water and capacity effect, if deciding release/development priority']),
      row_demand=dict(status='待确认',head='ordered logits_indices including every proposed candidate that target must verify plus bonus; no rejection-dependent pruning before verification',prompt='if num_prompt_logprobs nonempty, consume full scheduled prompt hidden rows including chunk boundaries; handle separately or fall back',dspark='aux comes from input to target zero-based layers 36/37/38; final layer39 output is not DSpark main context',other_guards=['true MTP requires full raw HC bank','PP/intermediate/broadcast and pooling are separate legal branches','EPLB/export/dump callbacks and graph padding/collective participants must be retained or guarded'],
          stopped_direct_recipes=['stock compact final FFN changes original mHC coefficients/routing weights (prior native witnesses)','retaining dense coefficients still produced one expert-combined/HCpost bit difference in a heavy branch, even when normalized readout was equal','changing collapse reduction tree has native 1/2-element witnesses; no final-post BF16 rounding may be removed'],
          open_larger_scope='last-layer attention query/output/FFN row-demand producer, preserving all K/V/index/cache writes and numerical geometry; not rejected by post-to-head budget'),
      costs=dict(path='terminal-cost-replay.json',C8=terminal['screen'],ordinary_whole_post_to_head_free_max_pct=ordinary_free['maximum_percent'],target_head_verify_C8_max_individual_pct=target['C8']['whole_target_head_verify_free']['makespan']['maximum_request_percent'],target_head_verify_ordinary_max_pct=max(x['percent_of_faster_reference'] for x in target['six_ordinary']),copy_only='dead bank copy is a strict subset of all-free post-to-head; do not extrapolate 5GiB capacity into latency',larger_sparse_FFN_plus_readout_free_P90_ms=sparse['request_bounds']['selected_FFN_and_full_readout_fixed_joins_ms']['max_P90_credit_ms'],qualification='Original request/clock/external schedule sensitivity; not a universal upper bound or expected speedup. 1% is ranking reference, not user accuracy/performance acceptance.',same_budget_small_variants='已停止: independent 1% latency budget for terminal Post->head input and separate target head/verify. Larger last-layer, next-input/host scheduling and capacity budgets are distinct.'),
      verification=dict(status='PARTIAL',greedy='K1 specialized and variable K kernels inspected; first mismatch emits target argmax then stops, all match appends bonus; prefilled -1 suffix retained; K0 appends bonus in general kernel',random='inspected ordinary per-token kernel and recovered vocab loop; preserve p/q FP32, uniform/exponential generator mapping and chunk tie order; block verifier and complete RNG/processor producer paths remain open',fusion='Can co-produce accepted prefix count/last seed and output packet after reject decision. A single independent CTA/request can avoid count rescan, but cannot read other CTAs before join; output D2H/CPU stop/next-context consumers still distinct.'),
      asynchronous=dict(status='PARTIAL_SOURCE_BOUNDARIES_READ',already_overlapped='sampled IDs and optional logprobs/NaNs copied on independent stream after wait_stream; event protects device tensor lifetime and CPU parse; DSpark consumes GPU IDs without this CPU copy',histories='CPU -1 optimistic placeholders repaired by previous req-ID mapping and copy event only when logits processors need history; not real accepted token IDs',scheduler='async reserves bonus+draft placeholders, subtracts rejected tokens before output append/stop trim; stale/preempted output must not roll back current generation counters',mamba='need_accepted_tokens only hybrid cache; do not charge inactive hybrid state-update kernel to this target',
          actual_correction='NPU V1 imports vllm_ascend.spec_decode.utils (not upstream helper). It snapshots three advanced gathers before any in-place output: for old draft rows corrected=old_gpu_computed+valid_count; new/no-draft uses current CPU count. previous positions map current req ID to previous row.',
          input_ids='Inherited _prepare_input_ids scatters previous bonus at cu_num_tokens-1-current_draft_len and scheduled prefix of previous draft rows using previous K stride; new/prefill gets CPU source. Stable no-spec contiguous map already has direct slice fast path.',
          CPU_metadata='same rejection drift prev_drafts+1-valid_count subtracted from optimistic CPU seq_len only for participating old draft rows, after existing previous-step count-copy event; then slot mapping uses corrected GPU positions.',
          engine='FIFO result drain follows attempted queue filling; pending structured output defers sample until prior output, abort queue processed before scheduler update. Eliminating CPU output copy does not remove client/stop/grammar/history readers.',
          worker_bridge='Multiproc WorkerProc queues AsyncModelRunnerOutput to separate output thread; enqueue_output calls get_output/event sync, catches exceptions and publishes failure; Ascend executor patch does not override these output methods.',
          client='OutputProcessor runs detokenizer then logprob update and output formatting; final-only/stream_interval can suppress publication. Stop strings choose earliest completion (ties by stop-list order), retain holdback suffix for delta output; frontend-detected stop sends abort to core. CPU IDs may still include an accepted suffix that scheduler stopped/trimmed; preserve original counters and queue semantics.',
          retirement='Scheduler _free_request invokes KV/EC request_finished before freeing encoder state; connector delay/receive-inflight or defer_block_free can retain blocks. finished_sending/recving later retires only correct request. KV runner finalize happens after DSpark forward, wait_for_save then clear metadata; zeroing/reuse cannot race async transfer.',
          packet_candidate='reject may co-emit accepted count and last seed; next input row mapping unknown until next scheduler result, so direct writing future input positions needs stable request-slot/generation plan. In-place permutation correction must preserve snapshot before writes (cannot fuse load/store naively across CTAs).',
          gaps=['full logits processor/RNG/block-verify kernels and actual alternate backend bindings','joint reject-count/next-input host/device price, distinct from now-priced head/reject and DSpark score','KV/EC provider-specific failure/transfer completion and backend cache/slot retirement remain conditional beyond common source interface','larger last-layer attention/FFN row-demand producer including cache writes/exports']),
      next='Follow actual sampler processors/RNG, random block verifier and fallback backend guards; bind joint reject-count/next-input host-device parent before deciding packet fusion, then revisit larger final-layer attention/FFN producer. Common output/stop/connector interface read; provider-specific conditional internals remain open.',
      archive_pins=PINS,analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_archive_qualification='terminal-native-archive-qualification.json',new_NPU_measurements=0,service_requests=0,remote_writes=[],whole_chain_complete=False)
    (OUT/'terminal-execution-frontier.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(source_SHA_matches=len(source),bank_capacity_GiB=5,patch=str(OUT/'terminal-dspark-dead-bank.patch'),C8_screen=terminal['screen'],target_score_C8_percent={k:{m:v['percent'] for m,v in rows.items()} for k,rows in target['C8'].items()},ordinary_full_target_max_pct=max(x['percent_of_faster_reference'] for x in target['six_ordinary']),ordinary_terminal_free_max_pct=ordinary_free['maximum_percent'],larger_sparse_FFN_readout_free_ms=d['costs']['larger_sparse_FFN_plus_readout_free_P90_ms'],whole_chain_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
