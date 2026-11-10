#!/usr/bin/env python3
"""Price a source-bounded metadata parent from saved scoped evidence, not new profiling."""
from pathlib import Path
from hashlib import sha256
import json,itertools

ROOT=Path('audit/execution-search')
def write(name,data):
    (ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def main():
    depfile=ROOT/'common-metadata-dependencies.json';dep=json.loads(depfile.read_text())
    assert len(dep['sources'])==9 and all(r['matches_local'] for r in dep['sources'])
    ranges={
      'worker/model_runner_v1.py':[[3449,3488],[3695,3907]],
      'attention/context_parallel/dsa_v41_cp.py':[[1,185]],
      'attention/context_parallel/dsa_cp.py':[[1139,1241]],
      'attention/dsa_v41.py':[[879,1205],[1225,1278]],
      'worker/device_metadata.py':[[74,155]],
    }
    sources=[]
    for r in dep['sources']:
        suffix=r['runtime_path'].removeprefix('/vllm-workspace/vllm-ascend/vllm_ascend/')
        sources.append(dict(**r,reviewed_ranges=ranges.get(suffix,[])))
    findings=Path('audit/idle-attribution-171/findings.json');saved=json.loads(findings.read_text())
    prices=[]
    for p in saved['prepare_boundaries']:
        part=next(r for r in p['parts'] if r['key']=='after_dp_before_first_metadata_submission')
        group=saved['groups'][f"dp{p['ep']//8}"]
        scenarios=[]
        for attributable,exposure,h in itertools.product([0,.25,.5,.75],[.25,.5,1],[.1,.5,2]):
            # A sensitivity with independently unknown attribution and exposure.
            net=part['mean_wall_ms']*attributable*exposure-h
            scenarios.append(dict(shareable_attributable_fraction=attributable,critical_exposure_fraction=exposure,added_nonoverlap_ms_per_iteration=h,conditional_net_ms_per_iteration=net,conditional_net_for_11cycles_ms=net*p['calls'],scoped_window_sensitivity_pct=100*net*p['calls']/group['mean_device_window_ms']))
        prices.append(dict(ep=p['ep'],cycles=p['calls'],submission_boundary_wall_ms=part['mean_wall_ms'],boundary_free_overlap_ms=part['mean_free_ms'],where_calls_per_iteration=part['where_calls_per_prepare'],matching_DP_mean_device_window_ms=group['mean_device_window_ms'],scenarios=scenarios))
    detail_path=Path('audit/performance-breakdown-171/findings.json')
    detail=json.loads(detail_path.read_text())
    host=detail['all'];where=next(r for r in host['host_ops'] if r['name']=='aten::where')
    detail_ref=dict(path=str(detail_path),sha256=sha256(detail_path.read_bytes()).hexdigest(),public_reference_commit='2f0d0c3904efcb05303d1f561cf3adc6a3151b5c',scope='Already-produced same scoped sample, not another capture. Allprepare-input self intervals, not only243where boundary.',where=where,host_groups=host['host_groups'],qualification='These are disjoint self intervals inside prepare-input Free, not pure CPU instructions or causal removable time. No exact Python attribution; do not add to boundary budget.')
    price=dict(status='CONDITIONAL_PARENT_BUDGET_NOT_OPTIMIZATION_MEASUREMENT',source_path=str(findings),source_sha256=sha256(findings.read_bytes()).hexdigest(),profiles=prices,same_sample_detailed_host=detail_ref,
      model='net_per_iteration = submission_boundary_wall * shareable_attributable_fraction * critical_exposure_fraction - added_nonoverlap_cost. Both fractions and cost are unmeasured; no fraction is inferred from where count or Free overlap.',
      whole_request='No matched unprofiled per-iteration caller attribution/critical-rank propagation or complete eligible iteration count. Request net is sum over actual eligible critical iterations minus one-time/integration cost; do not substitute old227waves or original8-distinct-prompt request data. No request improvement admitted.',
      limits=['This boundary is coordination-end to first native metadata submit, not exact Python function. Profiler overhead, dispatch/backpressure and other work can be inside it.','Free overlap identifies co-occurrence, not removable cost or a causal upper bound. DP mean window is a scoped sensitivity denominator, not complete request latency.','Do not add27msFree to31mswall, sum ranks, or add oldCPU-plan/25H2D/length-elision/fused-group-slot credits.','Parent optimistic envelope is large enough to keep investigating. It does not prove any specific where/native microkernel worth development.'],
      new_profile=0,new_NPU_work=0,new_primitive_admissions=0)
    write('common-metadata-delivery-price.json',price)
    frontier=dict(status='PARTIAL_SAME_GENERATION_CP_FRAME_AND_PUBLICATION_REVIEW',whole_chain_complete=False,precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',sources=sources,
      introduction='One same-generation local CP coordinate plan, distinct globalcache namespace, assembled with each resource block/slot/ring and original producer frontiers.',
      reached=[
        'Runner3822-3883 resets batch dictionary once per build and physical dictionary perKVgroup. V41 extra args supplied before builder.build; take_tasks and per-layer sharing retained.',
        'Legacy CP112-167 computes helper before base batch-cache hit. CPU query_start_loc selects torchfallback atdsa_cp1180; it clamps/cumsums/intersects lengths even ifbase already hasbatch. Then CP qsl copy/device suffix/clamp/where also run before superbuild.',
        'Base964-968 already shares firstbatchcoordinates,1039-1045 shares C2 lengths,1052-1059 shares RoPE; _publish_task deduplicates eachkey. Removing these again earns no extra credit.',
        'Native SMLA/QLI closures capture common.query_start_loc and max_query_len/max_seq_len as well as sharedcoordinates. Merely sharing returned metadata does not remove all source inputs or guarantee callback equality.',
        'V1 submit inputs_ready and later buffer_reusable gate producer stream; release records main-stream reusable but does not join all submitted tasks. Oldempty-query source counterexample remains; moving canonical main-stream writes earlier needs a producer-complete fence, not only release.',
      ],
      candidates=[
        dict(scope='Larger same-generation CP commonplan + per-resource assembly/publication',status='待确认',result='Keep original firstproducer arithmetic/rounding; reuse immutable CPUfacts and device-authoritative local lengths only across fully compatible fields. Worth continuing given existing scoped parent budget; no specific net or nativebit accepted.'),
        dict(scope='Grouped public frame or metadata producer',status='已确认可实现',result='本轮0；source-supported commonness is not per-primitive bitwise or lifecycle acceptance.'),
        dict(scope='Share whole return object across resource groups or shape-only cross-iteration cache',status='已停止',result='Physical slot/table, C2 ring writes, dynamic native attrs and callback/task/event ownership differ.'),
        dict(scope='Replace device seq_lens by CPU mirror',status='已停止',result='Speculative rejection makes CPUmirror an upperbound; retained authoritative device arithmetic is mandatory.'),
      ],
      concrete_plan=dict(
        compatibility=['same runner generation/target-or-draft role/CP rank and world size','num_input/actual/padded tokens and num_actual/padded requests','same causal flag/CPU QSL contents and dtype/private length snapshot','same authoritative device seq_lens dependency, positions/RoPE owner and graph/capture descriptor'],
        retain=['original global builder and cp_global namespace','eachresource block table/slot and compression/ring policy','original native SMLA/QLI attrs and full defined output bytes','reset/take/publish frontiers, zero-query tasks and callback side effects','per-group returned objects and stable captured addresses until observer equivalence proved'],
        lifetime='Pin CPU snapshots until every asynchronous DMA finishes; hold frame/device inputs until producer_done AND all consumer readers complete. Executor nextproducer reuse wait alone does not prevent earlier mainstream overwrite. New commonstage/event must preserve FULL external frontier set and ordering.',
        rollout='First reuse unchanged original CPU computation once with perbuilder stable device publications retained; this overlaps historicalCPUplan capability and cannot be counted twice. Larger commondevice producer/packing may remove remaining submissions but requires separate per-primitive and fence validation.'),
      gaps=['Actual per-call attribution of243where and grouped builder multiplicities in current scoped stock trace','Complete observers/mutations of common inputs, shared returned tensors and captured closure attrs','Producer-complete +consumer-complete lifetime for canonical frame including unconsumedQLI/emptyquery/error/abort','Bitwise originalCPU qsl/length/metadata and device/native packet plus consumer inputs across protected loads','Paid unprofiled net and all-rank/request exposure; ordinary/tail/draft20/fullgraph fallback and collective sequence'],
      prior_refs=['metadata-batch-frame-protocol.md','metadata-reader-frontier.json','metadata-host-history-qualification.json'],price='common-metadata-delivery-price.json',next_action='Continue concrete frame input observers and producer/consumer complete fence; current large budget does not justify a tiny standalone where kernel or repeating oldCPUplan benchmarks.',
      new_NPU_work=0,new_profile=0,service_requests=0,model_loads=0,remote_writes=[],new_primitive_admissions=0)
    write('common-metadata-frontier.json',frontier)
    excerpts=[]
    for r in sources:
        lines=Path(r['local_path']).read_text().splitlines()
        excerpts.append(dict(runtime_path=r['runtime_path'],full_source_sha256=r['runtime_sha256'],reviewed_ranges=r['reviewed_ranges'],excerpts=[dict(first=a,last=b,text='\n'.join(lines[a-1:b])) for a,b in r['reviewed_ranges']]))
    write('common-metadata-source-excerpts.json',dict(sources=excerpts,qualification='Only explicit bounded ranges; []identity-only. No native bit or full-chain coverage.'))
    print(json.dumps(dict(pins=len(sources),scenarios=sum(len(p['scenarios']) for p in prices),examples=[next(r for r in p['scenarios'] if r['shareable_attributable_fraction']==.25 and r['critical_exposure_fraction']==.5 and r['added_nonoverlap_ms_per_iteration']==.5) for p in prices]),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
