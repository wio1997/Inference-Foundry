#!/usr/bin/env python3
"""Record bounded source/CPU review; admit no replacement or native timing."""
from pathlib import Path
from hashlib import sha256
import json

ROOT=Path('audit/execution-search')
def write(name,d):
    (ROOT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
    deps=json.loads((ROOT/'attention-head-layout-dependencies.json').read_text())
    assert len(deps['sources'])==17 and all(r['matches_local'] for r in deps['sources'])
    ranges={
      'sparse_flash_mla.cpp':[[20,131]],
      'sparse_flash_mla_csa_kernel.h':[[184,245],[437,480],[599,677],[780,866]],
      'sparse_flash_mla_csa_block_cube.h':[[57,111],[180,231],[347,429],[485,540]],
      'sparse_flash_mla_csa_block_vector.h':[[419,480],[493,555],[658,780]],
      'sparse_flash_mla_swa_kernel.h':[[640,728],[857,916]],
      'sparse_flash_mla_swa_block_cube.h':[[140,230],[328,390],[667,737]],
      'dsa_v41.py':[[70,140],[567,632],[640,735]],
      'dsa_v41_cp.py':[[29,75],[78,220],[450,490]],
      'dsa_cp.py':[[77,91]],
    }
    review=[]
    for r in deps['sources']:
        name=Path(r['runtime_path']).name
        review.append(dict(**r,reviewed_ranges=ranges.get(name,[]),
            qualifier='Nonempty ranges reviewed in this bounded continuation; [] means identity-only or separate prior/CPU body reference. SHA/count is not coverage.'))
    cpu=json.loads((ROOT/'attention-head-schedule-cpu-a3/result.json').read_text())
    out=dict(status='PARTIAL_HEAD_LAYOUT_NATIVE_CONSUMER_REVIEW',whole_chain_complete=False,
        precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',sources=review,
        original_scalar_body=cpu['source'],prior_review_references=['attention-head-layout-entry-review.json','smla-direct-l1-source-review.json','smla-schedule-source-review.json','metadata-reader-frontier.json','qkv-native-source-review.json'],
        conclusions=[
          'Arch22 dispatch uses CSA only for CSA_TEMPLATE; dense SWA has separate kernel/mBase256. Actual target native third-input presence partitions480calls into456CSA and24denseSWA per representative rank.',
          'CSA mBase=G makes one query perM block. Source offsets use query-major/head coordinates; integer causal suffix and TP-output address checks hold in finite fixtures, not native numerical equivalence.',
          'Original CPU metadata planner25fixtures: defaultCSA CP64 FA3 -> global8 FA24; denseSWA FA1 ->4. Both default noFD. Level3 contrasts anddraft20/tail/zero not enabled newroutes.',
          'CSA head8 halves AIV active row distribution8/0 instead of32/32; zero-row vector helper returns while outer crosscore synchronization remains. SoftmaxFlashV2 dynamic row tiling changes; source head independence alone cannot certify BF16.',
          'Same selected-ID bank is overwritten at real index sources and reused between sources. Empty long cache fills-1. A proposed globalgather must finish all consumers before overwrite and be epoch/role/graph stable; prior96producer count is not a complete capture/lifetime proof.',
          'Cache-only global builder explicitly omits query metadata. Device seq_lens rather than CPU upper mirror must drive new globalquery plan. Local rope views avoid overwrite of shared global gather buffers.',
        ],
        failed_attempts=[dict(path='attention-head-schedule-cpu-a1/result.json',status='ERROR',scope='CPU fixture compile, nonexistent start fields; no native execution'),dict(path='attention-head-schedule-cpu-a2/result.json',status='INVALID',scope='25 scalar packets executed, verifier wrongly equated dense mBase andG; not a native failure')],
        successful_attempt=dict(path='attention-head-schedule-cpu-a3/result.json',status=cpu['status'],primitive_bit_admission=False),
        still_unscanned_or_unqualified=['Exact loaded runtime tiling keys/intrinsic implementation and row-dependent softmax bit behavior','Complete dense-SWA N-buffer copy and actual native cost under globalquery geometry','New common globalquery metadata producer and stable graph/lifetime/error/empty-DP collective guards','Native NZ head-shard descriptor/ownership and exact original QuantMatmul epilogue','All-rank critical path and fullrequest/ordinary-load eligibility and net cost'],
        new_NPU_work=0,model_loads=0,new_profile=0,service_requests=0,remote_writes=[],new_primitive_admissions=0)
    write('attention-head-layout-source-review.json',out)
    frontier=dict(status='LOOP039_PIVOT_TO_LARGE_COMMON_METADATA_DELIVERY',whole_chain_complete=False,
        precision_contract=out['precision_contract'],source_review='attention-head-layout-source-review.json',delivery_price='attention-head-layout-delivery.json',
        candidates=[
          dict(scope='Head-partition Qb -> globalquery SMLA -> inverseRoPE -> original TP O-proj',status='待确认',introduction='Delete original head transpose and query A2A while sharing QR/index communication and original cache ownership.',reached='Host guards, unchanged scalar packets, CSA native coordinate/MMAD/softmax boundary, dense branch distinction and paid wholeparent margins.',result='Potential remains, but CSA KV logical visits8x and paddedM2x invalidate treating original SMLA cost as a prediction. Need globalquery commonplan, KV reuse/headroom, bit-qualified kernels and paid native/request cost.'),
          dict(scope='Same larger candidate',status='已确认可实现',result='本轮0；source/CPU plan/address checks are not replacement bit acceptance.'),
          dict(scope='Flag-only reuse of cache-only global metadata packet as globalquery packet',status='已停止',result='Original global builder build_query_metadata=False; wrong QSL/device length/epoch. A newly built commonquery plan remains pending, not stopped.'),
          dict(scope='Claim every attention layer has CSA8x traffic/2x paddedM',status='已停止',result='24dense-SWA samples/representative rank use separatemBase256 kernel. No such extrapolation.'),
        ],
        remaining=out['still_unscanned_or_unqualified'],
        next_large_delivery='Current scoped steady prepare-input common metadata: same-generation CP query ranges/device lengths/suffix/mask -> per-cache resource fields -> asynchronous SMLA/QLI/compressor consumers. Reuse existing243where/31ms submission boundary only as budget, not exact function/removable/request saving.',
        new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
    write('attention-head-layout-frontier.json',frontier)
    # Public source excerpts contain only explicitly reviewed ranges; identity-only sources get no text.
    excerpts=[]
    for r in review:
        lines=Path(r['local_path']).read_text().splitlines()
        excerpts.append(dict(runtime_path=r['runtime_path'],full_source_sha256=r['runtime_sha256'],reviewed_ranges=r['reviewed_ranges'],excerpts=[dict(first=a,last=b,text='\n'.join(lines[a-1:b])) for a,b in r['reviewed_ranges']]))
    write('attention-head-layout-source-excerpts.json',dict(sources=excerpts,qualification='Bounded exact excerpts only; complete raw sources remain local at paths in review. Identity-only [] does not mean reviewed.',whole_chain_complete=False))

if __name__=='__main__':main()
