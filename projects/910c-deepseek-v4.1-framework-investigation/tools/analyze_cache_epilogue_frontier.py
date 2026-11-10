#!/usr/bin/env python3
"""Persist bounded native review; distinguish current source, native calls and admission."""
from pathlib import Path
from hashlib import sha256
import json

ROOT=Path('audit/execution-search')
SUMMARY='介绍：从indexK/longKV双输出扩大到每层SWA的RMS→partialRoPE→strided cache发布，核真实OOT/native和保留原舍入的融合边界。到了哪里：当前registry/caller与CANN/custom源的限定ranges，两个代表rank配平steady的48条十算子链和516条SWA三算子链逐任务绑定。结果：小十算子链全免仅窗口.427%–.432%，不重开独立开发；SWA三段全免3.065%–3.080%，half/h5us条件净1.255%–1.265%窗口，值得继续查主辅流join曝光。已发现55,296B/24行的中间GM交付可删方向，但必须保留RMS BF16与RoPE BF16两处边界、FP32三角表和原乘乘加；SK已融合地址计算、负linear跳写及stride0。当前新增原语准入0，窗口值不是请求收益，全链路未扫全，每个替换原语逐位一致。'
NEXT='loop038：沿SWA完整WKV输出/主辅Q链→RMS/partialRoPE/cache发布→主流join→SparseFlashMla实际读者，绑定已有balanced全native任务和Host/runtime事件。先判省下时间能否越过现有Qb重叠与join；核RMS真实tiling/binary、slot唯一/skip和原数组末读。保留逐位舍入；不向旧请求分母套新镜像窗、不因小index链低值剪掉SWA/input-ready较大parent。'

RANGES={
 'vllm_ascend/ops/register_custom_ops.py':[[1,140]],
 'vllm/model_executor/custom_op.py':[[109,141],[174,212],[250,311],[339,360]],
 'csrc/torch_binding.cpp':[[1577,1620],[1971,1979],[3453,3472],[3597,3606]],
 'op_kernel/inplace_partial_rotary_mul.cpp':[[25,110],[219,272]],
 'op_kernel/inplace_partial_rotary_mul.h':[[15,304]],
 'op_host/inplace_partial_rotary_mul_a3_tiling.cpp':[[1,85],[137,217],[258,355],[527,579]],
 'op_host/inplace_partial_rotary_mul_tiling.cpp':[[54,114]],
 'op_kernel/scatter_nd_update_sk.cpp':[[1,75]],
 'op_host/scatter_nd_update_sk_tiling.cpp':[[1,82],[149,526]],
 'arch22/scatter_nd_update_hp.h':[[24,270]],
 'arch22/scatter_nd_update_common.h':[[20,142]],
 'arch22/scatter_nd_update_large_index.h':[[22,163]],
 'rms_norm/rms_norm.cpp':[[1,90]],
 'rms_norm/rms_norm.h':[[144,229]],
 'rms_norm/rms_norm_single_row.h':[[130,168]],
 'rms_norm/rms_norm_whole_reduce_sum.h':[[20,58],[111,265]],
 'dynamic_quant/dynamic_quant.cpp':[[1,111]],
 'dynamic_quant/dynamic_quant_single_row.h':[[99,183]],
 'dynamic_quant/dynamic_quant_multi_row.h':[[207,274]],
 'vllm_ascend/utils.py':[[819,848],[876,898],[900,956]],
 'vllm_ascend/platform.py':[[1346,1368]],
 'vllm_ascend/worker/worker.py':[[136,158]],
 'vllm_ascend/device/hardware_profile.py':[[235,314]],
}

def main():
    records={}
    for name in ['cache-epilogue-native-dependencies.json','cache-epilogue-selected-sources.json','cache-epilogue-registry-callers.json']:
        for r in json.loads((ROOT/name).read_text())['sources']:
            if r.get('missing'):continue
            rp=r['runtime_path'];p=Path(r['local_path']);raw=p.read_bytes();h=sha256(raw).hexdigest();assert h==r.get('runtime_sha256',r.get('sha256'))
            ranges=next((v for k,v in RANGES.items() if rp.endswith(k)),[])
            assert all(1<=a<=b<=len(raw.decode().splitlines()) for a,b in ranges)
            if rp in records:assert records[rp]['sha256']==h;continue
            records[rp]=dict(runtime_path=rp,path=str(p),sha256=h,bytes=len(raw),reviewed_ranges=ranges,scope='Onlylisted line ranges read. Empty ranges = fetched/identity only. Source/shape inference is not exact loaded binary/key or native bit proof.')
    entry=json.loads((ROOT/'cache-epilogue-entry.json').read_text())
    for r in entry['sources']:
        rp=r['runtime_path'];p=Path(r['path']);assert sha256(p.read_bytes()).hexdigest()==r['sha256']
        rr=([[227,242],[270,330],[347,688]] if rp.endswith('/attention/dsa_v41.py') else r['reviewed_ranges'])
        records[rp]=dict(r,reviewed_ranges=rr,scope='Actual Python source/branch review; not a loaded native bit proof.')
    review=dict(status='BOUNDED_ACTUAL_OOT_AND_910C_NATIVE_REVIEW',whole_chain_complete=False,sources=list(records.values()),
        correction='The actual dsa_v41.scatter_cache_sk calls torch.ops._C_ascend.npu_scatter_nd_update_sk directly, with values.to(cache.dtype).contiguous(). Generic DeviceOperator.scatter_cache is not this publication callee.',
        registry='worker init calls register_ascend_customop; RMSNorm maps AscendRMSNorm; A3 standard profile AUTO_ENABLE_CUSTOM_OPS sets all; CustomOp selects oot then enabled dispatch. Frozen source and native RmsNorm shapes consistent, not per-invocation loaded object identity.',
        numerical=['BF16 norm output must remain a rounding boundary. Generic/single-row and whole-reduce source implementations differ in instruction/round-mode paths; active tiling key not recorded here.',
            '910C FP32 trig special RoPE source selects key12 for single KV head and key11 for broadcast query under enumerated shape guards. Original uses FP32 Mul/Mul/negate/Add then BF16 CAST_RINT, only final64 coordinates; prefix448 preserved.',
            'DynamicQuant single-row key100 emits FP32 max*(1/127), computes 127/max separately, multiplies BF16-promoted input, then RINT integer -> half -> int8. Multi-row key10 uses int16 intermediate. Exact active key/special values still required.',
            'Index codes are quantized before scale casts to FP16; query first-half-scale recipe differs. Existing RMS+DQ fused op lacks intervening RoPE and is not a certified replacement.'],
        scatter=['HP splits indices and already computes linear index in the same kernel, no SyncAll; duplicated destinations have nondeterministic cross-core writes. Source skip is linearIndex<0; this is not arbitrary per-coordinate-negative checking.',
            'INT64 large key30 checks negative coordinates and partitions destination range, preserving sequential scan order within owner; INT64 HP first casts to INT32. Do not merge these into one claimed universal contract.',
            'Wrapper passes var.strides; only first-axis expanded stride supported; tail dims contiguous; GM out uses uint64 and explicit firstDimStrideRows. Frozen typed rows INT32 cache plane total 17304*128<2^24 -> source key11 (not loadedkey proof).',
            'Tiler launches all vectorCoreNum (48 in saved rows), but HP owner partition uses min(vectorCoreNum,indexRows), hence 24/20 useful row owners for these typed calls. No separate linear-index launch saving. Workspace source retains sys plus two aligned index arrays although HP does not use those arrays; actual allocator saving not proved.'])
    (ROOT/'cache-epilogue-native-source-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    price=json.loads((ROOT/'cache-epilogue-balanced-sites.json').read_text())
    frontier=dict(status='NATIVE_CACHE_PUBLICATION_BOUNDED_SWA_LARGER_PARENT_PENDING',whole_chain_complete=False,every_replacement_bitwise=True,
        report='原生缓存发布与SWA整段融合判断.md',source_review='cache-epilogue-native-source-review.json',price='cache-epilogue-balanced-sites.json',
        confirmed_implementable=[],stopped=[
            'Keep original-loop031 independent small index-key/long-key epilogue stopped under its old fixed-C8 1% request reference. Balanced representative ten-op allfree <.433%window corroborates small local envelope, not a new request threshold.',
            'Do not claim separate LinearIndex kernel removal, duplicate cos loads per query head, scalar norm cancellation, FP32->BF16 trig demotion, FMA replacement, or existing RMS-DQ op as bitwise certified recipes. These claims lack the original boundaries/consumer evidence.'],
        pending=[
            dict(scope='SWA per-layer RMS/RoPE/cache full publisher with WKV/Q overlap',value='516 typed chains/rank, full nativefree3.065%-3.080%window; half/h5us1.255%-1.265%window, not request',gaps=['actual Qb/end join exposure and CPU submit waits','full critical TP/DP rank and old ordinary/prefill loads','active RMS/DQ tiling+loaded binary and primitive bit gate','slot generation uniqueness/skip and cache view stride','UB/register/bank/flags and replacement cost']),
            dict(scope='Nonmutating raw-latent RoPE-to-cache branch beside indexWK',gaps=['all alias readers and exact indexWK lastread','same ring/source cache generation and event readiness','larger joint exposure/cost; old independent earlymove child remains stopped'])],
        mathematical_and_memory_boundary=dict(conditional_saved_intermediate_bytes_per_row=2304,per_24_rows=55296,per_rank_sample_window=28200960,
            meaning='Delete norm BF16 GM output/read plus partial64 BF16 GM read/write by retaining both exact rounded arrays in UB and writing strided final cache once. RMS input/gamma/trig/index reads and final512BF16 cache write remain. Only if norm/rstd/intermediate have no other observable consumer.',
            already_shared=['Q/KV input quant under source share_inputs/method/communication guards','cos/sin within query head broadcast','SK index calculation in nativekernel']),
        next_action=NEXT,new_NPU_work=0,new_profile_collection=0,service_requests=0,model_loads=0,remote_writes=[],service_changes=0,
        source_pins=len(records),reviewed_nonempty=sum(bool(x['reviewed_ranges']) for x in records.values()))
    (ROOT/'cache-epilogue-native-frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n')
    notes=dict(errors=[dict(stage='local metadata aiv sum',error='ValueError converting N/A to float',resolution='Preserve per-type missing counts, sum only available PMU fields; no native compute failure'),dict(stage='local read',error='Initially used run.json and missing platform pathname; correct files manifest.json and vllm_ascend/platform.py read afterwards',resolution='No mutation/remote operation from failed reads')],scope='Current scan only: stdlib remote read-only source/savedCSV; local stdlib analysis; no framework imports or NPU/model/request/profile/service action.',source_counts_are_not_coverage_proof=True,strict_gate='Every replacement primitive bitwise identical. Source/CPU or historical service correctness cannot waive gate.',new_primitive_admissions=0)
    (ROOT/'cache-epilogue-native-exploration-notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(source_pins=len(records),reviewed_nonempty=frontier['reviewed_nonempty'],new_primitive_admissions=0)))

if __name__=='__main__':main()
