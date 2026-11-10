#!/usr/bin/env python3
"""Persist the bounded OA/OB source review; no runtime or framework imports."""
from pathlib import Path
from hashlib import sha256
import json

ROOT = Path('audit/execution-search')
RANGES = {
 'models/deepseek_v41/model.py': [[360,453],[665,770],[819,878],[879,928]],
 'models/layer/attention/layer.py': [[79,175]],
 'ops/cv_linear.py': [[13,189]],
 'ops/linear.py': [[35,160],[305,445]],
 'ops/linear_op.py': [[1,265],[340,445]],
 'transpose_batch_mat_mul.cpp': [[1,133]],
 'transpose_batch_mat_mul.h': [[69,205]],
 'transpose_batch_mat_mul_tiling_key.h': [[1,96]],
 'pp_matmul_ein_sum_kernel.h': [[382,475]],
 'ops_legacy/dynamic/mat_mul.py': [[238,306]],
 'arch22/matmul_reduce_scatter_v2.cpp': [[1,73]],
 'matmul_reduce_scatter_aiv_mode_matmul.h': [[40,117]],
 'matmul_reduce_scatter.cpp': [[1,88]],
 'matmul_reduce_scatter_base.h': [[24,86]],
 'matmul_reduce_scatter_full_mesh.h': [[84,198]],
 'dynamic/matmul_reduce_scatter_v2.py': [[1,70]],
 'interface/kernel_operator_fixpipe_intf.h': [[68,107]],
 'basic_api/kernel_operator_fixpipe_intf_impl.h': [[90,156]],
 'dav_c220/kernel_operator_fixpipe_v2_impl.h': [[1,118],[158,233],[316,366]],
}

def main():
    entry = json.loads((ROOT/'o-projection-native-entry-review.json').read_text())
    entry_ranges = {r['runtime_path']: r['reviewed_ranges'] for r in entry['sources']}
    rows = []
    for name in ['o-projection-native-dependencies.json','o-projection-fixpipe-dependencies.json']:
        rows += json.loads((ROOT/name).read_text())['sources']
    excerpts=[]; sources=[]
    for r in rows:
        assert not r.get('missing')
        p=Path(r['local_path']); b=p.read_bytes(); h=sha256(b).hexdigest()
        assert h==r['runtime_sha256']
        ranges=entry_ranges.get(r['runtime_path'],[])
        for suffix, rr in RANGES.items():
            if r['runtime_path'].endswith('/'+suffix): ranges=rr; break
        lines=b.decode().splitlines()
        source=dict(runtime_path=r['runtime_path'],path=str(p),sha256=h,bytes=len(b),reviewed_ranges=ranges,
                    scope='Only listed contiguous ranges reviewed; [] = identity/filename/declaration only. No active loaded binary/key or native bit proof.')
        sources.append(source)
        for a,z in ranges:
            assert 1<=a<=z<=len(lines),(p,a,z,len(lines))
            excerpts.append(dict(runtime_path=r['runtime_path'],sha256=h,first_line=a,last_line=z,lines=lines[a-1:z]))
    review=dict(status='BOUNDED_ORIGINAL_O_PROJECTION_NATIVE_RESOURCE_AND_PARENT_PRICE_REVIEW',sources=sources,
      whole_chain_complete=False,precision_contract='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',
      resolved=[
        'Current checkpoint headers contain 43 each BF16 OA[8192,4096]/OB[5120,8192], quant description FLOAT; actual model o_groups and TP8 give one local OA group. Catalog n_groups:null is an unselected config-name, not missing actual grouping.',
        'Actual OA ND BF16 transposebatchmatmul 102/012/102 factor1, then original selected unquantized BF16 OB, full-sized output buffer copy, external SP reduce_scatter, rounded four-way HC residual, HCPre and RMSNormCast dualoutputs.',
        'OA Matmul API route IterateAll writes GM; alternate PP route has float accumulation and CopyCcToGm. Compiled six suffixes and raw block24/mix0 do not identify active tilingKey or original K association.',
        'Installed 2201 SDK supports F322BF16 L0C->L1 NZ with noChannelSplit; L0C->UB unsupported here. OA panel is the LEFT operand of OB; L1 is private to its owning AIC.',
        'Existing KimiBF16 MMRS adapter only pureprefill and ai_cpu, not selected by ordinary Deepseek wo_b route. V2 AIV ENABLE_SHUFFLE_K source is not proven its backend. FullM opaque output allocation and caller RS must both change before any sharded fused API can fit.',
        'Old MMRS fullmesh uses typed CType and HCCL dataType; untyped variable names do not establish FP32 allreduce. Original OB BF16 and collective rounding/order remain separate bit gates.',
        'Finite BF16 RNE example disproves universal rounding-erasing weight precomposition, not original-rounding fusion. Precomposition also increases logical MAC/weight operand volume for actual dimensions.',
        'Valid saved host profiler DB AICORE_FREQ has six records at1800MHz across two representative windows. Prior container-namespace zero-result attempt is INVALID and retained.'
      ],
      remaining=[
        'Exact active original OA/OB loaded binary, tiling key, MMA K order, Fixpipe rounding/saturation and BF16 input/output for all protected cases; sampled dtype alone insufficient.',
        'N-sharded replicated full OA geometry: actual aggregate L2/fanout service, flags/full buffers, original K and BF16 primitive bits, paid net. Reference floors are necessary constraints, not implementation admission.',
        'Largeprefill original global carrier M3352/6536/6544 and all local/full weight sets; currentM24 singleowner stop does not cover those cases.',
        'SP collective original rounding/order and fullM->shard output/capture contract with coherent fallback.',
        'Actual HCPost->HCPre->RMSNormCast dualoutput native reduction/rounding and later residual/router readers; continue next loop.',
        'Full original C8 request and sixordinary exposure/noise/regressions; two mirrored-load short windows are not request prediction.'
      ],new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
    (ROOT/'o-projection-native-source-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'o-projection-native-source-excerpts.json').write_text(json.dumps(dict(scope='Exact listed excerpts only; full local source bodies retained by path/SHA. Not a complete body review.',excerpts=excerpts),ensure_ascii=False,indent=2)+'\n')
    frontier=dict(status='PARTIAL_O_PROJECTION_NATIVE_REVIEW_CONTINUE_HC_DUALOUTPUT',whole_chain_complete=False,
      precision_contract=review['precision_contract'],sources=sources,resolved=review['resolved'],remaining=review['remaining'],
      candidates=[
        dict(scope='Original OA/OB->SP->HC/FFN whole delivery',classification='待确认',result='Two saved target windows free envelope6.6681/6.7856%; retainoriginal OA/OB/RS and halve remainder with5us newcost:1.0770/1.1009%. Worth larger consumer investigation, not request gain.'),
        dict(scope='CurrentM24 singleAIC owns complete OA panel and allOBcolumns',classification='已停止',result='At recorded1.8GHz, optimistic L0B weight-ingress floor81.6356us exceeds every observed OA+OB pair63.38–74.779us; free-fallback oracle0. Stop exact local ownership family, retain largeprefill/multiowner.'),
        dict(scope='Universal OA/OB weightprecomposition erasing OA BF16 rounding',classification='已停止',result='Finite BF16 formula0x4040 vs0x4041; actualdimensions also2.222x logicalMAC and40vs18MiB operands.'),
        dict(scope='Keep both whole kernels, remove only OAend->OBstart gap',classification='已停止',result='Only11.74/11.299us total across480 targetpairs/window; doesnot stop larger persistent schedule or real compute/memory sharing.'),
        dict(scope='Each OB-N owner recomputes complete rounded OA panel',classification='待确认',result='At24 owners idealfloor38.4us but202MiB logicalweight service and~5.516TB/s aggregate atthatfloor. Ideal h5 net1.2141/1.2718%window; bit/resource/achievedservice/nativecost pending.'),
        dict(scope='New primitive implementations admitted',classification='已确认可实现',result='0; source/legal SDK route, analytical models and toy formula do not meet nativebit gate.')
      ],evidence=['o-projection-native-source-review.json','o-projection-native-source-excerpts.json','o-projection-native-catalog.json','o-projection-parent-price.json','o-projection-resource-ownership.json','o-projection-saved-clocks.json','o-projection-saved-clocks-a1-invalid.json'],
      new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
    (ROOT/'o-projection-native-frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(source_identities=len(sources),nonempty_reviewed_files=sum(bool(s['reviewed_ranges']) for s in sources),excerpts=len(excerpts),all_sha_match=True,whole_chain_complete=False)))

if __name__=='__main__':main()
