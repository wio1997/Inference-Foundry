#!/usr/bin/env python3
"""Fetch selected current Python/CANN O-projection dependencies without importing them."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import REMOTE

OUT=Path('audit/execution-search')
def main():
    prior=json.loads((OUT/'o-projection-native-entry.json').read_text())['sources']
    for name in ['qkv-native-source-review.json','qkv-extra-dependencies.json']:
        prior+=json.loads((OUT/name).read_text())['sources']
    existing=OUT/'o-projection-native-dependencies.json'
    if existing.exists():prior+=json.loads(existing.read_text())['sources']
    names=[r['runtime_path'] for r in json.loads((OUT/'o-projection-native-entry.json').read_text())['sources']]
    names+=['/vllm-workspace/vllm-ascend/vllm_ascend/'+n for n in ['ops/linear.py','ops/linear_op.py','quantization/utils.py','quantization/configs/modelslim_config.py','device/hardware_profile.py']]
    names+=['/vllm-workspace/vllm/vllm/model_executor/layers/linear.py']
    root='/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/'
    names += [root+'impl/ops_nn/ascendc/transpose_batch_mat_mul/'+n for n in ['transpose_batch_mat_mul.cpp','transpose_batch_mat_mul.h','transpose_batch_mat_mul_block.h','transpose_batch_mat_mul_tiling_key.h','transpose_batch_mat_mul_tiling_data.h','pp_matmul_ein_sum_kernel.h','pp_matmul_common.h']]
    names += [root+'impl/'+n for n in ['ops_nn/dynamic/transpose_batch_mat_mul.py','ops_legacy/dynamic/mat_mul.py','ops_legacy/mat_mul.py','ops_transformer/ascendc/matmul_reduce_scatter_v2/op_kernel/arch22/matmul_reduce_scatter_v2.cpp','ops_transformer/ascendc/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_v2_matmul.hpp','ops_transformer/ascendc/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode_matmul.h','ops_transformer/ascendc/matmul_reduce_scatter_v2/op_kernel/mrs_v2_copy_l0c_to_gm.h']]
    names += [root+'impl/'+n for n in ['ops_transformer/dynamic/matmul_reduce_scatter.py','ops_transformer/dynamic/matmul_reduce_scatter_v2.py','ops_transformer/ascendc/matmul_reduce_scatter/op_kernel/matmul_reduce_scatter.cpp','ops_transformer/ascendc/matmul_reduce_scatter/op_kernel/matmul_reduce_scatter_base.h','ops_transformer/ascendc/matmul_reduce_scatter/op_kernel/matmul_reduce_scatter_full_mesh.h']]
    names+=[root+'kernel/ascend910_93/ops_nn/transpose_batch_mat_mul/TransposeBatchMatMul_ND_ND_ND_ND_ND_BF16_BF16_BF16_INT64_BF16_high_performance.json','/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/csrc/aten/npu_native_functions_by_codegen.yaml']
    spec=[]
    for name in names:
        row=next((r for r in reversed(prior) if r['runtime_path']==name),None)
        p=Path(row.get('local_path',row.get('path'))) if row and row.get('local_path',row.get('path')) else None
        spec.append(dict(runtime_path=name,local_path=str(p) if p else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p and p.exists() else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for row in d['sources']:
        body=row.pop('content',None)
        if body is not None:
            p=OUT/'o-projection-native-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current disk sources and compiler metadata only. Exact reviewed ranges separate; not active loadedkey/binary, numerical or native admission.');existing.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(requested=len(spec),received=sum(not r.get('missing') for r in d['sources']),previous_matches=sum(r.get('matches_local',False) for r in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')])))
if __name__=='__main__':main()
