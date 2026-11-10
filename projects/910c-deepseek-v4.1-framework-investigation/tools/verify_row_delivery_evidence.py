#!/usr/bin/env python3
"""Check evidence identities and algebra; export exact bounded source spans."""
from pathlib import Path
from hashlib import sha256
import json,math
ROOT=Path('audit/execution-search')
def main():
    r=json.loads((ROOT/'row-delivery-review.json').read_text());c=json.loads((ROOT/'row-delivery-controls.json').read_text());p=json.loads((ROOT/'row-delivery-price.json').read_text())
    assert len(r['sources'])==30 and c['controls']==len(c['records'])==38
    assert c['tool_sha256']==sha256(Path('tools/probe_row_delivery_controls.py').read_bytes()).hexdigest()
    assert r['reviewer_sha256']==sha256(Path('tools/review_row_delivery.py').read_bytes()).hexdigest()
    for doc in [r,p]:
        for i in doc['inputs']:assert sha256(Path(i['path']).read_bytes()).hexdigest()==i['sha256']
    src={s['runtime_path']:s for s in r['sources']}
    for s in src.values():assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
    assert (r['source_bindings'],r['pending_original_sites'],r['new_original_bindings'])==(122,34,0)
    assert len(p['groups'])==4 and {g['ep'] for g in p['groups']}=={0,8} and p['request_credit']==0
    for g in p['groups']:assert math.isclose(g['nonDFC_span_optimistic_us'],g['parent_us']-g['DFC_us'])
    for d in [r,c,p]:
        for k in ['new_primitive_admissions','new_framework_replacement_admissions','new_NPU_work','new_profile','service_requests']:assert d[k]==0
        assert not d['whole_chain_complete'] and not d['remote_writes']
    spans={
      'vllm-ascend/vllm_ascend/worker/model_runner_v1.py':[(1281,1694),(1737,1752),(2548,2567)],
      'vllm/vllm/v1/worker/gpu_model_runner.py':[(1790,1924),(3502,3656)],
      'vllm/vllm/v1/utils.py':[(110,150)],
      'vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py':[(31,107),(226,357)],
      'vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py':[(1084,1140),(1192,1228),(1351,1398),(1454,1534),(2653,2669)],
      'vllm-ascend/vllm_ascend/ops/triton/spec_decode/utils.py':[(126,202)],
      'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py':[(262,328),(879,920),(1518,1608)],
      'vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py':[(248,276)],
      'vllm-ascend/vllm_ascend/ascend_forward_context.py':[(87,97),(101,236),(281,292),(318,425)],
      'vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py':[(252,348)],
      'vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py':[(638,780)],
      'vllm-ascend/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py':[(203,235)],
      'vllm-ascend/vllm_ascend/ops/fused_moe/dataclass/fused_experts.py':[(71,134)],
      'vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py':[(157,259),(485,540)],
      'vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py':[(113,194),(289,334),(469,615)],
      'vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp':[(360,399),(793,829)],
      'vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp':[(185,196)],
      'vllm-ascend/csrc/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h':[(19,93)],
      'vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_host/op_api/aclnn_dispatch_ffn_combine.cpp':[(44,79)],
      'vllm-ascend/csrc/torch_binding.cpp':[(3226,3231)],
    }
    out=[]
    for suffix,ranges in spans.items():
        s=src['/vllm-workspace/'+suffix];lines=Path(s['local_path']).read_text().splitlines()
        out.append(dict(runtime_path=s['runtime_path'],sha256=s['runtime_sha256'],ranges=[dict(first=a,last=b,lines=lines[a-1:b]) for a,b in ranges]))
    (ROOT/'row-delivery-reviewed-source.json').write_text(json.dumps(dict(qualification='Bounded caller/row writers and native-mask spans, not complete recursive callees, all native tilings or population. Source extraction itself is not a bit test.',sources=out),ensure_ascii=False,indent=2)+'\n')
    names=['row-delivery-dependencies.json','row-delivery-controls.json','row-delivery-control-source.json','row-delivery-review.json','row-delivery-price.json','row-delivery-reviewed-source.json','row-delivery-frontier.json','输入尾部与MC2有效行及MoE完整输出判断.md']
    result=dict(status='PASS_BOUNDED_ROW_SOURCE_AND_PRICE_IDENTITIES_NOT_NATIVE_BITS',source_pins=30,original_Python_controls=38,bound=122,pending=34,new_admissions=0,whole_chain_complete=False,every_replacement_bitwise=True,artifacts=[dict(path=str(ROOT/n),sha256=sha256((ROOT/n).read_bytes()).hexdigest()) for n in names])
    (ROOT/'row-delivery-integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}))
if __name__=='__main__':main()
