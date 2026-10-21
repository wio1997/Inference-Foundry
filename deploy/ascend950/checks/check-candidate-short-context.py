import json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,'/work/src/DeepInfer')
import torch,torch_npu
from deepinfer.model_executor.models.v4_modules.index_attn import MQA
from deepinfer.model_executor.layers.tilelang_ops.candidate_blocks import candidate_block_max_ref
results=[]
for width,block_size in [(7,8),(64,8),(4096,8),(4096,16),(16401,8)]:
    topk_blocks=2048
    valid_cpu=torch.tensor([0,1,min(15,width),width],dtype=torch.int32)
    torch.manual_seed(17)
    logits_cpu=torch.randn(4,width,dtype=torch.bfloat16)
    logits=logits_cpu.npu()
    valid=valid_cpu.npu()
    output=torch.full((6,topk_blocks),-77,dtype=torch.int32,device='npu')
    runtime=SimpleNamespace(block_candidate_idxs=output)
    mqa=SimpleNamespace(candidate_topk_cfg=SimpleNamespace(topk_block=topk_blocks,block_size=block_size))
    MQA._topk_candidate_block(mqa,logits,valid,1,runtime)
    torch.npu.synchronize()
    actual=output.cpu()
    assert torch.all(actual[0]==-77) and torch.all(actual[5]==-77)
    scores,counts=candidate_block_max_ref(logits_cpu,valid_cpu,block_size)
    for row,count in enumerate(counts.tolist()):
        selected=actual[row+1]
        n=min(topk_blocks,count)
        good=selected[:n].to(torch.int64)
        assert torch.all(selected[n:]==-1),(width,row,'padding')
        assert torch.all(good>=0) and torch.all(good<count)
        assert torch.unique(good).numel()==n
        assert torch.all(good[1:]>=good[:-1])
        if count<=topk_blocks:
            assert torch.equal(good,torch.arange(count)),(width,row,'all-blocks')
        else:
            expected=torch.topk(scores[row,:count],n).values.sort().values
            chosen=scores[row,good].sort().values
            assert torch.equal(chosen,expected),(width,row,'scores')
        if count:
            assert count-1 in good.tolist()
    results.append(dict(width=width,block_size=block_size,topk_blocks=topk_blocks,valid_lens=valid_cpu.tolist(),passed=True))
    print('CANDIDATE_SHORT_CONTEXT_PASS',results[-1],flush=True)
Path('/work/runtime-20261021/candidate-short-context-correctness.json').write_text(json.dumps(results,indent=2))
