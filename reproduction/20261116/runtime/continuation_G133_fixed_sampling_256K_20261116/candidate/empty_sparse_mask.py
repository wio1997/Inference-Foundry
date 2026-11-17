"""Define zero attention values for explicit empty sparse rows (single KV head)."""


def zero_empty_sparse_rows(output, lengths):
    assert output.ndim == 3 and lengths.ndim == 2
    assert lengths.shape == (output.shape[0], 1) and lengths.device == output.device
    # No scalar extraction or data-dependent host branch: safe in NPU graphs.
    output.masked_fill_((lengths[:, 0] == 0).view(-1, 1, 1), 0)
    return output


import datetime,hashlib,json,os
from pathlib import Path
_RECORDED=False

def record_once(output,lengths):
    global _RECORDED
    if _RECORDED:return
    import torch
    assert not torch.npu.is_current_stream_capturing(), 'Cold empty-sparse proof inside capture'
    empty=(lengths[:,0]==0)
    count=int(empty.sum())
    passed=bool(torch.isfinite(output[empty]).all()) and int(torch.count_nonzero(output[empty]))==0
    assert count>0 and passed, 'First explicit draft startup must prove defined empty rows'
    rank=torch.distributed.get_rank();directory=Path('/task/sglang_penalty133/empty_sparse_hits');directory.mkdir(exist_ok=True)
    item={'rank':rank,'pid':os.getpid(),'rows':output.shape[0],'empty_rows':count,'empty_output_exact_zero':passed,'outside_capture':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'recorded_once_before_formal':True}
    path=directory/f'rank{rank}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(item,indent=2)+'\n');tmp.replace(path)
    _RECORDED=True
    print('G127_EMPTY_SPARSE_POSTCONDITION_ACTUAL_PASS '+json.dumps(item),flush=True)
