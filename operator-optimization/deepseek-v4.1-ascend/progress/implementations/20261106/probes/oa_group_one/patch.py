"""Pinned integration candidate; not installed by the native admission runner."""
import ast
from hashlib import sha256

ORIGINAL = '7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305'
IMPORT = 'from wio_oa_group_one import project as _wio_oa_group_one\n'


def render(raw):
    assert sha256(raw).hexdigest()==ORIGINAL
    text=raw.decode();tree=ast.parse(text)
    functions=[n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body
               if isinstance(n,ast.FunctionDef) and n.name=='_forward_o_proj' and n.lineno==1754]
    fn,=functions
    lines=text.splitlines(keepends=True);body=''.join(lines[fn.lineno-1:fn.end_lineno])
    old='o_proj_input = torch_npu.npu_transpose_batchmatmul('
    assert body.count(old)==1 and body.count('                batch_split_factor=1,\n')==1
    body=body.replace(old,'o_proj_input = _wio_oa_group_one(').replace(
        '                batch_split_factor=1,\n',
        '                batch_split_factor=1,\n                _wio_permitted=not self.support_fp8_attention and not full_gather_wo_a_enabled,\n')
    lines[fn.lineno-1:fn.end_lineno]=[body];text=''.join(lines)
    assert text.count('import torch_npu\n')==1
    text=text.replace('import torch_npu\n','import torch_npu\n'+IMPORT)
    ast.parse(text)
    return text.encode()
