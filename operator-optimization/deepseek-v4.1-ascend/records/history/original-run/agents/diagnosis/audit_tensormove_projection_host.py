#!/usr/bin/env python3
"""Audit a bounded, previously extracted profile window; no device work."""
import hashlib, json
from decimal import Decimal
from pathlib import Path

p = Path(__file__).with_name('tensormove-ep8-neighbors.json')
raw = p.read_bytes()
expected = '22c3cf7e9248ca2f11da46a2aac8856ac66165dde36db2c09248b6f04d71b14b'
assert hashlib.sha256(raw).hexdigest() == expected, 'input changed'
d = json.loads(raw)
host = d['host']
flow = d['flows'][0]
start, end = flow['start'], flow['end']
assert flow['s_count_global'] == flow['f_count_global'] == 1
assert start['name'] == end['name'] == 'torch_to_npu'
assert start['id'] == end['id']
dev = [x for x in d['device'] if x['args'].get('Task Id') == 43440]
assert len(dev) == 1 and str(dev[0]['ts']) == str(end['ts'])
assert dev[0]['pid'] == end['pid'] and dev[0]['tid'] == end['tid']

def one(name):
    matches = [h for h in host if h['name'] == name and h['pid'] == start['pid'] and h['tid'] == start['tid']]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]
def span(h):
    s = Decimal(h['ts']); return (s, s + Decimal(h['dur']))
def contains(a,b):
    a0,a1=span(a); b0,b1=span(b); return a0 <= b0 and b1 <= a1

names = ['vllm::unquantized_gemm','aten::linear','aten::matmul','aclnnMatmul']
items = [one(n) for n in names]
assert str(items[-1]['ts']) == str(start['ts'])
assert all(contains(a,b) for a,b in zip(items,items[1:]))
for h in items:
    stack = h.get('args',{}).get('Call stack','')
    assert 'dsa_cp.py(1663): _apply_wo_b' in stack
    assert 'dsa_cp.py(1795): _forward_o_proj' in stack
copy, copy_api = one('aten::copy_'), one('aclnnInplaceCopy')
assert contains(copy,copy_api)
cf=d['flows'][1]
assert cf['s_count_global'] == cf['f_count_global'] == 1
assert str(cf['start']['ts']) == str(copy_api['ts'])
assert cf['start']['pid']==start['pid'] and cf['start']['tid']==start['tid']
assert cf['end']['pid']==d['device'][1]['pid'] and cf['end']['tid']==d['device'][1]['tid']
assert str(cf['end']['ts'])==str(d['device'][1]['ts'])
assert d['device'][1]['args']['Task Id']==43441
rs = one('vllm::ascend_sp_reduce_scatter_impl')
c10 = one('c10d::_reduce_scatter_base_')
assert contains(rs,c10)
assert span(items[-1])[1] <= span(copy)[0] and span(copy)[1] <= span(rs)[0]
base=Decimal(start['ts'])
result={
 'source_sha256': expected,
 'trace_sha256_recorded': d['trace_sha256'],
 'binding': 'unique recorded flow, timestamp/thread match; nested host intervals',
 'host_scope': [{'name': h['name'], 'relative_start_us':str(span(h)[0]-base), 'relative_end_us':str(span(h)[1]-base), 'input_dims':h.get('args',{}).get('Input Dims'), 'input_type':h.get('args',{}).get('Input type')} for h in items+[copy,copy_api,rs,c10]],
 'scope_note':'Host event uniqueness applies to extracted +-1500us start-time window; flow counts apply to whole trace as recorded by extractor. Input dtype is host API dtype, not device instruction or accumulator dtype. No storage identity is available.',
}
print(json.dumps(result,ensure_ascii=False,indent=2))
