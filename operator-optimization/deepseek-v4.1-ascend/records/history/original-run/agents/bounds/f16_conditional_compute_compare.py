#!/usr/bin/env python3
"""Conditional dense BF16 arithmetic vs two saved samples; not a hardware proof."""
import hashlib, json
from fractions import Fraction
from pathlib import Path

root=Path(__file__).resolve().parents[2]
def load(path,expected):
    raw=(root/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
    return json.loads(raw)
small=load('agents/fusion/f16-layer24-run2-evidence.json','993d145ee0ed90284ebd76d2b33f4f1224d87f554c4cdd13bb6f4616bf591f97')
large=load('agents/fusion/f16-ep3-projection-provenance.json','996eb37b8741739d6edcc412736159194eea168f7298f7a46dbd57e8887071e1')
events=[]
for tid,typ,ins,outs in [('1282','TransposeBatchMatMul','24,1,4096;1,4096,1024','24,1,1024'),('1283','MatMulV2','24,1024;5120,1024','24,5120')]:
    hit=[x for x in small['events'] if x['task_id']==tid and x['stream']=='89'];assert len(hit)==1
    e=hit[0];assert e['type']==typ and e['input_shapes'].strip('"')==ins and e['output_shapes'].strip('"')==outs
    assert e['input_dtypes']=='DT_BF16;DT_BF16' and e['output_dtypes']=='DT_BF16'
    events.append(e['end_rel_ns']-e['start_rel_ns'])
large_durations=[x['task']['endNs']-x['task']['startNs'] for x in large['rows']]
assert events==[50559,14779] and large_durations==[270900,311400]
capacity=24*1_800_000_000*4096
assert capacity==176_947_200_000_000
out=[]
for rows,times in [(24,events),(8184,large_durations)]:
    macs=[rows*4096*1024,rows*1024*5120]
    for index,(work,elapsed_ns) in enumerate(zip(macs,times),1):
        lower_us=Fraction(work*1_000_000,capacity)
        ratio=lower_us/Fraction(elapsed_ns,1000)
        out.append({'physical_rows':rows,'projection':index,'logical_MAC':work,'zero_credit_ideal_us':float(lower_us),'profiled_task_us':elapsed_ns/1000,'conditional_ratio_percent':float(ratio*100)})
print(json.dumps({'assumed_capacity_MAC_per_s':capacity,'rows':out,'limitations':'Traditional dense full valid rows, all MACs within task on assumed 24 Cube pool, strict 1.8 GHz and 4096 MAC/cycle maxima, beta=0. Neither domain membership nor hard peak/clock/beta proved; profiled times and physical rows are not service tokens.'},indent=2))
