#!/usr/bin/env python3
"""Read one frozen EP9 kernel CSV in stage container; inspect one predecessor."""
import csv,hashlib,json,pathlib
from decimal import Decimal as D
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp1_dcp0_ep9_rank1_752_20261029023525649_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv')
sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha=='20af7cd5f9ca46fe2aeddc66a51cd30cea5048a374c87ba572f903bd1410cdf5'
with p.open(newline='') as f:rows=[x for x in csv.DictReader(f) if x.get('Stream ID')=='47']
t=[x for x in rows if x.get('Task ID')=='46755' and x.get('Name')=='DispatchFFNCombine' and D(x['Start Time(us)'].strip())==D('1793241333127617.287')]
assert len(t)==1;t=t[0];assert abs(D(t['Duration(us)'])-D('273565.15425'))<=D('0.001')
rows.sort(key=lambda x:D(x['Start Time(us)'].strip()));i=rows.index(t);assert i>0;p=rows[i-1]
assert p['Name']=='MoeGatingTopKHash' and p['Task ID']=='46754'
gap=D(t['Start Time(us)'].strip())-D(p['Start Time(us)'].strip())-D(p['Duration(us)']);wait=D(t['Wait Time(us)'])
keys=('Name','Task ID','Stream ID','Start Time(us)','Duration(us)','Wait Time(us)')
print(json.dumps({'csv_sha256':sha,'ep':9,'digest_prefix':'8c18c293','layer':0,'task':{k:t.get(k) for k in keys},'previous_same_stream_csv_row':{k:p.get(k) for k in keys},'previous_end_to_start_gap_us':str(gap),'recorded_wait_time_us':str(wait),'difference_us':str(wait-gap),'limits':['one current frozen CSV and its selected predecessor, not independent measurements','near agreement does not prove exporter rules, complete capture, or all prior tasks','Wait Time is not proven to measure internal MC2 phases, contention, or request time']},indent=2))
