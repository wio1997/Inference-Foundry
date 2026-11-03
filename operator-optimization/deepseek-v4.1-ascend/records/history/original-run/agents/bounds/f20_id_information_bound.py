#!/usr/bin/env python3
"""Independent synthetic 595-row ID interface information bound; CPU-only."""
import hashlib,json
from pathlib import Path
p=Path('agents/diagnosis/int64-pad-gating-scope.json');assert hashlib.sha256(p.read_bytes()).hexdigest()=='49e4028342df962d0724c7f95aac4537b4804f7227049ffa42bf22040829c218'
rows=json.loads(p.read_text())['rows'];assert len(rows)==320
for r in rows:
 s=r['pad']['signature'];assert s['Input Shapes'].strip('"')=='595;2;' and s['Output Shapes'].strip('"')=='643' and s['Output Data Types']=='INT64'
c=json.loads(Path('evidence/model-config.json').read_text());V=c['text_config']['vocab_size'];assert V==129280
num_inputs=V**595;bits=num_inputs.bit_length()-1
assert bits==10103 and (1<<bits)<=num_inputs<(1<<(bits+1))
assert (1<<bits)-1<num_inputs
print(json.dumps({'vocabulary':V,'rows':595,'output_rows':643,'distinct_output_lower_expression':'129280**595','worst_case_bits':bits,'whole_bytes':(bits+7)//8,'input_int64_bytes':595*8,'output_int64_bytes':643*8,'limitations':'Independent exact-output interface, arbitrary vocab-row input family, receiver initially uninformed, all distinguishing channels counted; worst case only. Not observed HBM traffic or full-model bound.'},indent=2))
