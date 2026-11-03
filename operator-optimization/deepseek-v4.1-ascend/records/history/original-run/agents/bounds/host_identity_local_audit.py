#!/usr/bin/env python3
"""Extract only non-identifying hardware facts from saved 169 inventory."""
import hashlib
import json
from pathlib import Path
import re

here=Path(__file__).resolve().parent
run=here.parent.parent
project=run.parents[4]
paths={
 'board':project/'analysis/operator-ceiling-20261028/model-source-inventory.json',
 'mapping':project/'analysis/operator-ceiling-20261028/actual-sku-and-cp.json',
 'trace':run/'evidence/profile-device-capacity-info.json',
}
board=json.loads(paths['board'].read_text())['hardware_queries']
mapping=json.loads(paths['mapping'].read_text())['mapping']['stdout']
rows=[]
for line in mapping.splitlines():
 match=re.match(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(Ascend910)\s*$',line)
 if match:
  npu,chip,logic,physical,name=match.groups()
  rows.append({'npu':int(npu),'chip':int(chip),'logic':int(logic),'physical':int(physical),'name':name})
assert len(rows)==16 and len({x['logic'] for x in rows})==16
assert sorted({x['npu'] for x in rows})==list(range(8))
assert all(sum(1 for r in rows if r['npu']==i)==2 for i in range(8))
def field(text,name):
 m=re.search(r'^\s*'+re.escape(name)+r'\s*:\s*(.+?)\s*$',text,re.M)
 return m.group(1) if m else None
board_out=board['board0']['stdout']
memory_out=board['memory0']['stdout']
out={
 'status':'LOCAL_INVENTORY_ONLY',
 'frozen_mapping_npu_count':8,
 'frozen_mapping_logical_chip_count':16,
 'all_generic_chip_names':sorted({r['name'] for r in rows}),
 'generic_mapping_shows_two_chips_per_npu':True,
 'board_query_scope':'NPU 0 chip 0 only',
 'board0':{'NPU_Name':field(board_out,'NPU Name'),'Chip_Name':field(board_out,'Chip Name'),
           'Board_ID':field(board_out,'Board ID'),'Chip_Version':field(board_out,'Chip Version'),
           'HBM_Capacity_MB':field(memory_out,'HBM Capacity(MB)'),
           'HBM_Clock_Speed_MHz':field(memory_out,'HBM Clock Speed(MHz)')},
 'server_model':'UNKNOWN','local_review_no_authoritative_server_model_found':True,
 'memory_clock_is_proven_hard_max':False,
 'soc_model_of_remaining_chips_individually_proven':False,
 'inputs_sha256':{key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()},
 'npu_or_network_access':False,'serial_or_uuid_output':False,
}
(here/'host_identity_local_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(out,indent=2,ensure_ascii=False))
