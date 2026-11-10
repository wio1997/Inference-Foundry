#!/usr/bin/env python3
"""Preserve finished artifacts; correct three EOF-bounded review range labels."""
from pathlib import Path
from hashlib import sha256
import json

root=Path('audit/execution-search')
original=root/'native-workspace-source-review.json'
d=json.loads(original.read_text());changes=[]
for row in d['sources']:
 n=len(Path(row['local_path']).read_text().splitlines())
 old=row['reviewed_ranges'];new=[[a,min(b,n)] for a,b in old]
 assert all(1<=a<=b<=n for a,b in new)
 if new!=old:changes.append(dict(path=row['path'],old=old,corrected=new,actual_lines=n))
 row['reviewed_ranges']=new
assert len(changes)==3
correction=dict(status='EOF_RANGE_LABEL_CORRECTION_NOT_NEW_SOURCE_OR_TEST',original=str(original),original_sha256=sha256(original.read_bytes()).hexdigest(),superseding_review='native-workspace-source-review-v2.json',changes=changes,reason='The actual sed/read outputs ended at EOF. Three declared range end labels exceeded EOF; no nonexistent lines count as coverage. Original finished Run artifacts remain unchanged. All other range endpoints validated.',new_NPU_work=0)
d['range_correction']=correction
(root/'native-workspace-source-review-v2.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
(root/'native-workspace-source-review-correction.json').write_text(json.dumps(correction,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(changes))
