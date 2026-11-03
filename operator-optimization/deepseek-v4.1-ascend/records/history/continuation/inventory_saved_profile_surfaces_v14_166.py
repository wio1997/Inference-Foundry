from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
OUT=P/'analysis/native-execution-identity-v14/saved-profile-surfaces-v14.json';assert not OUT.exists()
rows=[]
for p in sorted((P/'runtime').rglob('*')):
 if p.is_file():
  rel=str(p.relative_to(P/'runtime'));st=p.stat();rows.append(dict(relative_path=rel,bytes=st.st_size))
  assert len(rows)<20000,'bounded file inventory'
interesting=[r for r in rows if any(v in r['relative_path'].lower()for v in ['trace.json','frontend','api_server','pyspy','py-spy','profiler_info'])]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),runtime_files=len(rows),runtime_bytes=sum(r['bytes']for r in rows),interesting_files=len(interesting),suffix_counts=dict(Counter(Path(r['relative_path']).suffix for r in rows)),no_file_contents_or_target_source_read=True,no_device_or_service_work=True,overall_complete=False)
payload=(json.dumps(dict(summary=summary,files=rows,interesting_files=interesting),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
public=dict(summary=summary,interesting_files=interesting,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest())
if len(json.dumps(public).encode())>30000:public['interesting_files']=interesting[:80];public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
