from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
E=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-execution-identity-v14');M=E.parent/'native-PMU-complete-v18'
p=E/'native-compute-fullchains-v14-v2.json';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0';d=json.loads(b)
out=M/'F04-actual-router-linear-weight-fp32-branch-v29.json';assert not out.exists()
records=[];bychain=defaultdict(list)
for db in d['databases']:
 for a in db['API_records']:
  text='\n'.join(a['project_frames'])
  if '_apply_router_linear'in text:
   role='DIRECT_PRECAST_FP32_LINEAR_408'if'fused_moe.py(408)'in text else'REGISTERED_GATE_FORWARD_409'if'fused_moe.py(409)'in text else'OTHER_SOURCE_FRAME'
   row=dict(database_index=db['database_index'],API=a,branch_role=role);records.append(row);bychain[a['chain_key']].append(row)
headers=[dict(chain_key=key,API_records=len(rs),branch_roles=dict(Counter(r['branch_role']for r in rs)),project_frames=[f['frame']for f in d['canonical_fullchains'][key]if'/vllm'in f['frame']])for key,rs in bychain.items()]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),native_APIs=len(records),fullchains=len(headers),branch_API_counts=dict(Counter(r['branch_role']for r in records)),
 no_new_source_DB_model_device_service_work=True,overall_complete=False,
 limit='Positivesource-linebranchproof only;APIcountnotmoduleinvocationorgraphpointer/loadedbinaryproof. Explicitlyretainotherbranchesifpresent.')
result=dict(summary=summary,canonical_headers=headers,native_APIs=records)
blob=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(blob)
print(json.dumps(dict(summary=summary,canonical_headers=headers,full_result=str(out),result_sha256=hashlib.sha256(blob).hexdigest()),ensure_ascii=False))
