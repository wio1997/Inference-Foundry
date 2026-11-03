from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18/M19-allrank-DSpark-forward-embedding-HC-decoder-and-communication-scope-v35.json');b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='abdd564691080a7f537a4f65668d72768461ad637e29633641c77e05a5ba5fa6';d=json.loads(b);out=p.parent/'M19-exact-live-query-cache-only-and-output-fullcallers-v35.json';assert not out.exists()
prefixes=['b6930b3defab','96dc70dfdaaf','ef96118d07d2','3816277f36bf','d7d350ddf155']
headers=[h for h in d['chain_headers']if any(h['chain_key'].startswith(k)for k in prefixes)]
attention_frames=sorted({f for h in d['chain_headers']for f in h['project_frames']if '/attention/'in f})
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected_fullcallers=headers,all_forward_attention_frames=attention_frames,scope='Saved fullcallers only; distinguish actually dispatched override from base methods. No source or model/device/service work.',overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
