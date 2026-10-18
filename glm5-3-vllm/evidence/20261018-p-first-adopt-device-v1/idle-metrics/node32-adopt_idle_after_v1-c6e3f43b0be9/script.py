import hashlib,json
from pathlib import Path
import urllib.request

artifacts=Path(__file__).resolve().parent/'artifacts';artifacts.mkdir(exist_ok=False)
rows=[]
for node,port in ((32,29481),(33,29482)):
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(node,port),timeout=5) as r:raw=r.read()
 (artifacts/('node%d-metrics.txt'%node)).write_bytes(raw)
 selected=[s for s in raw.decode().splitlines() if not s.startswith('#') and any(k in s for k in ('num_requests_running','num_requests_waiting','cache_usage','abort','cancel','request_prefill_time_seconds_sum','request_decode_time_seconds_sum'))]
 rows.append({'node':node,'sha256':hashlib.sha256(raw).hexdigest(),'selected_metrics':selected})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'nodes':rows,'device_ops':False})+'\nGLM53_RESULT_JSON_END')
