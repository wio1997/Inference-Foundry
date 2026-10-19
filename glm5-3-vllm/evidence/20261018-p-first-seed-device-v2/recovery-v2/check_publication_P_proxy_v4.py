from pathlib import Path
import datetime,json,socket,subprocess,urllib.request
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-publication-live1-20261019/node32')
cid='84bdaf268d006b2d4d172bc6f9adb8f5f74cc9f38e1a622644536269f0918f8b'
proxy='a1fec8aedbff1f5fdbbba0e1d1ee40b21d9aecd52a31a0e9a80f79e01f8e2ef3'
plan=json.loads((root/'plan.json').read_text());launch=json.loads((root/'official-proxy-v1/launch.json').read_text())
assert launch['CID']==proxy and launch['parent_P_CID']==cid and launch['source_sha256']=='85b8982948530758be655d3448ff10fc3979e56622221feace560272a81f1469'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];p=json.loads(subprocess.check_output(['docker','inspect',proxy]))[0]
assert d['State']['Running'] and p['State']['Running'] and d['Name']=='/'+plan['container']['name']
assert d['Image']==p['Image']==plan['image']['id'] and not p['HostConfig'].get('Devices')
left=plan['container']['lifetime_seconds']-(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds();assert left>900
with urllib.request.urlopen('http://172.16.12.32:29880/healthcheck',timeout=5) as f:health=json.load(f)
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,ready=True,P_CID=cid,proxy_CID=proxy,remaining_P_s=left,health=health,P_and_unmodified_stock_proxy_retained=True))+'\nGLM53_RESULT_JSON_END')
