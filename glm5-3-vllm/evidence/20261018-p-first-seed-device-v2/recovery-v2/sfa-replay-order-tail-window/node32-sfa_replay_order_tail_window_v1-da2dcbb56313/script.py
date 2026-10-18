from pathlib import Path
import datetime,json,subprocess,urllib.request
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/sfa-replay-order-live1-20261018/node32');pcid='e812d51531c504f8c7a036671499ad4b8fcf45ca3741a43dd3ef22a846a10db7';ccid='120fe694738b968313ea3a4006bf8f50517937708251afc9cd1b837416a70ee1';p=json.loads((root/'plan.json').read_text());pl=json.loads((root/'official-proxy-v1/launch.json').read_text());assert pl['CID']==ccid;now=datetime.datetime.now(datetime.timezone.utc);left={};states={}
for role,cid,life in [('P',pcid,p['container']['lifetime_seconds']),('CPU',ccid,pl['max_lifetime_seconds'])]:
 d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['Id']==cid and d['Image']==p['image']['id'];states[role]=d['State']['Running'];left[role]=life-(now-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds()
with urllib.request.urlopen('http://172.16.12.32:29880/healthcheck',timeout=5) as f:assert json.load(f)['request_num']==0
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'remaining_s':left,'states':states,'extension_one_ABBA_allowed':all(states.values()) and left['CPU']>210 and left['P']>300,'utc':now.isoformat()})+'\nGLM53_RESULT_JSON_END')
