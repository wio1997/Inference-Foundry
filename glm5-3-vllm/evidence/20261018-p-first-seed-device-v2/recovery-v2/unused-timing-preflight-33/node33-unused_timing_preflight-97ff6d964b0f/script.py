import json,socket,subprocess,datetime,urllib.request
host=socket.gethostname();node=32 if host=='B4-33U-Atlas900RCKA3-32' else 33
assert host in ['B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33']
expected={32:'4cb8a338da590f0d0fd624892b2af7a20884029c7418c05ada75f5dbf3bcb157',33:'6584fed1fd2712610c6840a6cc631914e6e4549958055a5b2175f22455c7426a'}
d=json.loads(subprocess.check_output(['docker','inspect',expected[node]]))[0];assert d['State']['Running'] and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
now=datetime.datetime.now(datetime.timezone.utc);start=datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00');elapsed=(now-start).total_seconds();assert elapsed<14400-1200
ports={32:29481,33:29482}
models={}
for n in [32,33]:
 with urllib.request.urlopen('http://172.16.12.%s:%s/v1/models'%(n,ports[n]),timeout=5) as response:models[str(n)]=[x['id'] for x in json.load(response)['data']]
assert all('glm5' in ids for ids in models.values())
procs=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();busy=[line for line in procs if any(x in line for x in ['msprof','msprof-export','export_profile','backup_artifacts.py','rsync ']) and 'ps -eo' not in line]
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'host':host,'model_CID':expected[node],'elapsed_s':elapsed,'watchdog_remaining_s':14400-elapsed,'model_ids':models,'profile_export_transfer_processes':busy,'utc':now.isoformat()})+'\nGLM53_RESULT_JSON_END')
assert not busy,busy
