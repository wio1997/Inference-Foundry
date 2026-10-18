from pathlib import Path
import datetime,hashlib,importlib.util,json,subprocess,sys
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=Path(__file__).resolve().parent
assert json.loads((r/'moe-prefix-cpu-check-v3.json').read_text())['passed']==17
assert json.loads((r/'moe-prefix-build-check-v3.json').read_text())['helper_sha256']=='173c720dd79dd656ec9cd88b51eeb373752ca71567254f88bcb1d367fcfac90c'
assert json.loads((r/'moe-prefix-D3-supply-analysis-v1.json').read_text())['cumsum_count']==0
ssh=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34'];pods={32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'}
spec=importlib.util.spec_from_file_location('lease',repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
plans={};limits=[]
for role,node in [('P',32),('D',33)]:
 p=json.loads((r/('moe-prefix-P-plan.json' if role=='P' else 'moe-prefix-D3-plan.json')).read_text());lease=json.loads((r/('moe-target-live1-lease'+str(node)+'.json')).read_text());res=p['reservation'];res.update(name=lease['metadata']['name'],expected_uid=lease['metadata']['uid'],owner_label=lease['metadata']['labels']['owner'])
 current=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=25));uid,allowed,expiry=m.fresh_lease(current,res,datetime.datetime.now(datetime.timezone.utc));assert uid==res['expected_uid'];limits.append(allowed)
 assigned=json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'],timeout=25))['items'];assert all(not sum(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in z['spec'].get('containers',[])) or z['metadata']['uid']==uid or z['status']['phase'] in ['Failed','Succeeded'] for z in assigned)
 smi=subprocess.check_output(ssh+['kubectl exec -i -n default '+pods[node]+' -- nsenter -t 1 -m -- npu-smi info'],timeout=25).decode();assert smi.count('No running processes found in NPU')==8
 (r/('moe-target-prelaunch-idle-'+str(node)+'.txt')).write_text(smi)
 # Existing verified immutable P build9886 and D builded175; no rebuild or source mutation.
 p['remote']['run_dir']='/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-prefix-target-live1-20261019/node'+str(node);p['container']['name']='glm53-moe-target-'+str(node)+'-live1-20261019'
 for key in ['GLM53_MTP_MOE_GRAPH_PROBE','GLM53_MTP_FFN_GRAPH_PROBE','GLM53_SFA_REPLAY_ORDER_PROBE','GLM53_PREFILL_PADDING_PROBE','GLM53_PREFILL_ATTENTION_PROBE']:assert p['env'][key]=='0'
 assert p['env']['GLM53_MOE_PREFIX_HISTOGRAM_PROBE']==('1' if role=='D' else '0') and p['env']['GLM53_ROPE_CACHE_LAYOUT_PROBE']==('1' if role=='D' else '0')
 argv=p['serve_argv'];i=argv.index('--kv-transfer-config')+1;kv=json.loads(argv[i]);kv['engine_id']='glm53-moe-target-'+role.lower()+'-live1-20261019';argv[i]=json.dumps(kv,separators=(',',':'))
 if role=='P':assert json.loads(argv[argv.index('--compilation-config')+1])=={'cudagraph_mode':'PIECEWISE','cudagraph_capture_sizes':[256,1024]}
 p.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),holder_expiry_utc=expiry,objective='Factor prefix representation change: target graph only; original native MTP in both arms',notes=['D3 complete both-prefix loop: C1 TPOT-4.63%, C4 aggregate+9.26% but native outlier7.52s TTFT and per-request meanTPOT p95+3.33%; keep both-prefix default off. No unchanged full-benchmark repeat.','Actual optimized trace eliminated400 Cumsum, but25 MTP core host submissions still lead first physical compute by12.88ms minimum; target-only leaves eager MTP native and isolates target producer representation.','Same immutable D3 helper/native family-specific warmup and fresh metadata; use existing target-only selector, no new operator/source build/precision/transport. RoPE accepted contiguous buffers enabled in BOTH arms.','P old watchdog naturally expired; D old remaining deadline cannot fit P startup plus full new validation. Both exact owned predecessor CIDs and all16 stopped profile raw SHA backed before owned finite holder renewal.','New holder7200/model5400 covers measured~650s startup and complete quality/cache/C1/rotated balanced C4/cancel/profile. Default marker native; target-only selector fixed in all arms, MTP actual paths must all be native.'])
 plans[role]=p
life=min(5400,*limits);assert life>=5300,limits
for role in ['P','D']:
 p=plans[role];p['container']['lifetime_seconds']=life;path=r/('moe-target-'+role+'-plan.json');path.write_text(json.dumps(p,indent=2)+'\n')
 q=subprocess.run([sys.executable,'glm5-3-vllm/tools/start_capacity.py','--plan',str(path),'--control','/tmp/glm53-active-20261018.sock','--output',str(r/('moe-target-'+role+'-launch'))],cwd=repo,capture_output=True,text=True,timeout=240);(r/('moe-target-'+role+'-launch-collector.txt')).write_text(q.stdout+q.stderr);print(json.dumps({'role':role,'life':life,'rc':q.returncode,'stdout':q.stdout}),flush=True);assert q.returncode==0,q.stderr
