from pathlib import Path
import datetime,hashlib,json,socket,subprocess,urllib.request
node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[socket.gethostname()]
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/'+('rope-step-reuse-p-live1-20261019/node32' if node==32 else 'rope-step-reuse-live1-20261019/node33'))
cid={32:'1a7c22de29006ffd698632527dacd522e746350770710e7f45a2ebacded857dc',33:'163c7646c30ab819514f112e0af510934aa592305f3ec37bd7e3b77a3edc2bdc'}[node]
plan=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==json.loads((root/'launch.json').read_text())['container_id'] and d['Name']=='/'+plan['container']['name']
assert d['Image']==plan['image']['id']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
env=dict(z.split('=',1) for z in d['Config']['Env'] if '=' in z)
required=dict(HCCL_INTER_HCCS_DISABLE='FALSE',HCCL_INTRA_ROCE_ENABLE='0',HCCL_INTRA_PCIE_ENABLE='1',HCCL_OP_EXPANSION_MODE='AI_CPU',
              GLM53_P_FIRST_ADOPT='0',GLM53_P_FIRST_ADOPT_TRACE='0',GLM53_EARLY_SAMPLE_COPY_SHADOW='0',
              GLM53_MOE_PREFIX_GMM2_COUNT_PROBE='0',GLM53_MOE_CUMSUM_I32_PROBE='0',GLM53_MOE_PREFIX_HISTOGRAM_PROBE='0',
              GLM53_MTP_MOE_GRAPH_PROBE='0',GLM53_SFA_REPLAY_ORDER_PROBE='0')
assert all(env[k]==v for k,v in required.items())
assert env['GLM53_EARLY_SAMPLE_PUBLICATION']=='0'
assert env.get('GLM53_ROPE_STEP_REUSE_PROBE','0')==('1' if node==33 else '0')
assert env['GLM53_ROPE_CACHE_LAYOUT_PROBE']==('1' if node==33 else '0')
assert env.get('ENPU_ENABLE','false').lower()!='true' and env.get('VLLM_USE_BREAKABLE_CUDAGRAPH','0')=='0'
if node==33:
    probe=root/'rope-step-reuse-probe'
    assert (probe/'disabled').exists() and not(probe/'enable').exists() and not(probe/'abort-test-once.json').exists()
    assert (root/'rope-cache-layout-probe/enable').exists() and (root/'rope-cache-layout-probe/disabled').exists()
for patch in plan['runtime_patches']:
    assert hashlib.sha256(Path(patch['source']).read_bytes()).hexdigest()==patch['sha256']
left=plan['container']['lifetime_seconds']-(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds();assert left>900
busy=[z for z in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
      if any(t in z for t in ['/data/wio/glm5-3','glm53-']) and any(t in z for t in ['msprof','msprof-export','export_profile','rsync ','tar -cz']) and 'ps -eo' not in z]
assert not busy,busy
idle={}
for n,port in [(32,29481),(33,29482)]:
    with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:text=f.read().decode()
    lines=[z for z in text.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))]
    assert len(lines)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in lines);idle[str(n)]=lines
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,node=node,CID=cid,remaining_s=left,required_env=required,
    idle_metrics=idle,related_busy=busy,overlay_SHA_verified=True,scope='Current exact loaded P/D native configuration, disabled observer and original native late publication/retirement and no related export/transfer. Existing same connector/SDMA selection retained; not fresh all-byte transport proof.'))+'\nGLM53_RESULT_JSON_END')
