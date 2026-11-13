from pathlib import Path
import datetime,fcntl,json,os,signal,socket,subprocess,time,urllib.request,hashlib,re
R=Path('/data/sglang-dsv41');P=R/'serve_g29';STATE=P/'state.json';N='yxy-dsv41-sglang-167-g29';IMAGE='sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517';locks=[];started=False;proc=None;stop=False
assert socket.gethostname()=='SERVER_HOSTNAME_REDACTED'
P.mkdir(exist_ok=True)
def stamp(phase,**kw):
 d={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'guard_pid':os.getpid(),'container':N,**kw};p=STATE.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(STATE);print(json.dumps(d),flush=True)
def call(a,timeout=30):return subprocess.run(a,capture_output=True,text=True,timeout=timeout)
def own_inspect():
 r=call(['docker','inspect',N]);
 if r.returncode:return None
 d=json.loads(r.stdout)[0];assert d['Config']['Labels']['codex.task']==R.name;return d
signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('stop',True));signal.signal(signal.SIGINT,lambda *_:globals().__setitem__('stop',True))
try:
 assert not STATE.exists(),'Previous serving attempt requires review'
 ng=json.loads((R/'compat/probe_g49/state.json').read_text());assert ng['phase']=='cleanup_terminal' and ng['all8idle']
 ranks=[json.loads(f.read_text()) for f in (R/'compat/probe_g49').glob('rank_*.json')]
 assert len(ranks)==16 and all(x['status']=='passed_identity_roundtrip_and_graph' and all(x['checks'].values()) and x['normal_shapes']['combined']==[1024,5120] and x['configuration']['HCCL_BUFFSIZE']=='512' for x in ranks)

 dr=json.loads((R/'compat/probe_g44/dspark_q_rope_result.json').read_text())
 assert dr['status']=='passed_actual_DSpark_q_no_headnorm_NPU_RoPE_and_graph' and len(dr['cases'])==4
 for x in dr['cases']:
  assert x['NoPE_unchanged'] and x['finite'] and x['changed_input_position_graph_eager_exact'] and x['provided_output_exact']
  cpu=json.loads((R/'compat/dspark_rope_g47_cpu/result.json').read_text())
  ref=next(y for y in cpu['rows'] if y['T']==x['T'])
  assert x['original_formula_mismatch_count']==ref['mismatch_count']==len(x['mismatch_details'])==len(ref['details'])
  assert x['max_abs_diff_original_formula']==ref['max_abs_diff']
  for actual,expected in zip(x['mismatch_details'],ref['details']):
   assert actual['index']==expected['index'] and actual['actual']==expected['expanded'] and actual['original_F32']==expected['original_complex']

 assert json.loads((R/'compat/probe_g44/state.json').read_text())['physical_card0_idle']

 assert json.loads((R/'serve_g28/state.json').read_text())['phase']=='own_container_terminal_device_locks_releasing'
 gs=json.loads((R/'compat/probe_g42/state.json').read_text());assert gs['phase']=='cleanup_terminal' and gs['all8idle']
 rows=[json.loads(f.read_text()) for f in (R/'compat/probe_g42').glob('rank_*.json')]
 assert len(rows)==16 and all(x['status']=='passed_identity_roundtrip_and_graph' and all(x['checks'].values()) for x in rows)
 for phase in ['after_DeepEP_eager_before_graph','after_DeepEP_graph']:
  tests=[json.loads(f.read_text()) for f in (R/'compat/probe_g42').glob('metadata_'+phase+'_rank*.json')]
  assert len(tests)==16 and all(x['status']=='passed_initial_verify_metadata_matrix_and_actual_attach' and len(x['cases'])==21 for x in tests)

 evidence_statuses={'ownership':'passed_current_cpu_load_and_exact_symbol_ownership','candidate':'passed_legal_candidate_boundary_and_graph','ratio2':'ratio2_32heads_indices_and_graph_passed_attention_differences_reported','native_swa':'native_swa_dspark_operator_and_graph_passed_oracle_differences_reported','zero':'passed_qli_zero_mixed_and_graph_lifecycle','sparse_zero':'passed_sparse_zero_mixed_and_graph_lifecycle_oracle_differences_reported'}
 for stage,status in evidence_statuses.items():
  result=json.loads((R/('compat/probe_g12/'+stage+'_result.json')).read_text());assert result['status']==status, 'Final component probe not passed: '+stage
 assert json.loads((R/'compat/probe_g12/state.json').read_text())['physical_card0_idle']
 proof=json.loads((R/'compat/probe_g20/same_process_result.json').read_text())
 assert proof['status']=='passed_all_providers_same_process_graph_and_changed_inputs'
 assert len(proof['stages'])==10 and all(x['status']=='passed' and x['returncode']==0 for x in proof['stages'])
 assert len({x['pid'] for x in proof['stages']})==1
 assert json.loads((R/'compat/probe_g20/state.json').read_text())['physical_card0_idle']
 ep=[json.loads(p.read_text()) for p in (R/'compat/probe_g25').glob('rank_*.json')];assert len(ep)==16 and all(p['status']=='passed_identity_roundtrip_and_graph' and all(p['checks'].values()) and p['configuration']['HCCL_BUFFSIZE']=='1536' and p['resolved_dispatch_quantization']=='int8' for p in ep)
 assert json.loads((R/'compat/probe_g25/state.json').read_text())['all8idle']
 ep16=[json.loads(p.read_text()) for p in (R/'compat/probe_g26').glob('rank_*.json')];assert len(ep16)==16 and all(p['status']=='passed_identity_roundtrip_and_graph' and all(p['checks'].values()) and p['configuration']['capacity']==16 and p['configuration']['tokens']==12 for p in ep16)
 assert json.loads((R/'compat/probe_g26/state.json').read_text())['all8idle']
 ep512=[json.loads(p.read_text()) for p in (R/'compat/probe_g27').glob('rank_*.json')];assert len(ep512)==16 and all(p['status']=='passed_identity_roundtrip_and_graph' and all(p['checks'].values()) and p['configuration']['capacity']==16 and p['configuration']['HCCL_BUFFSIZE']=='512' for p in ep512)
 assert json.loads((R/'compat/probe_g27/state.json').read_text())['all8idle']
 wp=json.loads((R/'compat/probe_g29/same_process_result.json').read_text());assert wp['status']=='passed_all_providers_same_process_graph_and_changed_inputs' and len(wp['stages'])==10 and all(x['status']=='passed' and x['returncode']==0 for x in wp['stages'])
 assert json.loads((R/'compat/probe_g29/state.json').read_text())['physical_card0_idle']
 mp=json.loads((R/'compat/probe_g23/mhc_dispatch_result.json').read_text());assert mp['status']=='passed_existing_triton_NPU_dispatch_and_native_mhc_pipeline_graph' and mp['checks'] and all(mp['checks'].values())
 assert json.loads((R/'compat/probe_g23/state.json').read_text())['physical_card0_idle']
 gp=json.loads((R/'compat/probe_g21/graph_metadata_result.json').read_text());assert gp['status']=='passed_native_graph_metadata_provider_size_and_replay' and gp['checks'] and all(gp['checks'].values())
 assert json.loads((R/'compat/probe_g21/state.json').read_text())['physical_card0_idle']
 hc=json.loads((R/'compat/probe_g14/hc_moe_result.json').read_text())['checks']
 assert hc['actual_hcpost_auto_public_cann920_shapes_1_6_32'] and hc['hcpost_npugraph_changed_input']
 vendor=R/'compat/vendor_strict_host_g2';vm=json.loads((vendor/'strict_host_manifest.json').read_text())
 assert vm['payload_count']==20 and vm['preserved_payloads_exact']==17 and vm['tiling_two_paths_same_sha']
 for item in vm['payloads']:assert hashlib.sha256((vendor/item['path']).read_bytes()).hexdigest()==item['sha256']
 ctor=json.loads((R/'compat/probe_g4/host_loader_result.json').read_text());assert ctor['status']=='passed_default_npu_device_install_and_tiny_host_loader' and all(ctor['checks'].values())
 manifest=json.loads((R/'compat/runtime/g29/manifest.json').read_text())
 for name,sha in manifest['files'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==sha, 'Runtime source hash mismatch'
 q=json.loads((R/'compat/probe_g3/quant_result.json').read_text());assert q['passed'] and len(q['actual_loader_entrypoints'])==2
 e=json.loads((R/'compat/probe_g1/engram_result.json').read_text());assert e['status']=='passed_local_operator_and_graph_probe'
 h=json.loads((R/'compat/probe_g2/hash_gate_result.json').read_text());assert all(h['checks'].values()) and h['gate_comparisons']['candidate_vs_actual_a3_reference_npu']['bitwise_equal']
 assert json.loads((R/'image_ready.json').read_text())['id']==IMAGE
 m=json.loads((R/'model_ready.json').read_text());assert m['on_local_data_filesystem'] and m['all_index_file_sizes_match']
 policy=json.loads(Path('/data/shared-npu-pool/.control/npu-pool-policy.json').read_text());assert not policy.get('leases'),'Existing pool leases require coordination'
 for root,pattern in [(Path('/data/shared-npu-pool/.control/pool-device-locks'),'{}'),(Path('/data/shared-npu-pool/.control/locks'),'device-{}.lock'),(Path('/data/shared-npu-pool/port-locks'),'sglang-port-{}.lock')]:
  devices=range(16) if 'coord' not in str(root) else [18167,18280,14699,*range(18400,18407)]
  for v in devices:
   f=(root/pattern.format(v)).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 n=call(['npu-smi','info']);(P/'npu_before.txt').write_text(n.stdout+n.stderr);assert n.returncode==0 and all(f'No running processes found in NPU {i}' in n.stdout for i in range(8)),'NPU workload conflict'
 # G9 loaded the same target+draft on all16dies. Require at least that successful preflight HBM availability; no KV tuning masks foreign allocations.
 baseline=(R/'serve_g9/npu_before.txt').read_text()
 def hbm_rows(txt):
  rows=[]
  for line in txt.splitlines():
   f=[v.strip() for v in line.split('|')[1:-1]]
   if len(f)==3 and re.fullmatch(r'\d+\s+\d+',f[0]) and ':' in f[1]:
    vals=re.findall(r'(\d+)\s*/\s*(\d+)',f[2])
    if vals:rows.append(tuple(map(int,vals[-1])))
  return rows
 current=hbm_rows(n.stdout);previous=hbm_rows(baseline)
 assert len(current)==len(previous)==16, 'Cannot verify all16 HBM rows'
 # Observed empty-device baseline fluctuates by driver bookkeeping; weights require >52GiB before KV.
 required_mib=53*1024
 assert all(total-used>=required_mib for used,total in current), 'Less than53GiB free HBM before model load; conflicting allocation'
 (P/'hbm_preflight.json').write_text(json.dumps({'current_used_total_mib':current,'g9_successful_load_preflight':previous,'minimum_free_mib':required_mib,'basis':'G9 actual52GiB target+draft plus1GiB pre-load allowance; startup only, no benchmark threshold'},indent=2))
 for port in [18167,18280,14699,*range(18400,18407)]:
  with socket.socket() as s:s.bind(('127.0.0.1',port))
 assert own_inspect() is None,'Serving name exists; inspect it before any retry'
 for d in ['home','tmp','logs','cache']:(P/d).mkdir(exist_ok=True)
 argv=['docker','run','--name',N,'--label','codex.task='+R.name,'--network','host','--ipc','private','--shm-size','64g','--ulimit','memlock=-1','--cap-add','IPC_LOCK','--pids-limit','32768','--memory','1500g']
 for i in range(16):assert Path('/dev/davinci'+str(i)).exists();argv+=['--device','/dev/davinci'+str(i)]
 for d in ['davinci_manager','devmm_svm','hisi_hdc']:argv+=['--device','/dev/'+d]
 for host,container in [(str(R),'/task'),(m['path'],'/model'),('/usr/local/Ascend/driver','/usr/local/Ascend/driver'),('/usr/local/Ascend/firmware','/usr/local/Ascend/firmware'),('/usr/local/dcmi','/usr/local/dcmi'),('/usr/local/sbin','/usr/local/sbin'),('/etc/ascend_install.info','/etc/ascend_install.info')]:
  if Path(host).exists():argv+=['-v',host+':'+container+('' if host==str(R) else ':ro')]
 env={'HOME':'/task/serve_g29/home','TMPDIR':'/task/serve_g29/tmp','XDG_CACHE_HOME':'/task/serve_g29/cache','TORCHINDUCTOR_CACHE_DIR':'/task/serve_g29/cache/inductor','TORCH_EXTENSIONS_DIR':'/task/compat/probe_g5/home/.cache/torch_extensions/py312_cpu','MAX_JOBS':'8','TRITON_CACHE_DIR':'/task/serve_g29/cache/triton','PYTHONPYCACHEPREFIX':'/task/serve_g29/tmp/pycache','HF_HUB_OFFLINE':'1','HF_HOME':'/task/serve_g29/cache/hf','ASCEND_PROCESS_LOG_PATH':'/task/serve_g29/logs/ascend','ASCEND_WORK_PATH':'/task/serve_g29/ascend','ASCEND_RT_VISIBLE_DEVICES':','.join(map(str,range(16))),'SGLANG_NPU_USE_MULTI_STREAM':'1','SGLANG_ENABLE_OVERLAP_PLAN_STREAM':'1','SGLANG_SET_CPU_AFFINITY':'1','STREAMS_PER_DEVICE':'32','HCCL_OP_EXPANSION_MODE':'AIV','HCCL_BUFFSIZE':'512','SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK':'16','DEEPEP_HYBRID_DEPLOYMENT':'1','DEEP_NORMAL_MODE_USE_INT8_QUANT':'1','INF_NAN_MODE_FORCE_DISABLE':'1','PYTORCH_NPU_ALLOC_CONF':'expandable_segments:True','SGLANG_DSV4_FP4_EXPERTS':'False','SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK':'0','SGLANG_OPT_BF16_FP32_GEMM_ALGO':'torch','SGLANG_OPT_DEEPGEMM_HC_PRENORM':'False','SGLANG_OPT_FP8_WO_A_GEMM':'0','SGLANG_OPT_FUSE_WQA_WKV':'0','SGLANG_OPT_USE_FUSED_HASH_TOPK':'False','SGLANG_OPT_USE_OVERLAP_STORE_CACHE':'False','SGLANG_OPT_USE_TILELANG_MHC_POST':'False','SGLANG_OPT_USE_TILELANG_MHC_PRE':'False','SGLANG_ZBAL_BOOTSTRAP_URL':'tcp://127.0.0.1:14699','ZBAL_ENABLE_GRAPH':'1','ZBAL_NPU_ALLOC_CONF':'use_vmm_for_static_memory:True'}
 env.update({'SGLANG_NPU_V41_A3_COMPAT':'1','SGLANG_V41_CANDIDATE_VENDOR_ROOT':'/task/compat/vendor_strict_host_g2','SGLANG_V41_CANDIDATE_EXTENSION':'/task/compat/candidate_build_g8/lib/_sglang_v41_a3_g4.cpython-312-aarch64-linux-gnu.so','SGLANG_V41_A3_OPAPI_LIBRARY':'/task/compat/vendor_strict_host_g2/op_api/lib/libsglang_v41_a3_opapi.so'})
 env.update({'PYTHONFAULTHANDLER':'1','SGLANG_V41_ROPE_CORE5_EXTENSION':'/task/compat/rope_build_core5_g1/lib/_sglang_v41_rope_core5_g3.cpython-312-aarch64-linux-gnu.so','SGLANG_V41_A3_ROUTER_EXTENSION':'/task/compat/router_build_current_g2/lib/_sglang_v41_router_g2.cpython-312-aarch64-linux-gnu.so','SGLANG_V41_A3_ROUTER_OPAPI_LIBRARY':'/usr/local/Ascend/cann-9.2.0/opp/vendors/customize/op_api/lib/libcust_opapi.so'})
 env.update({'PYTHONPATH':'/task/compat/runtime/g29:/sgl-workspace/sglang/python','SGLANG_NPU_ENGRAM_INT8_E8M0':'1','SGLANG_NPU_ENGRAM_ROTATION':'/model/optional/quarot.safetensors'})
 for k,v in env.items():argv+=['-e',k+'='+v]
 args=json.loads((R/'compat/runtime/g29/args.json').read_text());args=['/model' if x=='/source_model' else x for x in args];(P/'requested_args.json').write_text(json.dumps(args,indent=2));(P/'environment.json').write_text(json.dumps(env,indent=2))
 import shlex
 shell='source /usr/local/Ascend/cann-9.2.0/set_env.sh; for f in /usr/local/Ascend/cann-9.2.0/opp/vendors/customize/bin/set_env.bash /usr/local/Ascend/cann-9.2.0/opp/vendors/custom_transformer/bin/set_env.bash; do if test -f "$f"; then source "$f"; fi; done; export ASCEND_CUSTOM_OPP_PATH=/task/compat/vendor_strict_host_g2:${ASCEND_CUSTOM_OPP_PATH:-}; export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH; exec python3 -m sglang.launch_server '+shlex.join(args)
 for source in sorted((R/'compat/runtime/g29/native/sglang').rglob('*.py')):
  relative=source.relative_to(R/'compat/runtime/g29/native');argv+=['-v',str(source)+':/sgl-workspace/sglang/python/'+str(relative)+':ro']
 argv+=['-v',str(R/'compat/runtime/g29/sdk/inplace_partial_rotary_mul.py')+':/usr/local/python3.12.15/lib/python3.12/site-packages/sgl_kernel_npu/operators/rope/inplace_partial_rotary_mul.py:ro','-v',str(R/'compat/runtime/g29/sdk/mhc_post_dpskv4.py')+':/usr/local/python3.12.15/lib/python3.12/site-packages/sgl_kernel_npu/operators/hyper_conn/mhc_post_dpskv4.py:ro']
 argv+=['-v',str(R/'compat/runtime/g29/sdk/moe_gating_top_k.py')+':/usr/local/python3.12.15/lib/python3.12/site-packages/sgl_kernel_npu/operators/moe/moe_gating_top_k.py:ro']
 argv+=['-v',str(R/'compat/runtime/g29/modelslim.py')+':/sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelslim/modelslim.py:ro','-v',str(R/'compat/runtime/g29/engram.py')+':/sgl-workspace/sglang/python/sglang/srt/layers/engram.py:ro','-v',str(R/'compat/runtime/g29/weight_utils.py')+':/sgl-workspace/sglang/python/sglang/srt/model_loader/weight_utils.py:ro','--entrypoint','bash',IMAGE,'-lc',shell];(P/'docker_command.json').write_text(json.dumps(argv,indent=2));stamp('all16_npu_claimed_starting_native_model',logical_devices=list(range(16)),ports=[18167,18280,14699,*range(18400,18407)]);started=True
 out=(P/'server.log').open('w');proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT);(P/'docker_client.pid').write_text(str(proc.pid));start=time.monotonic();ready=False
 while proc.poll() is None and not stop and time.monotonic()-start<21600:
  if not ready:
   # Fail own startup promptly if an uncoordinated external job enters after initial idle check.
   ns=call(['npu-smi','info']);foreign=[]
   current_container=own_inspect()
   if ns.returncode==0 and current_container:
    cid=current_container['Id']
    for line in ns.stdout.splitlines():
     f=[v.strip() for v in line.split('|')[1:-1]]
     if len(f)>=4 and re.fullmatch(r'\d+\s+\d+',f[0]) and f[1].isdigit():
      cg=Path('/proc')/f[1]/'cgroup'
      try:owner=cg.read_text()
      except FileNotFoundError:continue
      if cid not in owner:
       item={'pid':int(f[1]),'npu_chip':f[0],'memory_mib':f[3],'cgroup':owner,'container_ids':re.findall(r'[0-9a-f]{64}',owner)}
       try:
        pp=Path('/proc')/f[1];item['comm']=(pp/'comm').read_text().strip();item['start_ticks']=(pp/'stat').read_text().split()[21]
       except FileNotFoundError:item['process_exited_during_audit']=True
       foreign.append(item)
    if foreign:
     (P/'foreign_startup_conflict.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'processes':foreign,'npu_smi':ns.stdout},indent=2))
     raise RuntimeError('Foreign NPU workload entered during startup; stopping own container only')
   try:
    with urllib.request.urlopen('http://127.0.0.1:18167/health',timeout=3) as f:
     if f.status==200:ready=True;stamp('native_http_health_ready_requires_dram_and_model_validation',docker_client_pid=proc.pid)
   except Exception:pass
   if not ready and time.monotonic()-start>2400:raise TimeoutError('Native model startup exceeded40min; no automatic benchmark')
  time.sleep(10)
 if proc.poll() is not None:stamp('native_model_terminal',exit_code=proc.returncode,http_health_was_ready=ready)
 else:stamp('own_service_stop_requested_or_6hour_bound',http_health_was_ready=ready)
except BaseException as e:stamp('native_serving_attempt_failed',exception=type(e).__name__,error=str(e));raise
finally:
 if started:
  d=own_inspect()
  if d and d['State']['Running']:
   call(['docker','stop','--time','20',N],timeout=45);d=own_inspect()
   if d and d['State']['Running']:call(['docker','kill',N]);d=own_inspect()
   while d and d['State']['Running']:
    stamp('cleanup_failed_holding_device_locks',reason='Own container remains active');time.sleep(30);d=own_inspect()
  if d:(P/'terminal_inspect.json').write_text(json.dumps(d,indent=2))
  n=call(['npu-smi','info']);(P/'npu_after.txt').write_text(n.stdout+n.stderr)
  stamp('own_container_terminal_device_locks_releasing',all_physical_cards_idle=all(f'No running processes found in NPU {i}' in n.stdout for i in range(8)))
 for f in locks:f.close()
