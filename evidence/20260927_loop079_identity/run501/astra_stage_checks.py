import hashlib,json,collections,types,sys,importlib.util
from pathlib import Path
root=Path('/data/wio/Inference_Foundry');e=root/'evidence/20260927_loop079_identity/run501';stage=root/'evidence/20260927_loop079_identity/run494/preflight_v4'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={};changed=[]
for line in (stage/'staged.sha256').read_text().splitlines():
 digest,path=line.split();p=root/path;live=root/'scripts'/p.name
 assert sha(p)==sha(live)==digest;changed.append(p.name)
checks['stage_live_sha_equal']=changed
baseline=Path('/tmp/astra_run501_v3_baseline')
for fn in ['loop079_target_frontier_run494.py','loop079_target_frontier_validate_run494.py']:
 old=(baseline/fn).read_text();new=(root/'scripts'/fn).read_text()
 assert new==old.replace("'MEMCPY','MEMSET'}","'MEMCPY','MEMCPY_ASYNC','MEMSET'}")
checks['production_parser_only_change']='one exact whitelist token per parser'
for fn in ['loop079_target_frontier_patch_run494.py','loop079_target_frontier_cleanup_cpu_test_run494.py','run_loop079_target_frontier_run494_b.sh']:
 assert (baseline/fn).read_bytes()==(root/'scripts'/fn).read_bytes()
checks['unchanged_patch_cleanup_original_controller']=True
old=(root/'scripts/run_loop079_target_frontier_run494_b.sh').read_text();new=(root/'scripts/run_loop079_target_frontier_run502_b.sh').read_text()
expected=old.replace('/run494/b_candidate','/run502/b_candidate').replace('LOOP079-RUN494-DUMP-B','LOOP079-RUN502-DUMP-B').replace('Run494 ','Run502 ')
assert new==expected
checks['controller_only_changes']='OUT run502, RUN_TS RUN502, fresh-path error text'
assert '-e EXTREME_RUN494_RUN_ID="${RUN_UUID}"' in new and "uuid.uuid4()" in new
assert not (root/'evidence/20260927_loop079_identity/run502/b_candidate').exists()
assert not (root/'logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP079-RUN502-DUMP-B.log').exists()
checks['run502_new_paths_absent']=True
# Run cleanup harness against the actual new controller, all external operations mocked.
spec=importlib.util.spec_from_file_location('cleanup',root/'scripts/loop079_target_frontier_cleanup_cpu_test_run494.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.CONTROLLER=root/'scripts/run_loop079_target_frontier_run502_b.sh';m.main()
checks['run502_actual_cleanup_harness']='success and all six failures pass'
sys.path.insert(0,str(root/'scripts'));import loop079_target_frontier_run494 as helper;import loop079_target_frontier_validate_run494 as validator
raw=[]
prior=json.loads((root/'evidence/20260927_loop079_identity/run500/raw_schema_checks.json').read_text())
prior_sha={r['file']:r['sha256'] for r in prior['raw']}
for p in sorted((root/'evidence/20260927_loop079_identity/run494/b_candidate/graph_dump').glob('*.json')):
 nodes=json.loads(p.read_bytes());meta=helper.graph_task_metadata(nodes);assert meta==validator.native_tasks(nodes)
 assert sha(p)==prior_sha[str(p.relative_to(root))]
 raw.append(dict(file=str(p.relative_to(root)),sha256=sha(p),tasks=meta['task_count'],memcpy_async=sum(n['args']['Task Type']=='MEMCPY_ASYNC' for n in nodes)))
checks['all8_live_v4_parsers_pass_raw_unchanged']=raw
reviewed=sorted(set(changed+['loop079_target_frontier_patch_run494.py','loop079_target_frontier_cleanup_cpu_test_run494.py','run_loop079_target_frontier_run494_b.sh','run_loop079_target_frontier_run502_b.sh']))
(e/'reviewed_scripts.sha256').write_text(''.join(f'{sha(root/"scripts"/fn)}  scripts/{fn}\n' for fn in reviewed))
(e/'stage_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({k:v for k,v in checks.items() if not isinstance(v,list)},indent=2))
