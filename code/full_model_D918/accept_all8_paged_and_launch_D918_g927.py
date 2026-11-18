"""Activate only staged D918 after the complete actual eight-rank component gate."""
from pathlib import Path
import ast,datetime,hashlib,json,re,socket,subprocess,time,urllib.request
host=socket.gethostname();assert host=='<ROLE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT')
gate=root/'sg_NPU_paged_indexer_gate_g924';run=root/'sg_pd_g918'
state=json.loads((gate/'state.json').read_text())
assert state['phase']=='stopped_released' and state['cleanup']['all8_process_lists_clear']
p=Path('/proc',str(state['pid']),'cmdline');assert not p.exists() or str(gate/'node_controller.py').encode() not in p.read_bytes()
assert not (run/'state.json').exists() and not (run/'actual_launch_handle.json').exists()
reports=[]
for rank in range(8):
    f=gate/f'evidence/rank{rank}.json';raw=f.read_bytes();x=json.loads(raw)
    assert x['rank']==rank and x['phase']=='passed' and len(x['cases'])==12 and len(x['graphs'])==12
    assert all(c['native_FP4_scores_bitwise_equal'] and c['selected_physical_slots_equal'] and c['inputs_unchanged'] for c in x['cases'])
    for ratio,capacity in [(1,263435),(2,131717)]:
        assert any(c['ratio']==ratio and c['rows']==96 and c['capacity_L']==capacity for c in x['cases'])
        for batch,bounds in [(96,{127,2048,63}),(1,{127,capacity,63})]:
            matches=[g for g in x['graphs'] if g['ratio']==ratio and g['rows']==batch]
            assert len(matches)==3 and {g['changed_live_bound'] for g in matches}==bounds
            assert all(g['native_eager_score_bits_and_physical_slots_equal'] for g in matches)
    reports.append({'rank':rank,'result':x,'result_sha256':hashlib.sha256(raw).hexdigest()})
for name,digest in json.loads((gate/'deployment_source_sha256.json').read_text()).items():assert hashlib.sha256((gate/name).read_bytes()).hexdigest()==digest
for name,digest in json.loads((run/'deployment_source_sha256.json').read_text()).items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest
trees=[ast.parse((p/'npu_paged_indexer.py').read_text()) for p in [gate,run]]
for name in ['score_paged','select_slots','_map_sorted_selected_slots']:
    nodes=[next(n for n in tree.body if getattr(n,'name','')==name) for tree in trees]
    assert ast.dump(nodes[0])==ast.dump(nodes[1]),('Gate-to-model core AST changed',name)
for device in range(8):
    owner=json.loads(Path('/root/.codex/coord',host+f'-npu{device}.owner.json').read_text())
    assert owner['state']=='released' and owner['pid']==state['pid'] and owner['run_directory']==str(gate)
cpu_owner=json.loads(Path('/root/.codex/coord',host+'-cpu144-151.owner.json').read_text());assert cpu_owner['state']=='released' and cpu_owner['pid']==state['pid']
check=subprocess.check_output(['npu-smi','info'],text=True)
assert len(re.findall(r'\|\s*\d+\s+910B1\s*\|\s*OK\s*\|',check))==8
assert all('No running processes found in NPU '+str(i) in check for i in range(8))
guard=json.loads((run/'P_revalidation_before_candidate_g926.json').read_text())
assert guard['P_ready']==200 and guard['canonical_workload_lock_fd_verified'] and guard['old_pairguard_actual_holding']['pid']==2187560
age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(guard['at'])).total_seconds();assert 0<=age<600
with urllib.request.urlopen('http://<ROLE_HOST>:19093/ready',timeout=10) as response:assert response.status==200
acceptance={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all8_strict_component_accepted':True,'cases':96,'changed_input_graph_replays':96,'reports':reports,'actual_component_controller_exited':True,'all8_NPU_and_CPU_owner_released':True,'model_score_selection_core_AST_equal_to_tested':True,'scope':'NativeFP4operanddomain bitwise scores/physicalslots/inputimmutability and changedinputNPUgraphs only; notmodelquality/matchedAISBench/gain.'}
(gate/'parent_acceptance_g927.json').write_text(json.dumps(acceptance,indent=2)+'\n')
print('G927_ALL8_PAGED_PARENT_ACCEPTANCE_JSON '+json.dumps(acceptance),flush=True)
with (run/'controller.log').open('xb') as log:
    argv=['python3','-u',str(run/'node_controller.py')]
    child=subprocess.Popen(argv,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(1);assert child.poll() is None,('D918 controller immediately exited',child.returncode,(run/'controller.log').read_text()[-3000:])
handle={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':child.pid,'argv':argv,'epoch':918,'role':'d','P832_retained':True,'all8_component_parent_accepted':True,'scope':'Actual model launch only; readiness/strict target-draft loads/graphs/functional/fullquality/AISBench/matchedprofiles/gain unverified. Must transfer pairguard provenance before model requests.'}
(run/'actual_launch_handle.json').write_text(json.dumps(handle,indent=2)+'\n')
print('G927_D918_MODEL_ACTUAL_START_JSON '+json.dumps(handle),flush=True)
