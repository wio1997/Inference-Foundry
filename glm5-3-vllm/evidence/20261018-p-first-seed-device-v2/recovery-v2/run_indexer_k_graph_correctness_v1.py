from pathlib import Path
import importlib.util
import json
import time

r=Path(__file__).resolve().parent
repo=r.parents[3]
spec=importlib.util.spec_from_file_location('transport',r/'run_prefix_count_service_v1.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
start=time.monotonic()
while not (r/'indexer-k-graph-startup-outcome-v1.json').exists():
    assert time.monotonic()-start<1500
    time.sleep(10)
assert json.loads((r/'indexer-k-graph-startup-outcome-v1.json').read_text())['both_ready']
cid=json.loads(next((r/'indexer-k-graph-D-launch-v1').rglob('launch.json')).read_text())['container_id']
pcid=json.loads(next((r/'indexer-k-graph-P-launch-v1').rglob('launch.json')).read_text())['container_id']
root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live1-20261019/node33'
setup=m.collect(r/'setup_indexer_k_graph_v1.py','indexer_k_graph_setup_v1','indexer-k-graph-setup-v1',timeout=90)
assert setup['K_graph_default_off']
proxy=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/start_proxy.py','indexer_k_graph_official_proxy_v1','indexer-k-graph-official-proxy-v1',
                ('--run-dir',root.replace('node33','node32'),'--expected-P-CID',pcid,'--minimum-lifetime-seconds','1800'),timeout=100,node=32)
assert proxy['ready']


def phase(name):
    x=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py',
                'indexer_k_graph_'+name.replace('-','_')+'_v1','indexer-k-graph-'+name+'-v1',
                ('--execute','--phase',name,'--probe-kind','indexer_k_graph_overlap',
                 '--run-dir',root,'--expected-D-CID',cid,'--cases',root+'/prefill-padding-cases-v1.json','--run-tag','v1'),timeout=650)
    assert x['completed'] and x['numeric_all_pass']
    print(json.dumps(dict(phase=name,rows=len(x['rows']),numeric_all=True)),flush=True)
    return x


phase('functional')
phase('c4-functional')
paths=m.collect(r/'read_indexer_k_graph_paths_v1.py','indexer_k_graph_paths_v1','indexer-k-graph-paths-v1',timeout=90)
assert paths['all16']
print(json.dumps(dict(phase='all16_actual_K_graph',selected=sum(z['selected'] for z in paths['rows']),current_native_bitwise=True)),flush=True)
m.collect(r/'disable_indexer_k_graph_observer_v1.py','indexer_k_graph_observer_off_v1','indexer-k-graph-observer-off-v1',timeout=60)
quality=phase('quality')
phase('cached')
unequal=[k for k,v in quality['sequence_comparisons'].items() if not v['all_same']]
out=dict(completed=True,functional16=True,C4_functional8=True,quality52=True,cached16=True,all16_actual_paths=True,
         all_selected_current_native_K_bitwise=True,quality_full64_equal_cases=26-len(unequal),unequal_cases=unequal,
         D_CID=cid,P_CID=pcid,formal_executed=False,
         next='Review quality, finite current native controls for initial+repeat candidate variants if needed, then actual all16 SDMA copy proof before any formal. No service gain claim.')
(r/'indexer-k-graph-correctness-outcome-v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
