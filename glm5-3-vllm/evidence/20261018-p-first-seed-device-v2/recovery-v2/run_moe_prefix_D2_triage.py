from pathlib import Path
import importlib.util,json,time
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=Path(__file__).resolve().parent
start=time.monotonic()
while not (r/'moe-prefix-D2-startup-outcome.json').exists():
 assert time.monotonic()-start<1400;time.sleep(10)
assert json.loads((r/'moe-prefix-D2-startup-outcome.json').read_text())['both_ready']
spec=importlib.util.spec_from_file_location('collect_base',r/'run_mtp_moe_live2_functional.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=m.collect(r/'triage_moe_prefix_D2.py','moe_prefix_D2_triage_v1','moe-prefix-D2-triage',timeout=600)
assert x['completed'],x.get('failure');print(json.dumps({'completed':True,'summary':x['summary'],'equal64':x['full64_equal_to_native_before'],'MTP_core_equal_by_serial':{str(k):[z['bitwise_equal'] for z in x['MTP_core_comparisons']['rows'] if z['serial_shared']==k] for k in [False,True]}}),flush=True)
