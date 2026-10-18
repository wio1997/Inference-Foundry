from pathlib import Path
import datetime,importlib.util,json,subprocess,sys,time
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=Path(__file__).resolve().parent
start=time.monotonic()
while not (r/'p-bucket-control-startup-outcome.json').exists():
 assert time.monotonic()-start<1100;time.sleep(10)
assert json.loads((r/'p-bucket-control-startup-outcome.json').read_text())['P_ready']
spec=importlib.util.spec_from_file_location('collect_base',r/'run_mtp_moe_live2_functional.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/start_proxy.py','p_bucket_control_proxy_v1','p-bucket-control-proxy',('--run-dir','/data/wio/glm5-3-vllm-execution/20261018-state-tail/p-bucket-control-v1-20261019/node32','--expected-P-CID','ccf21ea1b27e7fdbe4c45d4a862f9dc86a9ca4df38d937f55cdf7a6212465758'),node=32,timeout=100);assert x['ready']
print(json.dumps({'phase':'official_proxy','CID':x['CID']}),flush=True)
def run(phase,extra=()):
 x=m.collect(r/'validate_p_bucket_control.py','p_bucket_control_'+phase+'_v1','p-bucket-control-'+phase,('--execute','--phase',phase,'--probe-kind','mtp_moe_graph','--run-dir',m.root,'--expected-D-CID',m.cid,'--cases',m.root+'/prefill-padding-cases-v1.json')+extra,timeout=900)
 assert x['completed'] and x['numeric_all_pass'];print(json.dumps({'phase':phase,'rows':len(x['rows'])}),flush=True);return x
f=run('functional')
cfg=m.collect(r/'verify_p_bucket_config.py','p_bucket_control_config_v1','p-bucket-control-config',node=32,timeout=90)
assert cfg['completed'] and cfg['padding_observer_not_installed_in_this_build'] and cfg['P_D_idle']
print(json.dumps({'phase':'native_config_verified','graph_config':cfg['server_info_graph_config']}),flush=True)
c1=run('c1');c4=run('c4',('--rotate-cases','--c4-cycles','3'))
(r/'p-bucket-control-outcome-v1.json').write_text(json.dumps({'completed':True,'native_only':True,'P_capture_sizes':[256,512,1024],'P_CID':'ccf21ea1b27e7fdbe4c45d4a862f9dc86a9ca4df38d937f55cdf7a6212465758','D_CID':m.cid,'C1':c1['summary'],'C4':c4['summary'],'scope':'Native P configuration control across sequential P windows; unchanged loaded D; no custom mechanism benefit or randomized configuration ABBA claim.'},indent=2)+'\n')
print(json.dumps({'phase':'bucket_control_completed','C1':c1['summary'],'C4':c4['summary']}),flush=True)
