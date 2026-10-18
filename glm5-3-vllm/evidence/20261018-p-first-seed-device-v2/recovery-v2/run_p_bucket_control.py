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
paths=m.collect(r/'read_p_bucket_paths.py','p_bucket_control_paths_v1','p-bucket-control-paths',node=32,timeout=90)
proof=[]
for row in f['rows']:
 prefix=row['P_response']['id'];joined=[z for z in paths['rows'] if any(rid==prefix or rid.startswith(prefix+'-') for rid in z['request_ids'])]
 assert sorted(set(z['tp_rank'] for z in joined))==list(range(16)),(prefix,len(joined))
 if row['N'] in [256,257]:
  expected=256 if row['N']==256 else 512
  assert all(z['num_real_tokens']==row['N'] and z['native_padded_tokens']==expected and 'PIECEWISE' in z['native_mode'] for z in joined),joined
 proof.append({'case':row['case_id'],'N':row['N'],'all16':True,'actual_real_padded_pairs':sorted(set((z['num_real_tokens'],z['native_padded_tokens']) for z in joined))})
(r/'p-bucket-control-path-summary-v1.json').write_text(json.dumps({'all16_native_only':True,'rows':proof},indent=2)+'\n')
x=m.collect(r/'read_p_bucket_paths.py','p_bucket_control_disable_v1','p-bucket-control-preflight',('--disable',),node=32,timeout=90);assert x['observer_disabled'] and x['native_marker_absent']
c1=run('c1');c4=run('c4',('--rotate-cases','--c4-cycles','3'))
(r/'p-bucket-control-outcome-v1.json').write_text(json.dumps({'completed':True,'native_only':True,'P_capture_sizes':[256,512,1024],'P_CID':'ccf21ea1b27e7fdbe4c45d4a862f9dc86a9ca4df38d937f55cdf7a6212465758','D_CID':m.cid,'C1':c1['summary'],'C4':c4['summary'],'scope':'Native P configuration control across sequential P windows; unchanged loaded D; no custom mechanism benefit or randomized configuration ABBA claim.'},indent=2)+'\n')
print(json.dumps({'phase':'bucket_control_completed','C1':c1['summary'],'C4':c4['summary']}),flush=True)
