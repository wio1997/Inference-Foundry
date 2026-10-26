import sys
sys.dont_write_bytecode=True
import hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
R=Path('/data/wio/Inference_Foundry');O=R/'evidence/20260927_loop079_identity/run508/preflight';O.mkdir(parents=True,exist_ok=True)
RAW=R/'evidence/20260927_loop079_identity/run507/b_candidate'
V=R/'scripts/loop079_target_frontier_validate_run507.py';U=R/'scripts/loop079_target_frontier_update_validate_run507.py';C=R/'scripts/run_loop079_target_frontier_run508_b.sh'
checks=[];sha=lambda b:hashlib.sha256(b).hexdigest()
def ck(n,x,d=None):checks.append(dict(test=n,passed=bool(x),detail=d))
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
def cli(script,root,output,extra=(),cwd='/tmp'):
 p=subprocess.run(['python3',str(script),str(root),*extra,'--output',str(output)],cwd=cwd,env=env,capture_output=True,text=True)
 return p, json.loads(output.read_text()) if output.exists() else None
inputs=[V,U,C,R/'scripts/loop079_target_frontier_run507.py',R/'scripts/loop079_target_frontier_patch_run507.py']
pins={str(p):sha(p.read_bytes()) for p in inputs}
rawpins={str(p):sha(p.read_bytes()) for p in RAW.rglob('*') if p.is_file()}
p,x=cli(V,RAW,O/'posthoc_invalid_Run507_structure_only.json')
ck('absolute_CLI_from_tmp_no_PYTHONPATH',p.returncode==0 and x['update_rows']==40 and x['update_zip_iteration_counts']==[0],dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr))
p,x=cli(U,RAW,O/'posthoc_invalid_Run507_update_only.json')
ck('absolute_update_CLI_writes_summary',p.returncode==0 and x['rows']==40 and {s['zip_iterations'] for s in x['summaries']}=={0})
p,x=cli(V,RAW,O/'Run507_final_remains_invalid.json',('--final-only',))
ck('actual_Run507_final_rejects',p.returncode!=0 and x['valid'] is False,x)
status=dict(x.split('=',1) for x in (RAW/'cleanup_status.txt').read_text().splitlines())
ck('Run507_exit1_preserved',status['run_exit']==status['final_exit']=='1' and all(status[k]=='0' for k in ['stop_exit','stop_verify_exit','restore_exit','sha_exit','sha_compare_exit']))
old=R/'evidence/20260927_loop079_identity/run507/preflight/reviewed_v2/loop079_target_frontier_validate_run507.py'
p,x=cli(old,RAW,O/'old_import_failure_reproduced.json')
ck('prior_direct_script_import_failure_reproduced',p.returncode!=0 and 'ModuleNotFoundError' in x['error'],x)
with tempfile.TemporaryDirectory(prefix='astra_run508_CLI_') as tmp:
 root=Path(tmp)/'SYNTHETIC_ONLY';shutil.copytree(RAW,root)
 for rank in range(8):
  gp=root/'graph_dump'/f'rank{rank}_cohort5_acl_graph.json';mp=gp.with_suffix('.meta.json');cp=root/'capture'/f'rank{rank}_cohort5.json'
  meta=json.loads(mp.read_text());meta['path']=str(gp);write(mp,meta)
  row=json.loads(cp.read_text());row['debug_dump']['path']=str(gp);row['debug_dump']['meta_path']=str(mp);write(cp,row)
 p,x=cli(V,root,root/'validation.json');ck('synthetic_absolute_main_validation',p.returncode==0)
 p,x=cli(U,root,root/'update_validation.json');ck('synthetic_absolute_update_generation',p.returncode==0)
 # These status bytes exist ONLY in disposable synthetic fixtures, never Run507.
 (root/'cleanup_status.txt').write_text('\n'.join(k+'=0' for k in status)+'\n')
 p,x=cli(V,root,root/'final_admission.json',('--final-only',));ck('synthetic_absolute_cleanup_final_positive',p.returncode==0 and x['valid'])
 for label,mut in [('bad_count',lambda a:a.__setitem__('source_inferred_successful_event_records',1)),('missing_capture_snapshot',lambda a:a.pop('mla_params_capture')),('wrong_generation',lambda a:a.__setitem__('replay_capture_generation',999)),('promoted_completion',lambda a:a.__setitem__('device_event_completion','complete'))]:
  cp=root/'capture/rank0_cohort1.json';saved=cp.read_bytes();row=json.loads(saved);mut(row['actual_update']);write(cp,row)
  p,x=cli(V,root,Path(tmp)/('negative_'+label+'.json'),('--final-only',));ck('absolute_final_negative_'+label,p.returncode!=0 and x['valid'] is False);cp.write_bytes(saved)
 summary=root/'update_validation.json';saved=summary.read_bytes();summary.unlink();p,x=cli(V,root,Path(tmp)/'missing_summary.json',('--final-only',));ck('absolute_final_missing_summary_rejected',p.returncode!=0 and not x['valid']);summary.write_bytes(saved)
 x=json.loads(saved);x['summaries'][0]['zip_iterations']=999;write(summary,x);p,x=cli(V,root,Path(tmp)/'changed_summary.json',('--final-only',));ck('absolute_final_changed_summary_rejected',p.returncode!=0 and not x['valid']);summary.write_bytes(saved)
 # Exercise package mode separately; production uses the absolute-path mode above.
 p=subprocess.run(['python3','-m','scripts.loop079_target_frontier_validate_run507',str(root),'--output',str(Path(tmp)/'package.json')],cwd=R,env=env,capture_output=True,text=True)
 ck('package_import_mode',p.returncode==0,dict(stderr=p.stderr))
source=C.read_text();oldsource=(R/'scripts/run_loop079_target_frontier_run507_b.sh').read_text()
expected=oldsource.replace('/run507/b_candidate','/run508/b_candidate').replace('LOOP079-RUN507-MLA-B','LOOP079-RUN508-MLA-B').replace('Run507 dump','Run508 dump').replace('Run507 server','Run508 server')
ck('controller_only_fresh_run_identity_changes',source==expected)
ck('controller_bash_syntax',subprocess.run(['bash','-n',str(C)]).returncode==0)
cleanup=(R/'scripts/loop079_target_frontier_cleanup_cpu_test_run494.py').read_text();ns={'__file__':str(R/'scripts/loop079_target_frontier_cleanup_cpu_test_run494.py')};exec(compile(cleanup,'cleanup_mock','exec'),ns);ns['CONTROLLER']=C;ns['main']();ck('actual_Run508_seven_cleanup_mock_paths',True)
functions=source[source.index('verify_stopped() {'):source.index('\ntrap cleanup EXIT')]
for label,runexit,finalgate,expected in [('update_failure',7,0,7),('final_only_failure',0,9,1)]:
 with tempfile.TemporaryDirectory(prefix='astra_run508_cleanup_') as tmp:
  out=Path(tmp);(out/'patch_state').mkdir();(out/'patch_state/manifest.json').write_text('{}')
  script=f'''set -euo pipefail
OUT={out}
ROOT=/fake
CONTAINER=fake
SOURCES=(/fake/source)
RUN_STARTED=1
bash() {{ return 0; }}
curl() {{ return 1; }}
docker() {{ return 0; }}
npu-smi() {{ return 0; }}
python3() {{ if [[ " $* " == *' --final-only '* ]]; then return {finalgate}; fi; return 0; }}
sha256sum() {{ return 0; }}
cmp() {{ return 0; }}
{functions}
trap cleanup EXIT
exit {runexit}
'''
  p=subprocess.run(['bash','-c',script],capture_output=True,text=True);ck('cleanup_'+label,p.returncode==expected,p.stderr)
ck('Run507_all_raw_files_unmodified',all(Path(p).is_file() and sha(Path(p).read_bytes())==s for p,s in rawpins.items()))
ck('reviewed_sources_unmodified',all(sha(Path(p).read_bytes())==s for p,s in pins.items()))
write(O/'cpu_review_results.json',dict(checks=checks,all_expected=all(c['passed'] for c in checks),source_sha256=pins,raw_input_files=len(rawpins),scope='Source-only CLI and synthetic temporary fixture review; original Run507 remains INVALID.'))
(O/'reviewed_sources').mkdir(exist_ok=True)
for p in inputs:shutil.copyfile(p,O/'reviewed_sources'/p.name)
print(json.dumps(dict(checks=checks,all_expected=all(c['passed'] for c in checks),source_sha256=pins),indent=2))
