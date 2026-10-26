"""Record actual source identity, then exec unchanged SGLang launch_server with frozen argv."""
import hashlib,json,os,pathlib,subprocess,sys,traceback
root=pathlib.Path('/sgl-workspace/sglang');work=pathlib.Path('/work');plan=json.loads((work/'plan.json').read_text());v={'status':'STARTING','node':plan['node'],'role':plan['role'],'argv':plan['argv'],'weights_loaded':False}
try:
 v['git_head']=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert v['git_head']==plan['source_commit']
 v['git_status']=subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)
 p=subprocess.run(['git','-C',str(root),'diff','--quiet','HEAD','--','python/sglang']);v['tracked_python_tree_matches_HEAD']=p.returncode==0;assert p.returncode==0,'Python source differs from pinned HEAD'
 extra=subprocess.check_output(['git','-C',str(root),'ls-files','--others','--exclude-standard','python/sglang'],text=True);assert not extra.strip(),'Untracked Python package source requires review'
 paths=subprocess.check_output(['git','-C',str(root),'ls-files','python/sglang'],text=True).splitlines();files={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths if p.endswith('.py') and (root/p).is_file()};v['source_file_count']=len(files);v['source_manifest_sha256']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest();(work/'source-manifest.json').write_text(json.dumps(files,sort_keys=True));v['status']='SOURCE_PREFLIGHT_PASSED';(work/'preflight.json').write_text(json.dumps(v,indent=2));print('L039_BOOTSTRAP='+json.dumps(v),flush=True)
 os.execv(sys.executable,[sys.executable,'-u','-m','sglang.launch_server',*plan['argv']])
except BaseException as e:v['status']='FAILED_BEFORE_SERVER';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()};(work/'preflight.json').write_text(json.dumps(v,indent=2));print('L039_BOOTSTRAP='+json.dumps(v),flush=True);sys.exit(1)
