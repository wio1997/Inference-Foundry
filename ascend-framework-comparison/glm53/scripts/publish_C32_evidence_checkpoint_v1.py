"""Publish exact manifest of own source/config/aggregate evidence; no raw runs or force push."""
import ast,datetime,hashlib,json,os,re,subprocess,sys,urllib.parse
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1]
RUN=ROOT/'runs/20261017T2071-git-C32-evidence-checkpoint-v1'
PREFIX='ascend-framework-comparison/glm53/';BRANCH='codex/ascend-sglang-vllm-comparison'
def main():
 assert len(sys.argv)==2 and re.fullmatch('[0-9a-f]{64}',sys.argv[1])
 manifest_path=RUN/'publication-manifest.json';assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==sys.argv[1]
 manifest=json.loads(manifest_path.read_text());assert manifest['project_id']=='glm53' and manifest['raw_capture_paths_excluded'] and manifest['original_measurement_certificates_preserved_unmodified']
 with (RUN/'publication-once.lock').open('x') as f:f.write(str(os.getpid()))
 repo=ROOT.parents[1];assert (repo/PREFIX).resolve()==ROOT
 env=os.environ|{'GIT_TERMINAL_PROMPT':'0','GIT_EDITOR':'true','GIT_SEQUENCE_EDITOR':'true','GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'http.version','GIT_CONFIG_VALUE_0':'HTTP/1.1'}
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'manifest_sha256':sys.argv[1],'branch':BRANCH,'selected_files':len(manifest['files']),'committed':False,'pushed':False,'force_push':False,'events':[]}
 sequence=0
 def git(args,timeout=60,allow_failure=False):
  nonlocal sequence
  sequence+=1;r=subprocess.run(['git',*args],cwd=repo,env=env,capture_output=True,timeout=timeout)
  (RUN/f'git-{sequence:02d}.stdout').write_bytes(r.stdout);(RUN/f'git-{sequence:02d}.stderr').write_bytes(r.stderr)
  receipt['events'].append({'sequence':sequence,'git_args':args,'rc':r.returncode,'stdout_sha256':hashlib.sha256(r.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr).hexdigest()})
  if not allow_failure:assert r.returncode==0,(args,r.returncode,r.stderr.decode(errors='replace')[-2000:])
  return r
 def names(raw):return set(x for x in raw.decode().split('\0') if x)
 def verify_files():
  for relative,item in manifest['files'].items():
   p=ROOT/relative;assert p.resolve().is_relative_to(ROOT) and not p.is_symlink() and not relative.startswith('runs/')
   raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256'] and len(raw)==item['bytes'],relative
 try:
  assert git(['rev-parse','--show-toplevel']).stdout.decode().strip()==str(repo)
  assert git(['branch','--show-current']).stdout.decode().strip()==BRANCH
  remote=git(['remote','get-url','origin']).stdout.decode().strip();u=urllib.parse.urlsplit(remote)
  assert u.scheme=='https' and u.hostname=='github.com' and not u.username and not u.password and u.path.rstrip('/')=='/wio1997/Inference-Foundry.git'
  assert not names(git(['diff','--cached','--name-only','-z']).stdout)
  selected={PREFIX+p for p in manifest['files']};assert selected and names(git(['diff','--name-only','-z']).stdout)<=selected
  verify_files();receipt['head_before']=git(['rev-parse','HEAD']).stdout.decode().strip()
  git(['add','--',*sorted(selected)])
  staged=names(git(['diff','--cached','--name-only','-z']).stdout);assert staged and staged<=selected
  receipt['own_changed_paths']=len(staged)
  receipt['whitespace_check_rc']=git(['diff','--cached','--check','--',PREFIX,':(exclude)*.patch'],allow_failure=True).returncode
  git(['commit','-m','Record C32 control, native diagnostic and docs-first MemCache review','-m','Accept128of128 SGLangL04currentC32 at155.9674tokens/s; fixedSLAfailed. Preserve rawduplicate-stopHTTP500 alongside independentlyverified32ranknativeprofilecompletion. FreezeonlyDnativeOpscandidate andacceptedL04rollback, source/passthrough/configpreflightqualified butNOTmodeldeployed. SSHbannerexchangetimeoutpreventedshutdown; servicesuntouched. C32only/noC16C64L05extension. Publishsource/config/compactaggregateswithoutrequestIDs/text/rawlogs/traces/weights/SDKbinaries.'],120)
  receipt['committed']=True;receipt['checkpoint_commit']=git(['rev-parse','HEAD']).stdout.decode().strip()
  git(['fetch','--no-tags','origin','refs/heads/'+BRANCH],120)
  remote_tip=git(['rev-parse','FETCH_HEAD']).stdout.decode().strip()
  ancestor=git(['merge-base','--is-ancestor',remote_tip,'HEAD'],allow_failure=True)
  if ancestor.returncode:
   assert ancestor.returncode==1;common=git(['merge-base','HEAD',remote_tip]).stdout.decode().strip()
   changed=names(git(['diff','--name-only','-z',common,remote_tip]).stdout);assert not changed&selected,('Remote selected-path collision; preserve without overwrite',sorted(changed&selected))
   git(['merge','--no-edit',remote_tip],120)
  verify_files()
  for path,item in manifest['files'].items():assert hashlib.sha256(git(['show','HEAD:'+PREFIX+path]).stdout).hexdigest()==item['sha256'],path
  git(['push','origin','HEAD:refs/heads/'+BRANCH],120);receipt['pushed']=True
  receipt['published_commit']=git(['rev-parse','HEAD']).stdout.decode().strip()
  git(['fetch','--no-tags','origin','refs/heads/'+BRANCH],120)
  receipt['remote_tip_observed']=git(['rev-parse','FETCH_HEAD']).stdout.decode().strip()
  git(['merge-base','--is-ancestor',receipt['published_commit'],receipt['remote_tip_observed']]);receipt['remote_contains_checkpoint']=True
  print(json.dumps({k:v for k,v in receipt.items() if k!='events'}),flush=True)
 except BaseException as error:receipt['error']=repr(error);raise
 finally:
  receipt['as_of']=datetime.datetime.now(datetime.timezone.utc).isoformat();(RUN/'publication-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
