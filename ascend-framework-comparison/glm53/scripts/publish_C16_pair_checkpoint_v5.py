"""Publish only frozen glm53 source/config/compact evidence; merge remote, no force."""
import ast,datetime,hashlib,json,os,subprocess,urllib.parse
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1];RUN=ROOT/'runs/20261017T1926-git-C16-pair-checkpoint-v5'
MANIFEST_SHA='841c29dd2b4b39f92ff207044eebb17ecc6b0327f0ab97dce8135ffefd7455bd'
BRANCH='codex/ascend-sglang-vllm-comparison';PREFIX='ascend-framework-comparison/glm53/'

def main():
 manifest_path=RUN/'publication-manifest.json';assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==MANIFEST_SHA
 manifest=json.loads(manifest_path.read_text());assert manifest['project_id']=='glm53' and manifest['raw_capture_paths_excluded']
 with (RUN/'publication-once.lock').open('x') as file:file.write(str(os.getpid()))
 repo=ROOT.parents[1];assert (repo/PREFIX).resolve()==ROOT
 environment=os.environ|{'GIT_TERMINAL_PROMPT':'0','GIT_EDITOR':'true','GIT_SEQUENCE_EDITOR':'true','GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'http.version','GIT_CONFIG_VALUE_0':'HTTP/1.1'}
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'manifest_sha256':MANIFEST_SHA,'branch':BRANCH,'paths':len(manifest['files']),'events':[],'committed':False,'pushed':False,'force_push':False}
 sequence=0
 def git(args,timeout=60,allow_failure=False):
  nonlocal sequence
  sequence+=1
  result=subprocess.run(['git',*args],cwd=repo,env=environment,capture_output=True,timeout=timeout)
  (RUN/f'git-{sequence:02d}.stdout').write_bytes(result.stdout);(RUN/f'git-{sequence:02d}.stderr').write_bytes(result.stderr)
  receipt['events'].append({'sequence':sequence,'git_args':args,'rc':result.returncode,'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest()})
  if not allow_failure:assert result.returncode==0,(args,result.returncode,result.stderr.decode(errors='replace')[-3000:])
  return result
 def names(raw):return set(x for x in raw.decode().split('\0') if x)
 def verify_files():
  for relative,item in manifest['files'].items():
   p=ROOT/relative;assert p.resolve().is_relative_to(ROOT) if hasattr(p.resolve(),'is_relative_to') else str(p.resolve()).startswith(str(ROOT)+'/')
   raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256'] and len(raw)==item['bytes'],relative
 try:
  assert git(['rev-parse','--show-toplevel']).stdout.decode().strip()==str(repo)
  assert git(['branch','--show-current']).stdout.decode().strip()==BRANCH
  remote=git(['remote','get-url','origin']).stdout.decode().strip();url=urllib.parse.urlsplit(remote)
  assert url.scheme=='https' and url.hostname=='github.com' and not url.username and not url.password and url.path.rstrip('/')=='/wio1997/Inference-Foundry.git'
  assert not names(git(['diff','--cached','--name-only','-z']).stdout),'Existing index must remain untouched'
  selected={PREFIX+p for p in manifest['files']};assert selected and all(p.startswith(PREFIX) and not p.startswith(PREFIX+'runs/') for p in selected)
  assert names(git(['diff','--name-only','-z']).stdout)<=selected,'Unstaged tracked changes outside frozen snapshot; preserve them'
  verify_files();receipt['head_before']=git(['rev-parse','HEAD']).stdout.decode().strip()
  git(['add','--',*sorted(selected)])
  staged=names(git(['diff','--cached','--name-only','-z']).stdout);assert staged and staged<=selected
  # Patch context whitespace is intentional; preserve frozen upstream/native bytes.
  whitespace=git(['diff','--cached','--check','--',PREFIX,':(exclude)*.patch'],allow_failure=True)
  receipt['whitespace_check_rc']=whitespace.returncode
  git(['commit','-m','Record matched GLM53 C16 gap and bounded SGLang diagnosis','-m','Accept SG C16 64/64 at94.1764 tokens/s versus matchedvLLM441.2158, bothSLAfailed. Preserve rawSGcache-field rejection and qualify nullable semantics from installedformatter and independent native evidence. Record actualvLLMC32v2, replace overstrictall-vLLM gate with matchingstage ordering, freeze boundedD-first diagnostic. No optimizationgain yet. Rawbusinessdata/logs/traces/binaries excluded.'],120)
  receipt['committed']=True;receipt['checkpoint_commit']=git(['rev-parse','HEAD']).stdout.decode().strip()
  for attempt in range(1,4):
   git(['fetch','--no-tags','origin','refs/heads/'+BRANCH],120)
   remote_tip=git(['rev-parse','FETCH_HEAD']).stdout.decode().strip()
   ancestor=git(['merge-base','--is-ancestor',remote_tip,'HEAD'],allow_failure=True)
   if ancestor.returncode!=0:
    assert ancestor.returncode==1
    common=git(['merge-base','HEAD',remote_tip]).stdout.decode().strip()
    upstream=names(git(['diff','--name-only','-z',common,remote_tip]).stdout)
    collision=upstream&selected;assert not collision,('Remote changed selected frozen paths; preserve both for explicit review',sorted(collision))
    git(['merge','--no-edit',remote_tip],120)
   verify_files()
   for path,item in manifest['files'].items():assert hashlib.sha256(git(['show','HEAD:'+PREFIX+path]).stdout).hexdigest()==item['sha256'],path
   pushed=git(['push','origin','HEAD:refs/heads/'+BRANCH],120,allow_failure=True)
   if pushed.returncode==0:
    receipt['pushed']=True;receipt['published_commit']=git(['rev-parse','HEAD']).stdout.decode().strip();receipt['push_attempts']=attempt;break
   error=pushed.stderr.decode(errors='replace')
   assert attempt<3 and '[rejected]' in error and ('fetch first' in error or 'non-fast-forward' in error),'Non-race push failure; no retry/force'
  assert receipt['pushed']
  git(['fetch','--no-tags','origin','refs/heads/'+BRANCH],120)
  observed=git(['rev-parse','FETCH_HEAD']).stdout.decode().strip();git(['merge-base','--is-ancestor',receipt['published_commit'],observed]);receipt['remote_tip_observed']=observed;receipt['remote_contains_checkpoint']=True
  print(json.dumps({k:v for k,v in receipt.items() if k!='events'}),flush=True)
 except BaseException as error:
  receipt['error']=repr(error);raise
 finally:
  receipt['as_of']=datetime.datetime.now(datetime.timezone.utc).isoformat();(RUN/'publication-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
