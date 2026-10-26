"""Local publication state-machine fixture; git/network commands are all mocked."""
import hashlib,importlib.util,json,subprocess,tempfile,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'scripts/publish_deployment_checkpoint_v1.py';s=importlib.util.spec_from_file_location('publication',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def main():
 source=m.RUN/'publication-manifest.json';manifest=json.loads(source.read_text());selected={m.PREFIX+p for p in manifest['files']};cases=[]
 for scenario in ['publish_with_shared_remote_merge','remote_selected_collision_stops','preexisting_index_stops']:
  with tempfile.TemporaryDirectory(prefix='glm53-publish-fixture-') as td:
   m.RUN=Path(td);(m.RUN/'publication-manifest.json').write_bytes(source.read_bytes());calls=[];staged=set();head='0'*40;remote='2'*40;pushed=False
   def fake_run(command,**kwargs):
    nonlocal head,pushed
    assert command[0]=='git';args=command[1:];calls.append(args);stdout=b'';stderr=b'';rc=0
    assert not any(x in args for x in ['--force','--force-with-lease','reset','clean','rebase','stash','checkout'])
    if args==['rev-parse','--show-toplevel']:stdout=(str(ROOT.parents[1])+'\n').encode()
    elif args==['branch','--show-current']:stdout=(m.BRANCH+'\n').encode()
    elif args==['remote','get-url','origin']:stdout=b'https://github.com/wio1997/Inference-Foundry.git\n'
    elif args==['diff','--cached','--name-only','-z']:stdout=b'foreign/staged\0' if scenario=='preexisting_index_stops' else b''.join(x.encode()+b'\0' for x in staged)
    elif args==['diff','--name-only','-z']:stdout=b''
    elif args==['rev-parse','HEAD']:stdout=(head+'\n').encode()
    elif args[0]=='add':staged.update(args[2:])
    elif args[:3]==['diff','--cached','--check']:pass
    elif args[0]=='commit':head='1'*40
    elif args[0]=='fetch':pass
    elif args==['rev-parse','FETCH_HEAD']:stdout=((head if pushed else remote)+'\n').encode()
    elif args[:2]==['merge-base','--is-ancestor']:rc=0 if pushed else 1
    elif args[0]=='merge-base':stdout=('0'*40+'\n').encode()
    elif args[:3]==['diff','--name-only','-z']:stdout=((next(iter(selected)) if scenario=='remote_selected_collision_stops' else 'unrelated-existing-project/file')+'\0').encode()
    elif args[0]=='merge':head='3'*40
    elif args[0]=='show':
     relative=args[1].split(':',1)[1];assert relative in selected;stdout=(ROOT/relative[len(m.PREFIX):]).read_bytes()
    elif args[0]=='push':pushed=True
    else:raise AssertionError(('Unexpected mock git command',args))
    return types.SimpleNamespace(returncode=rc,stdout=stdout,stderr=stderr)
   original=subprocess.run;subprocess.run=fake_run
   error=None
   try:m.main()
   except AssertionError as e:error=repr(e)
   finally:subprocess.run=original
   receipt=json.loads((m.RUN/'publication-receipt.json').read_text())
   assert receipt['pushed']==(scenario=='publish_with_shared_remote_merge')
   assert receipt['committed']==(scenario!='preexisting_index_stops')
   if scenario!='publish_with_shared_remote_merge':assert error and not any(c[0] in ['merge','push'] for c in calls)
   cases.append({'scenario':scenario,'mock_git_calls':len(calls),'mock_commit':receipt['committed'],'mock_push':receipt['pushed'],'network_calls':0})
 result={'project_id':'glm53','pass':True,'launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':cases,'resource_actions':0,'actual_git_mutations':0};(ROOT/'results/publication-local-fixture-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
