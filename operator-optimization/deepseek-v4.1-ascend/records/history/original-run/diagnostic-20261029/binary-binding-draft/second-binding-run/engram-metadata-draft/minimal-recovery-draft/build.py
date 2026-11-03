"""Generate independent disabled adapter copies with run cancellation under the existing fence."""
import ast, pathlib, sys, hashlib
if sys.flags.optimize:raise RuntimeError('optimized Python forbidden')
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parent

# The marker resides in the stage container's *existing* shared-fence directory.
# Both check and create execute only after acquiring owner-fence.lock.
CHECK='''\nensure_owner_fence(fence_fd)
marker_dir='/work/'+LOCK_FOR_CANCEL[len(HOST_WORK_ROOT):]
marker=marker_dir+'/cancel-'+RUN_ID
probe="import os,sys\\nif sys.flags.optimize: raise RuntimeError('optimized Python forbidden')\\ntry: os.lstat(sys.argv[1])\\nexcept FileNotFoundError: sys.exit(0)\\nsys.exit(3)"
check=subprocess.run(['docker','exec',STAGE,'python3','-c',probe,marker],capture_output=True,timeout=20)
ensure_owner_fence(fence_fd)
if check.returncode!=0:raise RuntimeError('run cancelled or cancellation state unavailable')
'''
REVOKE='''\nensure_owner_fence(fence_fd)
marker_dir='/work/'+LOCK_ROOT[len(HOST_WORK_ROOT):]
marker=marker_dir+'/cancel-'+RUN_ID
create="import os,sys; sys.flags.optimize and sys.exit(4); p=sys.argv[1]; fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600); os.write(fd,b'cancelled'); os.fsync(fd); os.close(fd)"
r=subprocess.run(['docker','exec',STAGE,'python3','-c',create,marker],capture_output=True,timeout=20)
ensure_owner_fence(fence_fd)
if r.returncode:
    probe="import os,sys,stat; sys.flags.optimize and sys.exit(4); s=os.lstat(sys.argv[1]); sys.exit(0 if stat.S_ISREG(s.st_mode) and s.st_nlink==1 else 2)"
    if subprocess.run(['docker','exec',STAGE,'python3','-c',probe,marker],capture_output=True,timeout=20).returncode:
        raise RuntimeError('cannot persist cancellation marker')
ensure_owner_fence(fence_fd)
'''
LOCK_EXPR="ROOT + '/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic'"
FILES={'engram_adapter_start.py':('REMOTE',),'engram_adapter_supervise.py':('CAPTURE','RESTORE'),'engram_adapter_recover.py':('REMOTE',)}
for filename,names in FILES.items():
    tree=ast.parse((BASE/filename).read_text())
    seen=set()
    for n in tree.body:
        if not isinstance(n,ast.Assign) or len(n.targets)!=1 or not isinstance(n.targets[0],ast.Name):continue
        name=n.targets[0].id
        if name not in names:continue
        if not isinstance(n.value,ast.Constant) or not isinstance(n.value.value,str):continue
        src=n.value.value
        line=next((x for x in src.splitlines() if x.startswith('fence_fd = acquire_owner_fence(')),None)
        if not line:raise RuntimeError('missing exact fence acquisition')
        if filename.endswith('recover.py'):insert=REVOKE
        else:insert='\nLOCK_FOR_CANCEL='+LOCK_EXPR+CHECK
        src=src.replace(line,line+insert,1);compile(src,filename+':'+name,'exec')
        n.value=ast.Constant(src);seen.add(name)
    if seen!=set(names):raise RuntimeError('missing remote blocks')
    result=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    result=result.replace("HERE.parent / 'remote_guard.py'","HERE.parent.parent / 'remote_guard.py'").replace("HERE.parent / 'engram-audited-bootstrap'","HERE.parent.parent / 'engram-audited-bootstrap'").replace("HERE / 'bootstrap-approved-sha256.json'","HERE.parent / 'bootstrap-approved-sha256.json'").replace("HERE / 'engram_client.py'","HERE.parent / 'engram_client.py'")
    if 'DISABLED:' not in result:raise RuntimeError('entry not disabled')
    compile(result,filename,'exec')
    out=HERE/filename;out.write_text(result);out.chmod(0o600)
    print(filename,hashlib.sha256(result.encode()).hexdigest())
