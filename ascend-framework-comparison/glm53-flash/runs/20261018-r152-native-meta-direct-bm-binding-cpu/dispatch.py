import pathlib,json,hashlib,subprocess,sys,os
R=pathlib.Path(__file__).resolve().parent
for f,sha in json.loads((R/'review.json').read_text())['files_sha256'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==sha,f
assert not (R/'once.lock').exists()
fd=os.open(R/'dispatch.once.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
with (R/'runner.log').open('wb') as log:
 p=subprocess.Popen([sys.executable,str(R/'launcher.py')],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
(R/'dispatch.json').write_text(json.dumps({'pid':p.pid,'project_id':'glm53-flash','launcher':'launcher.py','detached_session':True}));print('DISPATCHED',p.pid)
