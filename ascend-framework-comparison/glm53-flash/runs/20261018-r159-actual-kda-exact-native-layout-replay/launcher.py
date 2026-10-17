import pathlib,json,hashlib,os,base64,subprocess,shlex,datetime
R=pathlib.Path(__file__).resolve().parent
P=R.parents[1]
for f,sha in json.loads((R/'prerequisites.json').read_text())['files_sha256'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==sha
assert json.loads((P/'runs/20261018-r157-native-smem-runtime-alias-cache-sdma/acceptance.json').read_text())['passed'];assert json.loads((P/'runs/20261018-r156-actual-kda-cpu-replay-inputs/analysis.json').read_text())['completed']
review=json.loads((R/'review.json').read_text())
for f,sha in review['files_sha256'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==sha,f
fd=os.open(R/'once.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
ssh=['ssh','-S','/Users/wio/work/sglang-glm53-pd/ssh-control','-p','60011','-o','BatchMode=yes','-o','ConnectTimeout=15','root@61.241.77.34']
assert subprocess.run(ssh[:-1]+['-O','check',ssh[-1]],capture_output=True).returncode==0
pod='zhixing-dockerctl-25xz2';identity=subprocess.run(ssh+[shlex.join(['kubectl','get','pod','-n','default',pod,'-o','jsonpath={.status.hostIP}'])],capture_output=True,text=True,timeout=25);assert identity.returncode==0 and identity.stdout.strip()=='172.16.12.14'
files={name:{'data':base64.b64encode((R/f).read_bytes()).decode(),'sha256':review['files_sha256'][f]} for name,f in {'probe.py':'container_probe.py','candidate_kda.py':'candidate_kda.py','expected-native-sources.json':'expected-native-sources.json',**{f:f for f in json.loads((R/'mounts.json').read_text())}}.items()}
code=(R/'host_probe.py').read_text().replace('__FILES__',repr(files)).replace('__MOUNTS__',repr(json.loads((R/'mounts.json').read_text())))
cmd=['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']
out={'project_id':'glm53-flash','started':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=subprocess.run(ssh+[shlex.join(cmd)],input=code,capture_output=True,text=True,timeout=620)
result={'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr};(R/'node14.json').write_text(json.dumps(result,indent=2));out['rc']=p.returncode;out['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat();(R/'receipt.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
