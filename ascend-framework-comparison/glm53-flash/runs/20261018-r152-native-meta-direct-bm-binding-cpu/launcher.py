import pathlib,json,hashlib,os,base64,subprocess,shlex,datetime
R=pathlib.Path(__file__).resolve().parent
prereq=R.parent/'20261017-r151-native-meta-direct-binding-cpu/analysis.json';assert hashlib.sha256(prereq.read_bytes()).hexdigest()=='83c8350fb4493dca47abbe89ce1d5e38ed94d7fa06dfe75226b1b27c733a446c';prior=json.loads(prereq.read_text());assert prior['passed'] is False and prior['parent0_idle8'] and prior['native_main_called'] is False
review=json.loads((R/'review.json').read_text())
for f,sha in review['files_sha256'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==sha,f
fd=os.open(R/'once.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
ssh=['ssh','-S','/Users/wio/work/sglang-glm53-pd/ssh-control','-p','60011','-o','BatchMode=yes','-o','ConnectTimeout=15','root@61.241.77.34']
assert subprocess.run(ssh[:-1]+['-O','check',ssh[-1]],capture_output=True).returncode==0
pod='zhixing-dockerctl-42pbf';identity=subprocess.run(ssh+[shlex.join(['kubectl','get','pod','-n','default',pod,'-o','jsonpath={.status.hostIP}'])],capture_output=True,text=True,timeout=25);assert identity.returncode==0 and identity.stdout.strip()=='172.16.12.15'
files={name:{'data':base64.b64encode((R/f).read_bytes()).decode(),'sha256':review['files_sha256'][f]} for name,f in {'probe.py':'container_probe.py','meta_child.py':'meta_child.py','mmc-meta.conf':'mmc-meta.conf'}.items()}
code=(R/'host_probe.py').read_text().replace('__FILES__',repr(files)).replace('__MOUNTS__',repr({}))
cmd=['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']
out={'project_id':'glm53-flash','started':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=subprocess.run(ssh+[shlex.join(cmd)],input=code,capture_output=True,text=True,timeout=620)
result={'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr};(R/'node15.json').write_text(json.dumps(result,indent=2));out['rc']=p.returncode;out['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat();(R/'receipt.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
