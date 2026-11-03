"""Identify only the exact archived served/speculative model metadata roots."""
from datetime import datetime, timezone
import json, subprocess
from finalize_origin_coverage_v9 import N, O, read, write
from server_readonly_v159 import ssh_args
from inspect_saved_raw_profile_v155 import BASE


def main():
    output=O/'checkpoint-roots-v159.json';receipt=N/'checkpoint-roots-receipt-v159.json';assert not output.exists()and not receipt.exists();container=read(O/'raw-record-schema-source-index-v155.json')['container_id']
    code='''import os,socket,json,subprocess,shlex,hashlib
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
base=Path(%r);container=%r;obj=json.loads(subprocess.run(['docker','inspect',container],capture_output=True,text=True,check=True).stdout)[0];assert obj['Id']==container and not obj['State']['Running']
serve=base/'runtime/scripts/serve.sh';raw=serve.read_bytes();lines=[x for x in raw.decode().splitlines()if x.startswith('exec /usr/local/python3.12.13/bin/vllm serve ')];assert len(lines)==1;argv=shlex.split(lines[0]);requested=[dict(role='target',container_path=argv[3])];selected_cli={}
for name in['--dtype','--load-format','--quantization','--speculative-config','--hf-overrides']:
 if name in argv:selected_cli[name]=argv[argv.index(name)+1]
spec=json.loads(selected_cli['--speculative-config'])if'--speculative-config'in selected_cli else None
if isinstance(spec,dict)and isinstance(spec.get('model'),str)and spec['model'].startswith('/'):
 requested.append(dict(role='draft',container_path=spec['model']))
roots=[]
for item in requested:
 path=Path(item['container_path']);mounts=[m for m in obj['Mounts']if m['Type']=='bind'and(path==Path(m['Destination'])or Path(m['Destination'])in path.parents)];assert mounts;mount=max(mounts,key=lambda m:len(Path(m['Destination']).parts));host=Path(mount['Source'])/path.relative_to(mount['Destination']);assert host.is_dir();entries=[]
 for p in sorted(host.iterdir()):
  s=p.lstat();entries.append(dict(name=p.name,bytes=s.st_size,is_file=p.is_file(),is_symlink=p.is_symlink()))
 assert len(entries)<500;roots.append(dict(**item,host_path=str(host),mount=dict(source=mount['Source'],destination=mount['Destination'],RW=mount['RW']),entries=entries))
print(json.dumps(dict(hostname=socket.gethostname(),euid=os.geteuid(),container_id=container,image=obj['Image'],container_running=False,serve_sha256=hashlib.sha256(raw).hexdigest(),selected_cli=selected_cli,roots=roots,read_only_filesystem_metadata=True,new_device_work=False)))
'''%(BASE,container)
    p=subprocess.run(ssh_args(code),capture_output=True,timeout=55);stdout=p.stdout.decode(errors='replace');stderr=p.stderr.decode(errors='replace');write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,stdout=stdout,stderr=stderr,no_device_work=True));assert p.returncode==0
    data=json.loads(stdout);assert data['hostname']=='bm-jn-zs-zone1-910C-64G-10-105'and data['euid']==0;write(output,data)
    print(json.dumps(dict(status='PASS',roots=[dict(role=x['role'],host_path=x['host_path'],files=len(x['entries']),metadata_files=[z['name']for z in x['entries']if z['name'].endswith('.json')])for x in data['roots']],selected_cli=data['selected_cli'],new_device_work=False)))


if __name__=='__main__':main()
