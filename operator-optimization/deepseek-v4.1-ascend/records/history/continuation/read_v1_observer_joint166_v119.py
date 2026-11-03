"""Read one previously saved, SHA-pinned compressed metadata file; no device or container work."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,write

def main():
    receipt=N/'v1-observer-joint166-v119.json';assert not receipt.exists()
    code='''import socket,os,json,hashlib,base64
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/observer-20261030T160315Z-76b79f0f/value-review/decoded-records.jsonl.gz')
assert p.is_file()and p.stat().st_size<8*1024*1024
b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
assert h=='d6e787e5bd8498c8e68ccd3f2d1874a8dfd20260071fae7f16ad6876ec3c3029'
print(json.dumps(dict(hostname=socket.gethostname(),path=str(p),bytes=len(b),sha256=h,source_b64=base64.b64encode(b).decode(),read_only=True,no_container_or_device_work=True)))
'''
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(N/'ssh166-v1-v119.sock'),'-o','ControlMaster=no','-o','ProxyCommand=false',
        '-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=60);result=None
    for line in p.stdout.splitlines():
        try:
            item=json.loads(line)
            if isinstance(item,dict)and item.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=item
        except ValueError:pass
    write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,stderr=p.stderr[-2000:]))
    assert result is not None,(p.returncode,p.stderr[-1000:])
    b=base64.b64decode(result['source_b64']);assert hashlib.sha256(b).hexdigest()==result['sha256']
    target=N/'MC2-V1-observer-source-v119/decoded-records.jsonl.gz';target.parent.mkdir(exist_ok=True,parents=True);assert not target.exists();target.write_bytes(b)
    meta={k:v for k,v in result.items()if k!='source_b64'};meta['local_path']=str(target)
    write(O/'combine-v1-observer-source-index-v119.json',meta)
    print(json.dumps(dict(status='READ_ONE_SAVED_METADATA_FILE',bytes=len(b),sha256=result['sha256'],new_device_work=False)))

if __name__=='__main__':main()
