"""Read saved routing tensor archive/index only; stdlib, no container or accelerator imports."""
from pathlib import Path
from datetime import datetime,timezone
import base64,gzip,hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,write

def main():
    receipt=N/'route-sample166-v122.json';assert not receipt.exists()
    records=[json.loads(l)for l in gzip.decompress((N/'MC2-V1-observer-source-v119/decoded-records.jsonl.gz').read_bytes()).splitlines()]
    sample=next(r for r in records if r['role']=='target'and r['M']==1023)
    code='''import os,socket,json,hashlib,base64
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
root=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/observer-20261030T160315Z-76b79f0f')
sample=SAMPLE
index_path=root/'tensor-index.json';assert index_path.stat().st_size<8*1024*1024
ib=index_path.read_bytes();index=json.loads(ib);assert len(index)==4736
item=index[sample['index']];assert item['tensor']==sample['tensor']and item['metadata']==sample['metadata']
files=[dict(name='tensor-index.json',source=str(index_path),bytes=len(ib),sha256=hashlib.sha256(ib).hexdigest(),source_b64=base64.b64encode(ib).decode())]
for field,name in [('tensor','sample.pt'),('metadata','sample-metadata.json')]:
 p=Path(item[field]);assert str(p).startswith(str(root/'runtime/observer')+'/')and p.stat().st_size<2*1024*1024
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==item[field+'_sha256']
 files.append(dict(name=name,source=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),source_b64=base64.b64encode(b).decode()))
print(json.dumps(dict(hostname=socket.gethostname(),files=files,sample_record=sample,CPU_stdlib_only=True,new_device_work=False)))
'''.replace('SAMPLE',repr(sample))
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(N/'ssh166-route-v122.sock'),'-o','ControlMaster=no','-o','ProxyCommand=false',
        '-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=45);result=None
    for line in p.stdout.splitlines():
        try:
            d=json.loads(line)
            if isinstance(d,dict)and d.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=d
        except ValueError:pass
    write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,stderr=p.stderr[-2000:]))
    assert result is not None,(p.returncode,p.stderr[-1000:]);folder=N/'route-snapshot-source-v122';folder.mkdir(exist_ok=False);rows=[]
    for r in result['files']:
        b=base64.b64decode(r['source_b64']);assert hashlib.sha256(b).hexdigest()==r['sha256'];q=folder/r['name'];q.write_bytes(b)
        rows.append(dict(path=str(q),source=r['source'],bytes=len(b),sha256=r['sha256']))
    write(O/'route-snapshot-source-index-v122.json',dict(files=rows,sample_record=sample,new_device_work=False,CPU_stdlib_only=True,overall_complete=False))
    print(json.dumps(dict(status='READ_SAVED_INDEX_AND_ONE_SAMPLE',files=[dict(name=r['name'],bytes=r['bytes'])for r in result['files']],new_device_work=False)))

if __name__=='__main__':main()
