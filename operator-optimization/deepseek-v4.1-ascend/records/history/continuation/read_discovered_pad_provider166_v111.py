"""Pin installed ConstantPadNd/MemSet provider files by exact operator paths;no device work."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,write

def main():
    receipt=N/'pad-provider-source166-v111b.json';assert not receipt.exists()
    base='/usr/local/Ascend/ascend-toolkit/latest/'
    import re
    from finalize_origin_coverage_v9 import read
    outer=read(N/'pad-provider-inventory166-v111.json')
    inventory=json.loads(json.loads(outer['stdout'])['stdout'])
    assert inventory['status']=='DONE' and inventory['match_count']<=200
    suffixes=['include/acl/acl_rt.h','include/aclnn/opdev/op_dfx.h',
      'opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/mem_set.py',
      'opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/mem_set.py',
      'opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/pad_v3.py',
      'opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910_93/ops_legacy/mem_set.json',
      'opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910_93/ops_legacy/pad_v3.json']
    paths=[base+x for x in suffixes]
    paths += [p for p in inventory['matches']if re.fullmatch(re.escape(base)+r'opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_legacy/mem_set/MemSet_[0-9a-f]+_high_performance\.json',p)]
    assert len(paths)<=28
    code='''import socket,os,json,subprocess,tarfile,io,hashlib,base64
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
cap=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/profile-capabilities-index-v12-v2.json')
assert hashlib.sha256(cap.read_bytes()).hexdigest()=='79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095'
cid='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d';rows=[]
for path in PATHS:
 p=subprocess.run(['docker','cp','-L',cid+':'+path,'-'],capture_output=True,timeout=12)
 if p.returncode:rows.append(dict(path=path,status='READ_ERROR',returncode=p.returncode,stderr=p.stderr.decode(errors='replace')[-700:]));continue
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as t:
  fs=[m for m in t.getmembers()if m.isfile()];assert len(fs)==1 and fs[0].size<2*1024*1024;b=t.extractfile(fs[0]).read()
 h=hashlib.sha256(b).hexdigest()
 if path.endswith('/acl/acl_rt.h'):assert h=='62d127fbb899b1476e054b1815ca6e0c71a362868c655f0e3da4616fb7598fb7'
 rows.append(dict(path=path,status='READ',bytes=len(b),sha256=h,source_b64=base64.b64encode(b).decode()))
print(json.dumps(dict(hostname=socket.gethostname(),container=cid,files=rows,read_only_archive_copy=True,new_container_process=False,NPU_Torch_or_library_load=False)))
'''.replace('PATHS',repr(paths))
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(N/'ssh166-pad-v111.sock'),'-o','ControlMaster=no','-o','ProxyCommand=false',
        '-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=100);result=None
    for line in p.stdout.splitlines():
        try:
            d=json.loads(line)
            if isinstance(d,dict)and d.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=d
        except ValueError:pass
    write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,stderr=p.stderr[-3000:]))
    assert result is not None,(p.returncode,p.stderr[-1000:]);rows=[]
    for item in result['files']:
        row={k:v for k,v in item.items()if k!='source_b64'}
        if item['status']=='READ':
            b=base64.b64decode(item['source_b64']);assert hashlib.sha256(b).hexdigest()==item['sha256']
            q=N/'pad-provider-discovered-v111'/Path(item['path']).relative_to('/');q.parent.mkdir(exist_ok=True,parents=True);assert not q.exists();q.write_bytes(b);row['local_path']=str(q)
        rows.append(row)
    write(O/'pad-provider-discovered-index-v111.json',dict(files=rows,installed_CANN_header_control=True,new_device_work=False,
        operator_names_not_historical_loaded_ABI=True,overall_complete=False))
    print(json.dumps(dict(status='READ_ONLY_OPERATOR_FILES',files=[{k:r[k]for k in ['path','status']}for r in rows])))

if __name__=='__main__':main()
