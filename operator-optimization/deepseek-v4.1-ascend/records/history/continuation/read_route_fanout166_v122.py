"""Execute bounded stdlib read-only archive analysis on the authorized source host."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    receipt=N/'route-fanout166-v122.json';assert not receipt.exists()
    parser=N/'saved_tensor_archive_v122.py';worker=N/'derive_route_fanout_saved_v122.py'
    source=worker.read_text();line='from saved_tensor_archive_v122 import load_saved_tensor_list\n';assert line in source
    source=parser.read_text()+'\n'+source.replace(line,'')
    index=N/'route-snapshot-source-v122/tensor-index.json';dist=N/'MC2-operator-source-v56/saved-observer/EP-count-distribution.json'
    source=source.replace('REPLACE_INDEX_SHA',hashlib.sha256(index.read_bytes()).hexdigest()).replace('REPLACE_DIST_SHA',hashlib.sha256(dist.read_bytes()).hexdigest())
    compile(source,'<bounded-saved-routing-analysis>','exec')
    control=N/'ssh166-route-v122.sock';assert control.exists()
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(control),'-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes',
        '-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(source)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=55);result=None
    for line in p.stdout.splitlines():
        try:
            d=json.loads(line)
            if isinstance(d,dict)and d.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=d
        except ValueError:pass
    write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,
        stderr=p.stderr[-4000:],payload_sha256=hashlib.sha256(source.encode()).hexdigest()))
    assert result is not None,(p.returncode,p.stderr[-1500:])
    write(O/'route-fanout-saved-records-v122.json',result)
    print(json.dumps(dict(status=result['status'],snapshot_records=result['snapshot_records'],read_bytes=result['read_bytes'],elapsed_seconds=result['elapsed_CPU_process_seconds'],groups=result['groups'],new_device_work=False)))

if __name__=='__main__':main()
