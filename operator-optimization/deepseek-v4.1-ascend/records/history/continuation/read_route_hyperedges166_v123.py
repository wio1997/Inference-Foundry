"""Read saved routing evidence through an already authenticated source-host socket."""
from datetime import datetime,timezone
import base64,gzip,hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,write


def main():
    receipt=N/'route-hyperedges166-v123.json'
    assert not receipt.exists()
    parser=N/'saved_tensor_archive_v122.py';worker=N/'export_route_hyperedges_saved_v123.py'
    source=worker.read_text();line='from saved_tensor_archive_v122 import load_saved_tensor_list\n'
    assert line in source
    source=parser.read_text()+'\n'+source.replace(line,'')
    source=source.replace('REPLACE_INDEX_SHA',hashlib.sha256((N/'route-snapshot-source-v122/tensor-index.json').read_bytes()).hexdigest())
    source=source.replace('REPLACE_DIST_SHA',hashlib.sha256((N/'MC2-operator-source-v56/saved-observer/EP-count-distribution.json').read_bytes()).hexdigest())
    compile(source,'<bounded-saved-routing-hyperedges>','exec')
    control=N/'ssh166-route-v123.sock'
    assert control.exists()
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(control),'-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes',
        '-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(source)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=60)
    assert len(p.stdout)<20*1024*1024
    result=None
    for output_line in p.stdout.splitlines():
        try:
            d=json.loads(output_line)
            if isinstance(d,dict)and d.get('status')=='SAVED_ROUTE_HYPEREDGES_EXPORTED':
                result=d
        except ValueError:
            pass
    if result is None:
        write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,
            result=None,stderr=p.stderr[-4000:],payload_sha256=hashlib.sha256(source.encode()).hexdigest()))
        raise AssertionError((p.returncode,p.stderr[-1500:]))
    compressed=base64.b64decode(result.pop('compressed_base64'),validate=True)
    assert len(compressed)==result['compressed_bytes'] and hashlib.sha256(compressed).hexdigest()==result['compressed_sha256']
    packed=gzip.decompress(compressed)
    assert len(packed)==result['packed_bytes']<=10*1024*1024 and hashlib.sha256(packed).hexdigest()==result['packed_sha256']
    out=N/'route-hypergraph-source-v123';out.mkdir(exist_ok=True)
    (out/'active-routes.bin.gz').write_bytes(compressed)
    write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,
        stderr=p.stderr[-4000:],payload_sha256=hashlib.sha256(source.encode()).hexdigest()))
    write(O/'route-hyperedges-source-index-v123.json',result)
    print(json.dumps({k:result[k]for k in ['status','records','formal_cohorts','active_rows','read_bytes',
        'packed_bytes','compressed_bytes','archive_reader_elapsed_wall_seconds','new_device_work']}))


if __name__=='__main__':
    main()
