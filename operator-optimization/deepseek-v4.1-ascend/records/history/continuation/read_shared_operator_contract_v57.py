"""Point-read the shared INT8 operator caller and optional fused-op SDK declarations."""
from pathlib import Path
import hashlib,base64,json
from datetime import datetime,timezone
from finalize_origin_coverage_v9 import N,O,write
import mirror_saved_profile_core_v53 as transport
from read_original_source_archive_v53 import CONTAINER
transport.CONTROL=N/'ssh166-v56-xinyufeng.sock'


def main():
    assert not(O/'shared-operator-inputs-v57.json').exists()
    paths=['/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/shared_experts.py']
    for name in ['aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h','aclnn_grouped_matmul_swiglu_quant_v2.h','aclnn_dequant_swiglu_quant_v2.h']:
        paths.extend('/usr/local/Ascend/ascend-toolkit/latest/'+p+'/'+name for p in ['include/aclnnop','opp/built-in/op_api/include'])
    code='''import socket,os,subprocess,tarfile,io,json,hashlib,base64
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
rows=[]
for path in %r:
 p=subprocess.run(['docker','cp','-L',%r+':'+path,'-'],capture_output=True,timeout=20)
 if p.returncode:
  rows.append(dict(path=path,status='UNAVAILABLE_AT_EXACT_PATH',stderr=p.stderr.decode(errors='replace')[-400:]));continue
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as tar:
  fs=[x for x in tar.getmembers()if x.isfile()];assert len(fs)==1 and fs[0].size<1024*1024
  b=tar.extractfile(fs[0]).read()
 sha=hashlib.sha256(b).hexdigest()
 if path.endswith('/shared_experts.py'):assert sha=='471db26e9e9e4ba70c2ccd43502ad39bff6699bbd4ca191319a9b66fc20458f9'
 rows.append(dict(path=path,status='READ',sha256=sha,bytes=len(b),data=base64.b64encode(b).decode()))
print(json.dumps(dict(files=rows,container_process_started=False)))
'''%(paths,CONTAINER)
    p=transport.run_readonly_python(code,timeout=160)
    rec=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,stdout=p.stdout.decode(errors='replace'),stderr=p.stderr.decode(errors='replace'),no_model_device_work=True)
    write(N/'shared-operator-source166-v57.json',rec);assert p.returncode==0
    rows=[]
    for r in json.loads(rec['stdout'])['files']:
        if r['status']!='READ':rows.append(r);continue
        b=base64.b64decode(r['data']);assert hashlib.sha256(b).hexdigest()==r['sha256']
        dest=N/'shared-operator-source-v57'/Path(r['path']).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
        rows.append(dict({k:v for k,v in r.items()if k!='data'},local_path=str(dest)))
    write(O/'shared-operator-inputs-v57.json',dict(files=rows,source_scope='Exactcurrent166stagefilesystem;sharedcallerhistoricalhashmatches. Headerpresence alone isnot runtimekernel/binding/precision support.',no_new_model_Torch_NPU_capture_or_service_process_work=True,overall_complete=False))
    print(json.dumps(dict(read=[r['path']for r in rows if r['status']=='READ'],unavailable=sum(r['status']!='READ'for r in rows),overall_complete=False)))


if __name__=='__main__':main()
