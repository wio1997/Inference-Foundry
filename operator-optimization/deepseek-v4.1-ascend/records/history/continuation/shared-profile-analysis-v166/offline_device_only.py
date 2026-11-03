"""One bounded CPU-only derivation of the 32 closed formal NPU profiles."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path

C=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/npu166')
H=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/npu166-official-db')
I='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis/npu166-official-db'
MAIN='wio-vllm-v41-repro171-20261102'

def write(name,d):
    p=H/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def pin(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}

def run():
    status={'state':'COPYING_CLOSED_NPU_ONLY_PROFILES','started_unix':time.time(),'pid':os.getpid(),
            'scope':'CPU offline vendor DB export; no model load, inference or capture; acquisition originals untouched',
            'source':str(C/'raw'),'output':str(H),'max_process_number':4}
    write('STATUS.json',status)
    receipt=json.loads((C/'STATUS.json').read_text())
    assert receipt['case_valid'] and receipt['exact_response_text_matches']==8 and receipt['final_stop_profile_http']==200
    sources=sorted((C/'raw').glob('*_ascend_pt'));assert len(sources)==32
    out=H/'profiles';out.mkdir()
    subprocess.run(['cp','-a','--reflink=auto',*[str(p) for p in sources],str(out)],check=True)
    manifest=[]
    for source in sources:
        for p in sorted(source.rglob('*')):
            if not p.is_file():continue
            q=out/source.name/p.relative_to(source)
            a=pin(p);b=pin(q)
            if a!=b:raise ValueError('Copied raw input mismatch: '+str(p))
            manifest.append({'profile':source.name,'relative':str(p.relative_to(source)),**a})
    write('INPUT-MANIFEST.json',{'profiles':32,'files':manifest,'original_and_copy_verified_equal':True})
    script="""import os,json,time,traceback
from pathlib import Path
import torch
torch.set_num_threads(1)
from torch_npu.profiler.profiler import analyse
root=Path(%r)
analyse(str(root/'profiles'),max_process_number=4,export_type='db')
dbs=[{'path':str(p),'bytes':p.stat().st_size} for p in sorted((root/'profiles').rglob('*.db'))]
(root/'EXPORT-RETURNED.json').write_text(json.dumps({'returned_unix':time.time(),'databases':dbs},indent=2)+'\\n')
print(json.dumps({'returned':True,'databases':len(dbs)}))
"""%I
    (H/'inside_export.py').write_text(script)
    (H/'tmp').mkdir()
    status['state']='VENDOR_ANALYSE_RUNNING';status['copied_input_files']=len(manifest);write('STATUS.json',status)
    with (H/'vendor-export.log').open('w') as log:
        p=subprocess.Popen(['docker','exec','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','TMPDIR='+I+'/tmp',MAIN,'python3',I+'/inside_export.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status['analysis_child_pid']=p.pid;write('STATUS.json',status)
        try:rc=p.wait(timeout=2700)
        except subprocess.TimeoutExpired:
            # Terminate only the exact analysis invocation, never the inference service.
            subprocess.run(['docker','exec',MAIN,'pkill','-TERM','-f','^python3 '+I+'/inside_export.py$'],check=False)
            rc=p.wait(timeout=60);raise TimeoutError('CPU analysis deadline exceeded; child rc='+str(rc))
    status.update(state='VENDOR_ANALYSE_RETURNED' if rc==0 else 'VENDOR_ANALYSE_FAILED',returncode=rc,finished_unix=time.time())
    if (H/'EXPORT-RETURNED.json').exists():status['export']=json.loads((H/'EXPORT-RETURNED.json').read_text())
    write('STATUS.json',status)

if __name__=='__main__':
    H.mkdir(parents=True,exist_ok=True)
    try:run()
    except Exception:
        write('FAILURE.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
