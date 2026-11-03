"""Inside the existing stage:32immutable raw profile copies -> CPU-only DB export."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, sqlite3, time

ROOT=Path('/work/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
RAW=ROOT/'runtime/official-traces'
JOB=ROOT/'analysis/offline-db-v12'
assert not JOB.exists(),'Do not repeat a live or completed export;inspect its exact process first.'
assert json.loads((ROOT/'controller-finished.json').read_text())['state']=='CAPTURED_AND_RESTORED'
assert os.environ.get('CANN_VERSION')=='9.2.0'
JOB.mkdir();copies=JOB/'profiles';copies.mkdir()
def state(name,**kw):
    obj=dict(state=name,pid=os.getpid(),time_utc=datetime.now(timezone.utc).isoformat(),**kw)
    raw=Path('/proc/self/stat').read_text();obj['process_starttime_ticks']=raw[raw.rfind(')')+2:].split()[19]
    tmp=JOB/'status.tmp';tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(JOB/'status.json')
    print(json.dumps({k:v for k,v in obj.items() if k not in ['databases','error']}),flush=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
try:
    sources=sorted(RAW.glob('*_ascend_pt'));assert len(sources)==32,len(sources)
    assert shutil.disk_usage(JOB).free>24*1024**3
    state('COPYING_ALL32_RAW_PROFILES')
    manifest=[]
    for source in sources:
        for p in source.rglob('*'):
            assert not p.is_symlink() and (p.is_file() or p.is_dir()),p
        target=copies/source.name;shutil.copytree(source,target,copy_function=shutil.copy2)
        for p in sorted(source.rglob('*')):
            if not p.is_file():continue
            before=p.stat();source_sha=sha(p);copy_sha=sha(target/p.relative_to(source));after=p.stat()
            assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
            assert source_sha==copy_sha
            manifest.append(dict(relative_path=str(p.relative_to(RAW)),bytes=before.st_size,sha256=source_sha,source_mtime_ns=before.st_mtime_ns))
    (JOB/'input-copy-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    tmp=JOB/'tmp';tmp.mkdir();os.environ['TMPDIR']=str(tmp)
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    import torch
    torch.set_num_threads(1)
    from torch_npu.profiler.profiler import analyse
    state('ANALYSING_ALL32_WITH_MAX4_PROFILE_PROCESSES',profile_directories=32,input_files=len(manifest),input_bytes=sum(x['bytes'] for x in manifest),export_type='db',new_device_capture=False)
    analyse(str(copies),max_process_number=4,export_type='db')
    dbs=[]
    for p in sorted(copies.rglob('*.db')):
        assert not p.is_symlink()
        db=sqlite3.connect(p.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        wanted={}
        for name in ['PYTORCH_API','PYTORCH_CALLCHAINS','TASK','COMPUTE_TASK_INFO','STRING_IDS','CANN_API','CONNECTION_IDS']:
            if name in tables:wanted[name]=dict(count=db.execute('SELECT COUNT(*) FROM "'+name+'"').fetchone()[0],columns=[dict(r) for r in db.execute('PRAGMA table_info("'+name+'")')])
        db.close();dbs.append(dict(path=str(p),relative_path=str(p.relative_to(JOB)),bytes=p.stat().st_size,sha256=sha(p),tables=sorted(tables),wanted=wanted))
    changed=[]
    for m in manifest:
        p=RAW/m['relative_path']
        if p.stat().st_size!=m['bytes'] or p.stat().st_mtime_ns!=m['source_mtime_ns'] or sha(p)!=m['sha256']:changed.append(m['relative_path'])
    assert not changed,changed
    (JOB/'database-index.json').write_text(json.dumps(dict(databases=dbs,source_profile_directories=32,original_raw_unchanged=True,no_new_inference_or_kernel_launch=True,whole_modules_closed=0,overall_complete=False),ensure_ascii=False,indent=2)+'\n')
    valid=[d for d in dbs if 'PYTORCH_API'in d['wanted'] and 'PYTORCH_CALLCHAINS'in d['wanted']]
    state('ANALYSE_RETURNED',databases=dbs,database_files=len(dbs),databases_with_API_and_callchains=len(valid),original_raw_unchanged=True,
          note='Return/DBpresence are not full semantic coverage. Next verify same-call markers,prepare roots,source chains,shapes andall32schema/phase coverage.')
except BaseException as exc:
    state('FAILED',error_type=type(exc).__name__,error=str(exc)[:1800])
    raise
