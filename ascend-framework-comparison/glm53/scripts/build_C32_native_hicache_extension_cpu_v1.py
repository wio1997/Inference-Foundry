"""Build the SDK-shipped optional extension into a task directory, CPU only."""
import hashlib,json,os,re,subprocess,tempfile,time
from pathlib import Path

def main():
    root=Path('/run/glm53');assert root.is_dir()
    assert not Path('/dev/davinci_manager').exists() and not list(Path('/dev').glob('davinci[0-9]*'))
    for name in ['extension','tmp','build-logs']:(root/name).mkdir(exist_ok=False)
    tempfile.tempdir=str(root/'tmp')
    import memfabric_hybrid
    from memfabric_hybrid import _acc_offload_build as build
    assert hashlib.sha256(Path(build.__file__).read_bytes()).hexdigest()=='017c878a4409fbde8c0591a45da674745eceeb5489f29b267f8c33d51135000b'
    pkg=Path(build.__file__).parent
    paths=[pkg/'__init__.py',pkg/'_acc_offload_build.py',pkg/'lib/libmf_acc_offload.so',pkg/'lib/libmf_hybm_accoffload.so']
    def identity(paths):return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
    before=identity(paths);source=identity(sorted(p for p in (pkg/'_kernel/acc_offload').rglob('*') if p.is_file()))
    assert len(source)==16 and before[str(pkg/'lib/libmf_hybm_accoffload.so')]['bytes']==0
    stages=[]
    def command(args,stage,deadline):
        assert re.fullmatch('[a-z_]+',stage);actual=list(args)
        if actual[:2]==['cmake','--build']:
            at=actual.index('-j');actual[at+1]='4'
        start=time.monotonic();p=subprocess.run(actual,capture_output=True,timeout=max(1,deadline-start))
        prefix=root/'build-logs'/(str(len(stages))+'-'+stage)
        Path(str(prefix)+'.stdout').write_bytes(p.stdout);Path(str(prefix)+'.stderr').write_bytes(p.stderr)
        record={'stage':stage,'original_argv':args,'actual_argv':actual,'rc':p.returncode,'elapsed_seconds':time.monotonic()-start,'stdout_bytes':len(p.stdout),'stdout_sha256':hashlib.sha256(p.stdout).hexdigest(),'stderr_bytes':len(p.stderr),'stderr_sha256':hashlib.sha256(p.stderr).hexdigest()};stages.append(record)
        (root/'build-stages.json').write_text(json.dumps(stages,indent=2)+'\n')
        assert len(p.stdout)+len(p.stderr)<=32*1024**2
        if p.returncode:raise build.ExtendBuildError(stage,p.returncode,'Complete stdout/stderr retained under '+str(prefix))
    build._run_cmd=command
    result={'project_id':'glm53','model_requests':0,'SDK_init_calls':0,'NPU_devices_exposed':False,'kernel_source_files':source,'before_package_files':before,'native_A3_config':build._SOC_CONFIG['A3'],'max_build_jobs':4,'install_dir':str(root/'extension'),'pass':False}
    try:
        target=build.ensure_acc_offload_ext('A3',install_dir=str(root/'extension'),force=False)
        assert target==str(root/'extension')
        lib=root/'extension/libmf_hybm_accoffload.so';raw=lib.read_bytes();assert raw[:4]==b'\x7fELF' and raw[18:20]==(183).to_bytes(2,'little')
        symbols=subprocess.run(['nm','-D','--defined-only',str(lib)],capture_output=True,text=True,timeout=30);assert symbols.returncode==0
        expected=['AccOffloadSparseCopy','AccOffloadGroupPackCopy','AccOffloadKvExchange','AccOffloadEntryGather'];assert all(name in symbols.stdout.split() for name in expected)
        links=subprocess.run(['ldd',str(lib)],capture_output=True,text=True,timeout=30);assert links.returncode==0 and 'not found' not in links.stdout
        result.update(pass_=True,extension_bytes=len(raw),extension_sha256=hashlib.sha256(raw).hexdigest(),native_symbols=expected,ldd=links.stdout,torch_environment=build._torch_env(),ascend_home=build._get_ascend_home())
        result['pass']=result.pop('pass_')
    except Exception as error:
        import traceback
        result.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        result['after_package_files']=identity(paths);result['SDK_package_files_unchanged']=result['after_package_files']==before;assert result['SDK_package_files_unchanged'];result['stages']=stages
        (root/'build-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GLM53_PRIVATE_EXTENSION_BUILD='+json.dumps(result));return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
