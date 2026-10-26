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
    expected={'launch/acc_offload_operators_launch.cpp': {'bytes': 3980, 'sha256': '3b556db6f2c7ca457707327bf0dd72e27cd90372deb83e39db8f7804e74c11bb'}, 'operators/CMakeLists.txt': {'bytes': 1482, 'sha256': 'b016f000a8b6f98d278d5144b99d510e7cedf8141fa09c13a8ff8022a19b9682'}, 'operators/acc_offload_entry_gather.cpp': {'bytes': 1669, 'sha256': '73d16c091f0a68ced6b7f7bd84cfaeda0adedf3f5c4f63eee6c1654d25cdc265'}, 'operators/acc_offload_entry_gather.h': {'bytes': 7017, 'sha256': '1edcc9bef0a3d6a0c770392eb214a20c8a43e81b52c6555ffe0621b6e3a88c24'}, 'operators/acc_offload_group_pack_copy.cpp': {'bytes': 2023, 'sha256': '1f1c79e0b7cf3bef2403ece61efe99963b0d9af5b953419ce93ed0b7792f6ebc'}, 'operators/acc_offload_group_pack_copy.h': {'bytes': 15008, 'sha256': 'b11e0145556fa5ce3c360271d3760131d7f7fb1820c29568334ab6704852a420'}, 'operators/acc_offload_kv_exchange.cpp': {'bytes': 1057, 'sha256': '6e2b7a117fb17dba339a356d350b300d7ffc5a1d6b5b554d3a68dc1d230f54e3'}, 'operators/acc_offload_kv_exchange.h': {'bytes': 18562, 'sha256': '049f5115fde8f80e752a9359c6dedc58aa91aba86855291d3363b7c12bd4ca10'}, 'operators/acc_offload_operators.h': {'bytes': 1916, 'sha256': '3587b1e6a2acd1f0804c591cfa6a769971e81ca37896718769507cccab5f0234'}, 'operators/acc_offload_sparse_copy.cpp': {'bytes': 1433, 'sha256': '7ada7bc62a05d0506de916171e76f0186f458eb7f45621b0388f7b3703fa3435'}, 'operators/acc_offload_sparse_copy.h': {'bytes': 9831, 'sha256': '30469dd0d4eebe910f6e02a3976d36358c7224017ed33b8884e6585ad59f461a'}, 'operators/config_ascendc.cmake': {'bytes': 975, 'sha256': '99f5eb6eb961a6dcd045a7a6ae107d3d3adc9633ce6da1e3f787b69ce1a3fa32'}}
    relative={str(Path(k).relative_to(pkg/'_kernel/acc_offload')):v for k,v in source.items()}
    (root/'source-preflight.json').write_text(json.dumps({'actual':relative,'expected':expected,'package_files':before},indent=2)+'\n')
    assert relative==expected and before[str(pkg/'lib/libmf_hybm_accoffload.so')]['bytes']==0
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
