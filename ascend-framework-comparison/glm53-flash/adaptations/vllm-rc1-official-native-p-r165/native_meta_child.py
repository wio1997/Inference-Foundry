import pathlib,json,sys,os,importlib.metadata,importlib.util,hashlib
R=pathlib.Path('/run-evidence');assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0';os.environ.pop('ASCEND_HOME_PATH',None);assert not list(pathlib.Path('/dev').glob('davinci*'))
# Execute the unchanged installed native pybind extension directly. Its own
# auditwheel DT_NEEDED/RPATH resolves dependencies; no package stub or CDLL hook.
def binding(distribution,package,module):
 dist=importlib.metadata.distribution(distribution);candidates=[dist.locate_file(f) for f in dist.files if str(f).startswith(package+'/'+module+'.') and str(f).endswith('.so')];assert len(candidates)==1
 path=pathlib.Path(candidates[0]);spec=importlib.util.spec_from_file_location(module,path);native=importlib.util.module_from_spec(spec);sys.modules[module]=native;spec.loader.exec_module(native)
 return native,path,dist
# InstallednativepackageimportsMF.bm beforeMMC; register exactlythat native
# pybind type dependency without MF Python offload/Torch imports or stubs.
hybrid,hybrid_path,mfdist=binding('memfabric-hybrid','memfabric_hybrid','_pymf_hybrid');assert hasattr(hybrid,'bm') and hasattr(hybrid.bm,'BmCopyType')
native,path,dist=binding('memcache-hybrid','memcache_hybrid','_pymmc');MetaService=native.MetaService
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules and 'memfabric_hybrid' not in sys.modules and 'memcache_hybrid' not in sys.modules
native_paths=['memcache_hybrid','memcache_hybrid.libs','memfabric_hybrid','memfabric_hybrid.libs']
def mapped():return sorted({l.split()[-1] for l in pathlib.Path('/proc/self/maps').read_text().splitlines() if any(p in l for p in native_paths) or '/usr/local/Ascend/' in l or '/torch/' in l})
def threads():return sorted([{'tid':int(t.name),'comm':(t/'comm').read_text().strip()} for t in pathlib.Path('/proc/self/task').iterdir()],key=lambda v:v['tid'])
out={'project_id':'glm53-flash','CPU_only_metadata':True,'direct_unchanged_native_pybind':True,'native_BM_types_registered_first':True,'MF_native_extension_path':str(hybrid_path),'MF_native_extension_SHA256':hashlib.sha256(hybrid_path.read_bytes()).hexdigest(),'native_extension_path':str(path),'native_extension_SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'SDK_versions':[importlib.metadata.version('memfabric-hybrid'),dist.version],'torch_imported':False,'torch_npu_imported':False,'mapped_before_Meta':mapped(),'threads_before_Meta':threads()};assert out['SDK_versions']==['1.2.0','1.2.0'];assert not any('/usr/local/Ascend/' in p or '/torch/' in p for p in out['mapped_before_Meta']);(R/'meta-child-status.json').write_text(json.dumps(out,indent=2));code=int(MetaService.main());out['native_main_rc']=code;out['mapped_after_Meta']=mapped();out['threads_after_Meta']=threads();assert 'torch' not in sys.modules and not any('/usr/local/Ascend/' in p or '/torch/' in p for p in out['mapped_after_Meta']);(R/'meta-child-result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True);raise SystemExit(code)
