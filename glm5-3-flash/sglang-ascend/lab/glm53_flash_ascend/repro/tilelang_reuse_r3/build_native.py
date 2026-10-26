"""Build pinned TileLang native libraries in an isolated directory, without install."""
import argparse,hashlib,json,os,pathlib,signal,subprocess,sys,time,traceback

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--build',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 own=pathlib.Path(__file__).resolve();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=own.parent,text=True).strip()
 inputs=[{'path':str(f.relative_to(a.source)),'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(a.source.rglob('*')) if f.is_file() and not f.is_symlink()]
 identity={'scope':'CPU source/native build only; no installation/NPU/model','builder_source_commit':head,'builder_sha256':digest(own),'source_files':inputs,'source_tree_sha256':hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'patches':'exact two pinned upstream TVM patches applied in source preparation; original archives separately bound','parallel_jobs':4,'timeout_per_build_seconds':3600}
 (a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
 configure=['cmake','-S',str(a.source),'-B',str(a.build),'-G','Ninja','-DUSE_ASCEND=ON','-DUSE_CUDA=OFF','-DUSE_ROCM=OFF','-DUSE_LLVM=OFF','-DUSE_GTEST=OFF','-DUSE_LIBBACKTRACE=OFF','-DUSE_CCACHE=OFF','-DTVM_SOURCE_DIR='+str(a.source/'3rdparty/tvm'),'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY='+str(a.build/'lib'),'-DCMAKE_INSTALL_PREFIX='+str(a.build/'unused-install-prefix')]
 build=['cmake','--build',str(a.build),'--target','tvm','tvm_runtime','tilelang','tilelang_module','--parallel','4']
 steps=[]
 for name,cmd,limit in [('configure',configure,180),('build',build,3600)]:
  entry={'name':name,'command':cmd,'start_epoch':time.time(),'timeout_seconds':limit};log=a.output/(name+'.log')
  with log.open('x') as stream:
   proc=subprocess.Popen(cmd,cwd=a.source,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CCACHE_DISABLE':'1'})
   entry['pid']=proc.pid;entry['proc_start_stat']=pathlib.Path('/proc/'+str(proc.pid)+'/stat').read_text();(a.output/(name+'-intent.json')).write_text(json.dumps(entry,indent=2)+'\n')
   try:entry['returncode']=proc.wait(timeout=limit)
   except subprocess.TimeoutExpired:
    entry['timeout']=True;os.killpg(proc.pid,signal.SIGTERM)
    try:entry['returncode']=proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);entry['returncode']=proc.wait(timeout=10)
  entry.update(end_epoch=time.time(),log_sha256=digest(log),log_bytes=log.stat().st_size);steps.append(entry);(a.output/'steps.json').write_text(json.dumps(steps,indent=2)+'\n');print(json.dumps({'step':name,'returncode':entry['returncode'],'seconds':entry['end_epoch']-entry['start_epoch']}),flush=True)
  if entry['returncode']:
   print('\n'.join(log.read_text(errors='replace').splitlines()[-45:]),flush=True);return 1
 native=[{'path':str(f.relative_to(a.build)),'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(a.build.rglob('*.so')) if f.is_file()]
 (a.output/'result.json').write_text(json.dumps({'builder_source_commit':head,'steps':steps,'native_libraries':native,'installed':False,'npu_or_model_started':False},indent=2)+'\n');print(json.dumps({'native_libraries':native,'installed':False}),flush=True);return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception:traceback.print_exc();raise
