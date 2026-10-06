"""Official CPU-only offline parser, preserving pre-existing capture data."""
import json,pathlib,subprocess,sys,hashlib,shlex,time
J=pathlib.Path(__file__).resolve().parent
job=json.loads(pathlib.Path(sys.argv[1]).read_text());R=pathlib.Path(job['run_directory'])
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def raw_snapshot(host):
 code="""import pathlib,json,hashlib
root=pathlib.Path(ROOTPATH);folders=list(root.glob('*_ascend_pt'));assert len(folders)==16
rows=[]
for d in folders:
 for p in d.rglob('*'):
  if p.is_file() and any(x in p.parts for x in ['FRAMEWORK','data']):
   rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
assert rows
print(json.dumps(rows))
""".replace('ROOTPATH',repr(str(R/('profiles_'+host))))
 argv=shlex.join(['/usr/bin/python3','-c',code])
 if host=='167':argv=shlex.join(['ssh','-o','BatchMode=yes','root@172.16.10.167',argv])
 p=subprocess.run(argv,shell=True,capture_output=True,timeout=90);assert p.returncode==0,p.stderr[-2000:]
 return json.loads(p.stdout)
state=json.loads((R/'state.json').read_text());assert state['status']=='completed'
for host in ['166','167']:
 before=raw_snapshot(host);write(J/('raw_before_'+host+'.json'),before)
 code='from torch_npu.profiler.profiler import analyse;analyse('+repr(str(R/('profiles_'+host)))+',max_process_number=4)'
 argv=shlex.join(['docker','exec','glm52-single','/usr/local/python3.12.13/bin/python3','-c',code])
 if host=='167':argv=shlex.join(['ssh','-o','BatchMode=yes','root@172.16.10.167',argv])
 start=time.monotonic();p=subprocess.run(argv,shell=True,capture_output=True,timeout=700)
 (J/('parser_'+host+'.stdout')).write_bytes(p.stdout);(J/('parser_'+host+'.stderr')).write_bytes(p.stderr)
 write(J/('parser_'+host+'.json'),dict(exit_code=p.returncode,wall_s=time.monotonic()-start))
 assert p.returncode==0,p.stderr[-2000:]
 after=raw_snapshot(host);write(J/('raw_after_'+host+'.json'),after)
 assert before==after,'original device/framework capture data changed during parsing'
 # A decorator swallows parser errors; actual traces below determine coverage.
p=subprocess.run(['/usr/bin/python3',str(J/'reduce.py'),sys.argv[1]],capture_output=True,timeout=800)
(J/'reduce.stdout').write_bytes(p.stdout);(J/'reduce.stderr').write_bytes(p.stderr);assert p.returncode==0,p.stderr[-2000:]
summary=json.loads((J/'summary.json').read_text());assert summary['all32_nonempty_device_traces'],'missing actual device traces after offline parse;no model rerun'
