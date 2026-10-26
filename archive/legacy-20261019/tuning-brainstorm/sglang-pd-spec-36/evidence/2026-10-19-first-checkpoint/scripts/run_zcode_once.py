#!/usr/bin/env python3
"""Bounded one-shot execution with independent outer receipt; raw output stays local."""
import datetime,hashlib,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
name,command=sys.argv[1:]
out=ROOT/'runs'/name
out.mkdir(parents=True,exist_ok=False)
args=[str(ROOT/'zcode-task.sh'),'--json','-p',command]
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (out/'stdout.jsonl').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
 p=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr)
 (out/'started.json').write_text(json.dumps({'pid':p.pid,'command':args,'started_utc':started})+'\n')
 try:code=p.wait(timeout=900)
 except subprocess.TimeoutExpired:
  p.terminate()
  try:code=p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();code=p.wait()
  code=124
receipt={'command':args,'cwd':str(ROOT),'exit_code':code,'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'artifacts':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['stdout.jsonl','stderr.log']}}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
sys.exit(code)
