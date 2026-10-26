#!/usr/bin/env python3
"""Task-local CPU dependency environment. No global installation or device usage."""
import datetime,hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runs/L003-R000-cpu-setup';OUT.mkdir(parents=True,exist_ok=False)
python='/Users/wio/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
commands=[[python,'-m','venv',str(ROOT/'.venv-cpu')],[str(ROOT/'.venv-cpu/bin/python'),'-m','pip','install','--index-url','https://pypi.org/simple','torch==2.8.0']]
rows=[]
for i,cmd in enumerate(commands):
 with (OUT/f'{i}.log').open('xb') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 rows.append({'command':cmd,'exit_code':p.returncode,'log_sha256':hashlib.sha256((OUT/f'{i}.log').read_bytes()).hexdigest()})
 if p.returncode:break
(OUT/'receipt.json').write_text(json.dumps({'commands':rows,'exit_code':p.returncode,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n');print(json.dumps({'exit_code':p.returncode}));sys.exit(p.returncode)
