import pathlib,subprocess,json,hashlib,sys
J=pathlib.Path(__file__).parent
job=json.loads((J/'job.json').read_text())
with (J/'cpu.stdout.log').open('w') as out,(J/'cpu.stderr.log').open('w') as err:
 p=subprocess.run(['docker','exec','glm52-single','/usr/local/python3.12.13/bin/python3',str(J/'check.py'),str(J/'original.py'),str(J/'patched.py'),str(J/'checks')],stdout=out,stderr=err,timeout=180)
rows=[json.loads(f.read_text()) for f in sorted((J/'checks').glob('rank*.json'))] if (J/'checks').exists() else []
ok=p.returncode==0 and len(rows)==4 and all(x['actual_collective_cases']==60 for x in rows)
b=(J/'cpu.stdout.log').read_bytes()
result={'schema_version':1,'job_id':job['job_id'],'status':'completed' if ok else 'failed','summary':'Actual 4-rank Gloo CPU gather/lifetime correctness '+('passed' if ok else 'failed')+'; no NPU used','execution':{'inner_exit_code':p.returncode,'acceptance':'passed' if ok else 'failed','processes':[]},'findings':[{'kind':'fact','text':'Actual four-rank CPU collectives, dynamic rows, dtypes, noncontiguous inputs, fresh output lifetime tested','scope':{'ranks':[0,1,2,3],'case_count_per_rank':60,'device':'CPU'},'evidence_ids':['cpu']}],'evidence':[{'id':'cpu','path':str(J/'cpu.stdout.log'),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'locator':'all rank checks and test exits'}],'unknowns':['HCCL/native model correctness and E2E performance not yet verified'],'decision_request':None,'next_check_at':None}
(J/'result.json').write_text(json.dumps(result,indent=2)+'\n');sys.exit(0 if ok else 1)
