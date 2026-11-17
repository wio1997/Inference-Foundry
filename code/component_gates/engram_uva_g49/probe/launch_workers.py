from pathlib import Path
import subprocess,os,json,time
R=Path(__file__).resolve().parents[1];os.chdir(R);out=R/'evidence';out.mkdir(exist_ok=True);jobs=[];logs=[]
for rank in range(8):
 f=(out/f'worker_{rank}.log').open('x');logs.append(f);jobs.append(subprocess.Popen(['python3','probe/worker.py','--rank',str(rank)],cwd=R,stdout=f,stderr=subprocess.STDOUT))
rc=[p.wait() for p in jobs]
for f in logs:f.close()
reports=[json.loads((out/f'rank_{i}.json').read_text()) if (out/f'rank_{i}.json').exists() else {'rank':i,'passed':False,'failure':'missingreport'} for i in range(8)]
result={'worker_exit_codes':rc,'all8_passed':all(c==0 for c in rc) and all(r['passed'] for r in reports),'reports':reports,'scope':'EightindependentphysicalNPUcomponentchecks;notTP/EP/model/PD/performanceacceptance'};(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(not result['all8_passed'])
