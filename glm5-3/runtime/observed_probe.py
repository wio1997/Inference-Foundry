"""Diagnostic observer only: monotonic P/D metrics during a real streaming probe."""
import argparse,json,subprocess,sys,threading,time,urllib.request
from pathlib import Path
from phase_runner import utc

def main(path):
    path=Path(path).resolve();out=path.parent;stop=threading.Event()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    targets={'P':'http://172.16.10.166:9081/metrics','D':'http://172.16.10.167:9900/metrics','proxy':'http://127.0.0.1:8000/healthcheck'}
    def observe():
        with (out/'telemetry.jsonl').open('x') as f:
            while True:
                t=time.monotonic()
                for role,url in targets.items():
                    row={'role':role,'at':utc(),'start_monotonic':time.monotonic()}
                    try:
                        with opener.open(url,timeout=3) as response: row['text']=response.read().decode()
                    except Exception as e:row['error']=type(e).__name__+': '+str(e)
                    row['end_monotonic']=time.monotonic();f.write(json.dumps(row)+'\n');f.flush()
                if stop.wait(max(0,2-(time.monotonic()-t))):break
    worker=threading.Thread(target=observe);worker.start()
    try:code=subprocess.call([sys.executable,str(Path(__file__).with_name('stream_probe.py')),str(path)])
    finally:stop.set();worker.join()
    return code
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');a=p.parse_args();sys.exit(main(a.spec))
