"""按实测合同串行运行四档 AISBench。默认打印计划；执行前自行协调服务和卡组独占。"""
from pathlib import Path
import argparse,datetime,fcntl,json,re,subprocess,time,urllib.request,uuid
ROOT=Path(__file__).resolve().parents[1]
CLIENT='ghcr.io/aisbench/aisbench_benchmark:v3.1-20261013-master-openeuler22.03-py310-aarch64'
CID='sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--server-container',default='sglang-dsv41-910c');p.add_argument('--execute',action='store_true')
    p.add_argument('--benchmark-lock',type=Path)
    p.add_argument('--generation',choices=['g30','g37','g42','g43'],default='g42')
    a=p.parse_args();run=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    plan=[{'input':n,'output':1024,'concurrency':16,'warmup':16,'formal':32} for n in [32768,65536,131072,262144]]
    if not a.execute:print(json.dumps({'plan':plan,'client':CLIENT,'server':a.server_container},indent=2));return
    assert a.benchmark_lock and a.benchmark_lock.parent.is_dir(),'Use the shared benchmark coordination lock'
    lock=a.benchmark_lock.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    subprocess.run(['python3',str(ROOT/'scripts/unpack_data.py')],cwd=ROOT,check=True)
    def inspect(name):return json.loads(subprocess.check_output(['docker','inspect',name]))[0]
    server=inspect(a.server_container);assert server['State']['Running']
    image=json.loads(subprocess.check_output(['docker','image','inspect',CLIENT]))[0]
    assert image['Id']==CID and image['Architecture']=='arm64','Use the exact archived AISBench image'
    def api(path,method='GET'):
        with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18167'+path,method=method),timeout=60) as r:
            assert r.status==200;return r.read()
    api('/health')
    out=ROOT/'results'/run;out.mkdir(parents=True,exist_ok=False)
    (out/'server_info_before.json').write_bytes(api('/server_info'))
    name='sglang-archive-aisbench-'+uuid.uuid4().hex[:8]
    label='archive.run='+run
    argv=['docker','run','-d','--name',name,'--label',label,'--network','host','--cpus','8','--memory','32g',
          '-v',str(ROOT)+':/task','-v',str(a.model_path.resolve())+':/task/tokenizer:ro',
          '-e','TMPDIR=/task/.work/client-tmp','--entrypoint','sleep',CID,'infinity']
    (ROOT/'.work/client-tmp').mkdir(parents=True,exist_ok=True)
    subprocess.run(argv,check=True,cwd=ROOT)
    def ownership(tag):
        text=subprocess.check_output(['npu-smi','info'],text=True);own=set();foreign=[]
        for line in text.splitlines():
            fields=[x.strip() for x in line.split('|')[1:-1]]
            if len(fields)<4 or not re.fullmatch(r'\d+\s+\d+',fields[0]) or not fields[1].isdigit():continue
            try:cg=(Path('/proc')/fields[1]/'cgroup').read_text()
            except FileNotFoundError:continue
            if server['Id'] in cg:own.add(int(fields[1]))
            else:foreign.append(int(fields[1]))
        sample={'tag':tag,'own_npu_process_count':len(own),'foreign':foreign}
        with (out/'ownership.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
        assert not foreign and len(own)>=16,'NPU overlap or missing model ranks; reject this measurement'
    start=time.monotonic()
    try:
        for n in [32768,65536,131072,262144]:
            for phase,count in [('warmup',16),('formal',32)]:
                assert time.monotonic()-start<21600
                ownership(f'{n}_{phase}_before')
                flushed=api('/flush_cache?timeout=60','POST');assert b'Cache flushed' in flushed
                (out/f'{n}_{phase}_flush.txt').write_bytes(flushed)
                config=f'/task/configs/g37/aisbench_sglang_{phase}_{n}.py' if a.generation!='g30' else f'/task/configs/aisbench_{phase}_{n}.py'
                cmd=['docker','exec',name,'timeout','--signal=TERM','--kill-after=30','7200','ais_bench',
                     config,'--mode','perf','--num-warmups','0',
                     '--num-prompts',str(count),'--max-num-workers','1','--work-dir',f'/task/results/{run}/{phase}_{n}']
                (out/f'{n}_{phase}_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
                with (out/f'{n}_{phase}_cli.log').open('w') as f:
                    proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);last=time.monotonic()
                    while proc.poll() is None:
                        if time.monotonic()-last>=30:ownership(f'{n}_{phase}_during');last=time.monotonic()
                        time.sleep(2)
                    assert proc.returncode==0,'Inspect raw output before retry'
                ownership(f'{n}_{phase}_after')
        (out/'server_info_after.json').write_bytes(api('/server_info'))
        (out/'runner_terminal.json').write_text(json.dumps({'phase':'all_CLI_terminal_pending_raw_result_acceptance','validated':False}))
    finally:
        d=inspect(name);assert d['Config']['Labels']['archive.run']==run
        subprocess.run(['docker','stop','--time','20',name],check=True,timeout=60)
if __name__=='__main__':main()
