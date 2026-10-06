"""One D reload, native correctness, then stock/events-elided A/B/A/B.

Reuses the accepted native PD client and ownership lifecycle. No profiler,
parallel-layout change, gather patch, scan, large workload or kernel change.
"""
import ast
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
old=ROOT.parent/'GLM-RUN-0251/pd_gather_compare.py'
s=importlib.util.spec_from_file_location('oldclient',old);x=importlib.util.module_from_spec(s);s.loader.exec_module(x)
PLAN=json.loads((ROOT/'functional_plan.json').read_text());x.ROOT=ROOT;x.PLAN=PLAN;m=x.m;m.ROOT=ROOT
fetch=x.fetch
text=old.read_text();tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='request')
text=ast.get_source_segment(text,node)
text=text.replace('1+1等于几？最终答案只写数字。','只输出以下句子一次，不要添加其他内容：P 负责处理输入并生成 KV；D 复用 KV，逐步生成输出。')
text=text.replace("re.fullmatch(r'\\s*2[。.!！]?\\s*', content)","bool(content.strip())")
exec(compile(text,'same_PD_client_new_complete_fixture','exec'),x.__dict__)
request=x.request


def rpc(host,action,*args):
    cmd=shlex.join(['/usr/bin/python3',str(Path(__file__).resolve()),action,host,*map(str,args)])
    argv=['/usr/bin/python3',str(Path(__file__).resolve()),action,host,*map(str,args)] if host=='166' else ['ssh','-o','BatchMode=yes','root@172.16.10.167',cmd]
    p=subprocess.run(argv,capture_output=True,text=True,timeout=300 if action=='micro' else 180)
    assert p.returncode==0,(host,action,p.stderr[-5000:]);return json.loads(p.stdout)


def source_digest():
    code='import pathlib,hashlib,json;root=pathlib.Path(%r);print(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in %r}))'%(PLAN['source_root'],list(PLAN['source_files'])+list(PLAN['unchanged_sources']))
    return json.loads(subprocess.check_output(['docker','exec','glm52-single',m.PYTHON,'-c',code],text=True))


def guard(host,fresh=False,kind='original'):
    root=m.new_owner(host) if fresh else PLAN['resident_roles'][host];assert root and m.live(root)
    members=m.tree(root['pid'],m.catalogue());owners,_=m.device_owners();assert len(owners)==16 and owners.issubset({p['pid'] for p in members})
    m.assert_idle(PLAN['ports'][host]);status,_=fetch('http://127.0.0.1:%d/health'%PLAN['ports'][host]);assert status==200
    namespaces={}
    for pid in owners:
        line=next(l for l in Path('/proc/%d/status'%pid).read_text().splitlines() if l.startswith('NSpid:'));namespaces[str(pid)]=int(line.split()[-1])
    if host=='167':
        actual=source_digest();expected={n:z[kind+'_sha256'] for n,z in PLAN['source_files'].items()};expected.update(PLAN['unchanged_sources']);assert actual==expected
        if not fresh:assert Path(PLAN['previous_mode_file']).read_bytes()==b'\x00'
    return dict(root=root,device_owners=sorted(owners),worker_namespace_pids=namespaces,health=status,idle=True)


def replace_source(kind):
    assert kind in ('shim','original')
    actual=source_digest()
    for n,z in PLAN['source_files'].items():
        assert actual[n] in (z['original_sha256'],z['shim_sha256'])
        assert hashlib.sha256((ROOT/kind/n).read_bytes()).hexdigest()==z[kind+'_sha256']
    assert all(actual[n]==sha for n,sha in PLAN['unchanged_sources'].items())
    code='''import pathlib,hashlib,json
target=pathlib.Path(%r);artifact=pathlib.Path(%r);rows=%r;before=%r
for n,z in rows.items():
 p=target/n;b=(artifact/n).read_bytes();assert hashlib.sha256(p.read_bytes()).hexdigest()==before[n];assert hashlib.sha256(b).hexdigest()==z[%r];p.write_bytes(b)
print(json.dumps({n:hashlib.sha256((target/n).read_bytes()).hexdigest() for n in rows}))
'''%(PLAN['source_root'],str(ROOT/kind),PLAN['source_files'],actual,kind+'_sha256')
    got=json.loads(subprocess.check_output(['docker','exec','glm52-single',m.PYTHON,'-c',code],text=True));assert got=={n:z[kind+'_sha256'] for n,z in PLAN['source_files'].items()}
    m.write('source_'+kind+'.json',dict(before=actual,after=got));return got


def switch(mode):
    guard('167',True,'shim');assert mode in (0,1,2,3)
    fd=os.open(ROOT/'event_mode.bin',os.O_WRONLY)
    try:assert os.pwrite(fd,bytes([mode]),0)==1;os.fsync(fd)
    finally:os.close(fd)
    return dict(mode=mode,idle=True)


def launch():
    assert not m.device_owners()[0]
    env=json.loads((ROOT/PLAN['baseline_env']['167']).read_text())['environment'];env.update(PLAN['environment']['167'])
    env['PYTHONPATH']=':'.join(p for p in env.get('PYTHONPATH','').split(':') if '/glm52-pd/deploy/plugins/' not in p)
    command=['docker','exec','-d']
    for k,v in sorted(env.items()):command+=['-e',k+'='+v]
    command+=['glm52-single',m.PYTHON,str(Path(__file__).resolve()),'native','167'];subprocess.run(command,check=True,timeout=30)
    m.write('launch_167.json',dict(command=command,launch_exit=0,readiness='unknown'));return dict(launch_exit=0)


def micro():
    assert not m.device_owners()[0]
    expected={n:z['shim_sha256'] for n,z in PLAN['source_files'].items()};expected.update(PLAN['unchanged_sources']);assert source_digest()==expected
    env=json.loads((ROOT/PLAN['baseline_env']['167']).read_text())['environment'];env.update(PLAN['environment']['167'])
    cmd=['docker','exec']
    for k,v in sorted(env.items()):cmd+=['-e',k+'='+v]
    cmd+=['glm52-single',m.PYTHON,str(ROOT/'check_moe_event_hccl.py'),str(ROOT/'micro_checks'),str(ROOT/'namespace_owner_167.json')]
    with (ROOT/'micro.stdout.log').open('wb') as out,(ROOT/'micro.stderr.log').open('wb') as err:
        try:p=subprocess.run(cmd,stdout=out,stderr=err,timeout=240);code=p.returncode
        except subprocess.TimeoutExpired:code=None
    if m.device_owners()[0]:m.cleanup('167',PLAN)
    assert not m.device_owners()[0]
    f=ROOT/'micro_checks/result.json';row=json.loads(f.read_text()) if f.exists() else dict(passed=False)
    row.update(inner_exit_code=code,timed_out=code is None);row['passed']=bool(row['passed'] and code==0);m.write('micro_execution.json',row);return row


def witness(mode):
    g=guard('167',True,'shim');rows=[json.loads(p.read_text()) for p in sorted((ROOT/'witnesses').glob('model_mode%d_rank*.json'%mode))]
    assert len(rows)==16 and {z['rank'] for z in rows}==set(range(16))
    assert {z['pid'] for z in rows}==set(g['worker_namespace_pids'].values())
    assert all(z['mode']==mode and not z['configured_overlap'] and z['returned_none']==bool(mode%2) for z in rows)
    return rows


def workflow():
    owner=json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text());assert owner['run_id']==PLAN['run_id'] and owner['status']=='running' and m.live(owner['owner'])
    mutated=False;launched=False;fitted=False;results=[]
    try:
        m.write('guards_before.json',{h:rpc(h,'guard') for h in ('166','167')});rpc('167','retire')
        mutated=True;rpc('167','install');probe=rpc('167','micro');m.write('micro_result.json',probe)
        if not probe['passed']:rpc('167','restore');mutated=False
        (ROOT/'event_mode.bin').write_bytes(b'\x00');rpc('167','resetmode');rpc('167','launch');launched=True
        deadline=time.monotonic()+1800
        while True:
            row=rpc('167','poll');m.write('readiness.json',row)
            if row['failure']:raise RuntimeError('D initialization failed')
            if row['ready']:fitted=True;break
            if time.monotonic()>deadline:raise TimeoutError('D readiness')
            time.sleep(10)
        if not probe['passed']:
            result=request('stock_recovery_warm');m.write('final_status.json',dict(status='stock_recovered_candidate_rejected',micro=probe,result=result));return
        for mode,name in enumerate(('A1','B1','A2','B2')):
            rpc('166','guard');rpc('167','switch',mode);request(name+'_warm');m.write(name+'_witness.json',rpc('167','witness',mode))
            for i in range(2):results.append(request(name+'_measure%d'%i))
            results.append(request(name+'_complete',True));m.write('comparison_results.json',results)
        m.write('final_status.json',dict(status='completed',results=results,micro=probe,Current=None,formal_SLA=False,full_API_contract=False))
    except BaseException as error:
        m.write('failure.json',dict(error=repr(error),launched=launched,fitted=fitted,results=results));raise
    finally:
        restored={}
        if mutated:
            if fitted:
                try:restored['mode']=rpc('167','switch',0)
                except Exception as e:restored['mode_error']=repr(e)
            elif launched:
                try:restored['cleanup']=rpc('167','cleanup')
                except Exception as e:restored['cleanup_error']=repr(e)
            try:restored['source']=rpc('167','restore')
            except Exception as e:restored['source_error']=repr(e)
        m.write('restoration.json',restored);assert not any(k.endswith('_error') for k in restored),restored
        if fitted:m.write('guards_after.json',{h:rpc(h,'guardfresh' if h=='167' else 'guard') for h in ('166','167')})


def action(name,host,args):
    if name=='guard':return guard(host)
    if name=='guardfresh':return guard(host,True,'original')
    if name=='retire':
        g=guard(host);members=m.tree(g['root']['pid'],m.catalogue());m.stop_tree(members,g['root']);assert not m.device_owners()[0];return dict(owned_D_retired=True)
    if name=='install':return replace_source('shim')
    if name=='restore':return replace_source('original')
    if name=='micro':return micro()
    if name=='resetmode':(ROOT/'event_mode.bin').write_bytes(b'\x00');return dict(mode=0)
    if name=='launch':return launch()
    if name=='native':return m.native(host,PLAN)
    if name=='poll':return m.poll(host,PLAN)
    if name=='cleanup':return m.cleanup(host,PLAN)
    if name=='switch':return switch(int(args[0]))
    if name=='witness':return witness(int(args[0]))
    raise ValueError(name)


if __name__=='__main__':
    if sys.argv[1]=='workflow':workflow()
    else:print(json.dumps(action(sys.argv[1],sys.argv[2],sys.argv[3:])))
