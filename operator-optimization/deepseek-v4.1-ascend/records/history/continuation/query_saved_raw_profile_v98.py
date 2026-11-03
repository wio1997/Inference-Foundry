"""Read only an existing pinned raw DB over an already verified dedicated SSH master."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE,ORIGINAL

CONTROL=N/'ssh166-v98-xinyufeng.sock'
ACCOUNT='xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn'

def remote(code,timeout=30):
    assert CONTROL.exists(),'No verified master;do not fall back to another asset or unverified fresh SSH.'
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(CONTROL),'-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224',ACCOUNT,'python3 -I -S -c '+shlex.quote(code)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    result=None
    if p.returncode==0:
        for line in p.stdout.splitlines():
            try:
                x=json.loads(line)
                if isinstance(x,dict)and x.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=x
            except ValueError:pass
    return dict(observed_UTC=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,stderr=p.stderr[-2000:],
      no_model_Torch_NPU_service_or_new_capture=True,operation='Readonlyexistingcapturefilesonly')

def main():
    assert not(O/'REVIEW-CHECKPOINT-v98.json').exists()
    identity=read(N/'original166-readonly-recheck-v98-fixed.json');assert identity['status']=='READONLY_REMOTE_IDENTITY_VERIFIED'
    ref=read(CACHE/'analysis/profile-capabilities-index-v12-v2.json')['databases'][0]
    p=Path(ref['path']);assert p.is_relative_to(ORIGINAL)if hasattr(p,'is_relative_to')else str(p).startswith(str(ORIGINAL)+'/')
    code='''import os,socket,json,sqlite3,hashlib,re
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
p=Path(PATH);b=p.stat();assert b.st_size<128*1024*1024
digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==EXPECTED
con=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);con.row_factory=sqlite3.Row
tables=[r[0]for r in con.execute("select name from sqlite_master where type='table' order by name")]
def q(x):return '"'+x.replace('"','""')+'"'
schema={t:[dict(x)for x in con.execute('pragma table_info('+q(t)+')')]for t in tables}
relevant=[t for t in tables if re.search('MEMORY|MEMCPY|ARG|ATTR|TENSOR|WORKSPACE|TILING|KERNEL',t,re.I)]
details={t:dict(columns=[x['name']for x in schema[t]],rows=con.execute('select count(*)from '+q(t)).fetchone()[0])for t in relevant}
assert 'STRING_IDS'in tables
strings=[dict(x)for x in con.execute('select * from STRING_IDS where id=6')]
ops={}
for op in ['DispatchFFNCombine','MemSet','ZerosLike']:
 rows=con.execute('select c.rowid as definition_rowid,c.attrInfo,c.inputShapes,c.outputShapes,a.value as attrs,i.value as input_shapes,o.value as output_shapes from COMPUTE_TASK_INFO c join STRING_IDS n on c.opType=n.id left join STRING_IDS a on c.attrInfo=a.id left join STRING_IDS i on c.inputShapes=i.id left join STRING_IDS o on c.outputShapes=o.id where n.value=? order by c.rowid limit 12',(op,)).fetchall()
 ops[op]=[dict(x)for x in rows]
con.close();a=p.stat();assert (b.st_size,b.st_mtime_ns)==(a.st_size,a.st_mtime_ns)
print(json.dumps(dict(hostname=socket.gethostname(),path=str(p),sha256=digest,bytes=b.st_size,tables=tables,relevant=details,attribute_ID6=strings,operator_definition_samples=ops,stat_unchanged=True,SQLite_mode='ro&immutable=1',no_model_Torch_NPU_service_or_new_capture=True)))
'''.replace('PATH',repr(str(p))).replace('EXPECTED',repr(ref['sha256']))
    result=remote(code)
    write(O/'raw-profile-schema-and-args-v98.json',result)
    assert result['returncode']==0 and result['result'],result
    x=result['result'];print(json.dumps(dict(tables=len(x['tables']),relevant=x['relevant'],attribute_ID6=x['attribute_ID6'],
      op_sample_counts={k:len(v)for k,v in x['operator_definition_samples'].items()},sha256=x['sha256'],no_new_capture=True)))

if __name__=='__main__':main()
