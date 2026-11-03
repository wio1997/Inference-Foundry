"""Reuse complete derived neighborhoods;finalize statistics after output-size assertion."""
import base64,json,sys,zlib
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102');from control import invoke
H=Path(__file__).resolve().parent;s=(H/'recover_public_evidence.py').read_text()
start=s.index("related_count=0;");end=s.index("\nwith trace.open('rb')",start)
replacement='''related_count=0;related_ops=collections.Counter();storage_events=collections.defaultdict(list);bool_path=OUT/'ep0-BOOL24-storage-native-neighborhood.jsonl'
assert bool_path.exists() and bool_path.stat().st_size<160*1024**2
with bool_path.open() as f:
    for line in f:
        n=json.loads(line);related_count+=1;related_ops[n['name']]+=1
        hit={t[1] for side in ['inputs','outputs'] for t in tensor_values(n[side].get('values',[])) if t[1] in bool_storages}
        for st in hit:
            if len(storage_events[st])<12:storage_events[st].append({'node':n['id'],'name':n['name'],'ctrl_deps':n['ctrl_deps']})
'''
s=s[:start]+replacement+s[end:];compile(s,'cpu-recovery-finalize.py','exec');(H/'recover_public_evidence_finalize.py').write_text(s)
code='PAYLOAD='+repr(base64.b64encode(zlib.compress(s.encode())).decode())+'\n'+r'''from pathlib import Path
import base64,json,subprocess,zlib
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint176-derived');h=json.loads((p/'SUBMITTED.json').read_text());q=Path('/proc/'+str(h['pid'])+'/stat');v=q.read_text().rsplit(')',1)[1].split() if q.exists() else [];assert not (v and v[19]==h['start_ticks'] and v[0]!='Z');assert not (p/'SUBMITTED-FINALIZE.json').exists()
f=p/'cpu-recovery-finalize.py';f.write_bytes(zlib.decompress(base64.b64decode(PAYLOAD)));compile(f.read_text(),str(f),'exec');log=(p/'cpu-finalize.log').open('ab');c=subprocess.Popen(['timeout','180','python3','-I','-S',str(f)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
y={'pid':c.pid,'start_ticks':Path('/proc/'+str(c.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],'root':str(p),'CPU_only':True,'existing_native_neighborhood_reused':True}
(p/'SUBMITTED-FINALIZE.json').write_text(json.dumps(y,indent=2)+'\n');print(json.dumps(y))'''
x=invoke('target',code,timeout=30);(H/'cpu-finalize-submission.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps(x))
