"""Select only dummy-head/index and BOOL24 zero witnesses from cached profiles."""
import collections
import hashlib
import json
import time
from pathlib import Path
O=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint172')
D=O.parent/'joint173';D.mkdir(exist_ok=True)
groups={};profiles=[];started=time.time()
for p in sorted((O/'profiles').glob('*.json')):
    d=json.loads(p.read_text());counts=collections.Counter()
    for nr in d['native_rows']:
        r=d['regimes'][nr[9]];fs=[];cn=None
        if len(nr[10])==1:
            c=d['CPU_rows'][str(nr[10][0])];cn=c['name'];fs=[d['source_frame_strings'][str(x)] for x in d['callchains'].get(str(c['callchain_id']),[])]
        tags=[]
        if any('dummy_compute_logits' in f or 'dummy_drafter_compute_logits' in f for f in fs):tags.append('DUMMY_HEAD_CALLBACK')
        if any('model_runner_v1.py(4241): _dummy_run' in f for f in fs):tags.append('DUMMY_ZERO_INDEX_ALLOCATION')
        iszero=any(w in r['opType'].lower() for w in ['zero','fill','memset'])
        booldtype='BOOL' in r['inputDataTypes']+' '+r['outputDataTypes']
        shape24=any(s.strip('" ').strip()=='24' for s in (r['inputShapes']+';'+r['outputShapes']).split(';'))
        if iszero and booldtype and shape24:tags.append('BOOL24_ZERO_REGIME')
        for tag in tags:
            counts[tag]+=1
            key=json.dumps([tag,cn,fs,r],sort_keys=True)
            if key not in groups:groups[key]={'tag':tag,'CPU_name':cn,'source_frames':fs,'regime':r,'executions':0,'kernel_sum_ns':0,'profile_counts':collections.Counter(),'examples':[]}
            g=groups[key];g['executions']+=1;g['kernel_sum_ns']+=nr[3]-nr[2];g['profile_counts'][d['profile']]+=1
            if len(g['examples'])<2:g['examples'].append({'profile':d['profile'],'TASK_ROW':nr[0],'globalTaskId':nr[1],'start_ns':nr[2],'end_ns':nr[3],'CPU_rows':nr[10]})
    profiles.append({'profile':d['profile'],'input':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selected_counts':dict(counts)})
for g in groups.values():g['profile_counts']=dict(g['profile_counts'])
out={'status':'CACHED_PROFILE_LIFECYCLE_OPERATOR_WITNESSES','groups':list(groups.values()),'profiles':profiles,'elapsed_seconds':time.time()-started,'new_device_work':False,'limits':'Positive selected CPU/native connections only. Empty query result is not a missing-branch independence or deadness proof; capture/noCPUcreator cases retained in the full v172 population.'}
(D/'LIFECYCLE-PROFILE-WITNESSES.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'groups':len(groups),'counts':dict(collections.Counter({tag:sum(g['executions'] for g in groups.values() if g['tag']==tag) for tag in ['DUMMY_HEAD_CALLBACK','DUMMY_ZERO_INDEX_ALLOCATION','BOOL24_ZERO_REGIME']})),'elapsed_seconds':out['elapsed_seconds']}))
