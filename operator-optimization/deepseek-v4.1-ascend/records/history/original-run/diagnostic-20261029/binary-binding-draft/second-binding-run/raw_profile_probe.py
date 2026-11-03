"""Read-only raw-profile visibility/stability clues; does not parse profile data."""
from pathlib import Path
import json,time,sys

def snapshot(root):
    root=Path(root);issues=[];progress={}
    try:
        progress=json.loads((root/'profile-client-results/progress.json').read_text())
        if not isinstance(progress,dict):progress={};issues.append('progress_not_object')
    except (OSError,ValueError) as e:issues.append('progress_unreadable:'+type(e).__name__)
    if progress.get('state')!='REQUESTS_COMPLETE_EXPORT_PENDING':issues.append('client_assertions_not_completed')
    if progress.get('profile_stop_http')!=200:issues.append('stop_profile_not_200')
    deltas=progress.get('measure1_counter_deltas')
    if not isinstance(deltas,dict):deltas={};issues.append('measure1_counters_missing')
    expected={'request_success_total':8,'prompt_tokens_total':262144,'generation_tokens_total':8192}
    if any(deltas.get(k)!=v for k,v in expected.items()):issues.append('measure1_counters_incomplete')
    base=root/'diagnostic-traces';roots=sorted(x for x in base.glob('*_ascend_pt') if x.is_dir())
    if len(roots)!=16:issues.append('worker_root_count_not_16')
    counts={}
    for r in roots:
        files=[]
        try:
            framework=r/'FRAMEWORK/torch.op_range'
            device=[x for x in r.glob('PROF_*/device_*/data/*') if x.is_file() and not x.name.endswith('.done')]
            if not framework.is_file():issues.append(r.name+':framework_missing')
            if not device:issues.append(r.name+':device_data_missing')
            files=([framework] if framework.is_file() else [])+device
            size=0
            for f in files:
                n=f.stat().st_size;size+=n
                if n<=0:issues.append(r.name+':empty_data')
                with f.open('rb') as stream:
                    if not stream.read(1):issues.append(r.name+':unreadable_data')
            counts[r.name]={'files':len(files),'bytes':size}
        except OSError as e:issues.append(r.name+':io_'+type(e).__name__)
    return {'profile_stop_http':progress.get('profile_stop_http'),'client_state':progress.get('state'),'measure1_counters':{k:deltas.get(k) for k in expected},'roots':counts,'issues':issues}

def evaluate(samples):
    ready=len(samples)==3 and all(not x['issues'] for x in samples)
    stable=len(samples)==3 and all(x['roots']==samples[0]['roots'] for x in samples[1:])
    return {'status':'RAW_VISIBILITY_STABLE_CLUE' if ready and stable else 'RAW_PROFILE_INCOMPLETE_OR_UNSTABLE','samples':samples,'count_size_stable':stable,'evidence_limit':'readable data and stable file counts/sizes do not prove complete profiler flush'}

def main():
    root=Path(sys.argv[1]);samples=[]
    for i in range(3):
        samples.append(snapshot(root))
        if i<2:time.sleep(2)
    out=evaluate(samples);p=root/'metadata/raw-profile-readiness.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
