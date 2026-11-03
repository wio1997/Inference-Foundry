"""CPU-only, anonymized analysis of one captured 169 EngineCore sample.

Runs inside the existing stage container against retained outputs; emits no
prompt or generated text. It does not launch a service or device work.
"""
import collections
import hashlib
import json
from pathlib import Path
import re

RUN='framework-20261030T091949Z-3d9bcf87'
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/runtime/logs')/RUN

def data(name):return json.loads((ROOT/name).read_text())
def sha(name):return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
def frame_key(frame):
    return Path(frame.get('file') or '').name+':'+frame.get('name','')
def summarize_profile(profile,frames):
    leaf=collections.Counter();incl=collections.Counter();ancestry=collections.Counter();empty=0
    samples=profile['samples'];weights=profile['weights']
    assert len(samples)==len(weights)
    for stack,weight in zip(samples,weights):
        if not stack:empty+=1;continue
        leaf[frame_key(frames[stack[-1]])]+=weight
        keys={frame_key(frames[i]) for i in stack}
        for key in keys:incl[key]+=weight
        if 'multiproc_executor.py:_wait_for_response' in keys:
            if 'core.py:step_with_batch_queue' in keys:ancestry['response_in_batch_queue']+=weight
            elif 'core.py:step' in keys:ancestry['response_in_single_step']+=weight
            elif 'core.py:execute_dummy_batch' in keys:ancestry['response_in_dummy_batch']+=weight
            else:ancestry['response_other_or_partial']+=weight
    total=sum(weights)
    return {'name':re.sub(r'Thread \d+','Thread [redacted]',profile['name']),
            'samples':len(samples),'empty_stacks':empty,'sample_weight_s':total,
            'response_wait_ancestor_sample_weight_s':dict(ancestry),
            'leaf_top':[{'frame':key,'weight_s':value,'pct_of_thread_sample_weight':100*value/total}
                        for key,value in leaf.most_common(10)] if total else [],
            'selected_inclusive':{key:{'weight_s':incl.get(key,0),
                                      'pct_of_thread_sample_weight':100*incl.get(key,0)/total if total else None}
                                  for key in ('multiproc_executor.py:_wait_for_response',
                                              'scheduler.py:schedule','shm_broadcast.py:dequeue',
                                              'threading.py:wait')}}

def main():
    result=data('result.json');requests=data('measure-results.json');proc=data('enginecore-proc.json')
    start=min(x['start_ns'] for x in requests);end=max(x['end_ns'] for x in requests)
    included=[x for x in proc['samples'] if start<=x['monotonic_ns']<=end]
    assert len(included)>=2
    begin=included[0];finish=included[-1];elapsed=(finish['monotonic_ns']-begin['monotonic_ns'])/1e9
    tick=proc['clock_ticks_per_second'];cpus={}
    for engine in sorted(begin['engines']):
        a=begin['engines'][engine];b=finish['engines'][engine]
        item={}
        for name in ('process','leader_thread'):
            if a[name]['starttime_ticks']!=b[name]['starttime_ticks']:raise RuntimeError('process identity changed')
            user=(b[name]['utime_ticks']-a[name]['utime_ticks'])/tick
            system=(b[name]['stime_ticks']-a[name]['stime_ticks'])/tick
            item[name]={'user_s':user,'system_s':system,'total_s':user+system,
                        'mean_cpu_cores':(user+system)/elapsed}
        item['leader_states']=dict(collections.Counter(x['engines'][engine]['leader_thread']['state'] for x in included))
        item['leader_wchan_top']=collections.Counter(x['engines'][engine].get('leader_wchan','missing') for x in included).most_common(6)
        cpus[engine.rsplit('_',1)[-1]]=item
    output={'run_id':RUN,'status':'sampled_not_cpu_profile',
      'limits':['speedscope sampled weights have no per-sample absolute wall times; cannot precisely crop to requests',
                'idle and waiting stacks included; weights are not CPU seconds or communication duration',
                'nonblocking may produce partial stacks; no native attribution or per-request attribution',
                'proc ticks are time-windowed per process and leader thread, not per Python function'],
      'input_sha256':result['prompt_sha256'],'case_valid':result['valid'],
      'per_engine_counters':result['per_engine'],'total_counters':result['total'],
      'request_window':{'start_monotonic_ns':start,'end_monotonic_ns':end,'duration_s':(end-start)/1e9,
                        'first_visible_latest_ns':max(x['first_nonempty_ns'] for x in requests)},
      'proc_window':{'first_ns':begin['monotonic_ns'],'last_ns':finish['monotonic_ns'],
                     'duration_s':elapsed,'points':len(included),'total_points':len(proc['samples']),
                     'clock_ticks_per_second':tick,'engines':cpus},'pyspy':{},
      'file_hashes':{name:sha(name) for name in ('result.json','measure-results.json','enginecore-proc.json',
                        'pyspy-DP0.speedscope.json','pyspy-DP1.speedscope.json')},
      'final_health':{'health_idle':data('FINAL_HEALTH.json').get('health_idle'),
                      'owner_ok':'owner' in data('FINAL_HEALTH.json')}}
    for dp in ('DP0','DP1'):
        datafile=data('pyspy-'+dp+'.speedscope.json');frames=datafile['shared']['frames']
        profiles=[summarize_profile(p,frames) for p in datafile['profiles']]
        main_threads=[p for p in profiles if 'MainThread' in p['name']]
        if len(main_threads)!=1:raise RuntimeError('missing unique main thread')
        window=result['sampler_windows']['pyspy-'+dp]
        output['pyspy'][dp]={'main_thread':main_threads[0],
            'other_threads':[p for p in profiles if 'MainThread' not in p['name']],
            'sampler_launch_ns':window['launch_monotonic_ns'],'sampler_end_ns':window['ended_monotonic_ns'],
            'sampler_parent_observed_elapsed_s':(window['ended_monotonic_ns']-window['launch_monotonic_ns'])/1e9,
            'exit_code':window['exit_code'],'stderr_empty':not (ROOT/('pyspy-'+dp+'.stderr')).read_text().strip(),
            'stderr_sha256':sha('pyspy-'+dp+'.stderr')}
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
