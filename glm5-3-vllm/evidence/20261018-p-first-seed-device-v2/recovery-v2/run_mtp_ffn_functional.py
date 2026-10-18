from pathlib import Path
import collections
import json
import subprocess
import sys
import time

repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2'
root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-ffn-graph-live1-20261018/node33'
cid='1bb3a19d7f6ed621b5fcf72189bbfa85c9a3ecb46753706ae5bf7cd1743a1ad2'
def collect(script,label,out,args=(),timeout=600):
    dest=r/out
    existing=list(dest.glob('*/metrics.json')) if dest.exists() else []
    if existing:
        assert len(existing)==1
        x=json.loads(existing[0].read_text());assert not x.get('failure');return x
    q=subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/collect_node.py'),'--node','33','--script',str(script),'--label',label,'--output',str(dest),'--control','/tmp/glm53-active-20261018.sock','--timeout',str(timeout)]+['--script-arg='+v for v in args],cwd=repo,capture_output=True,text=True,timeout=timeout+180)
    (r/(out+'-collector.txt')).write_text(q.stdout+q.stderr)
    assert q.returncode==0,q.stdout+q.stderr
    return json.loads(next(dest.glob('*/metrics.json')).read_text())

def run(phase):
    x=collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py','mtp_ffn_graph_'+phase.replace('-','_')+'_v1','mtp-ffn-graph-'+phase,('--execute','--phase',phase,'--probe-kind','mtp_ffn_graph','--run-dir',root,'--expected-D-CID',cid,'--cases',root+'/prefill-padding-cases-v1.json'))
    assert x['completed'] and x['numeric_all_pass'],(phase,x.get('failure'))
    print(json.dumps({'phase':phase,'rows':len(x['rows']),'numeric_all':True}),flush=True)
    return x

if __name__=='__main__':
    start=time.monotonic()
    while not (r/'mtp-ffn-graph-startup-outcome.json').exists():
        assert time.monotonic()-start<1100;time.sleep(10)
    assert json.loads((r/'mtp-ffn-graph-startup-outcome.json').read_text())['both_ready']
    functional=run('functional')
    paths=collect(r/'read_mtp_ffn_paths.py','mtp_ffn_graph_paths_v1','mtp-ffn-graph-paths',timeout=90)
    assert paths['ranks']==list(range(16)) and not paths['marker_present']
    proof=[]
    for row in functional['rows']:
        prefix=row['chunks'][0]['id']
        joined=[x for x in paths['rows'] if any(z==prefix or z.startswith(prefix+'-') for z in x['ids'])]
        assert sorted(set(x['rank'] for x in joined))==list(range(16)),(prefix,len(joined))
        counters={rank:collections.Counter() for rank in range(16)}
        shapes=set()
        for x in joined:
            assert x['requested']==(row['mode']=='adaptive')
            assert x['calls']
            for call in x['calls']:
                counters[x['rank']][call['action']]+=1
                shapes.add(tuple(call['shape']))
                if row['mode']=='adaptive':assert call['eligible'],call
        assert all(counters[i]==counters[0] for i in range(16)),counters
        count=sum(v for k,v in counters[0].items() if 'replay' in k)
        assert (count>0)==(row['mode']=='adaptive'),(row['case_id'],row['mode'],counters)
        proof.append({'case':row['case_id'],'mode':row['mode'],'all16':True,'native_actions_per_rank':dict(counters[0]),'shapes':sorted(shapes)})
    (r/'mtp-ffn-graph-path-summary-v1.json').write_text(json.dumps({'all16':True,'rows':proof,'scope':'Actual native MTP MoE calls/guards/cache actions, not physical task or graph output correctness proof.'},indent=2)+'\n')
    print(json.dumps({'phase':'all16_pure_ffn_path','passed':True}),flush=True)
    quality=run('quality')
    before=collect(r/'read_sfa_prefix_metrics.py','mtp_ffn_prefix_before_v1','mtp-ffn-graph-prefix-before',timeout=60)
    cached=run('cached')
    after=collect(r/'read_sfa_prefix_metrics.py','mtp_ffn_prefix_after_v1','mtp-ffn-graph-prefix-after',timeout=60)
    assert any(v>before.get(k,0) for k,v in after.items() if 'hits' in k)
    (r/'mtp-ffn-graph-cache-summary-v1.json').write_text(json.dumps({'before':before,'after':after,'native_P_prefix_hits_increased':True,'complete16_numeric':True},indent=2)+'\n')
    run('c4-functional')
    collect(r/'read_mtp_ffn_paths.py','mtp_ffn_graph_paths_v2','mtp-ffn-graph-paths-v2',timeout=90)
    (r/'mtp-ffn-graph-functional-outcome.json').write_text(json.dumps({'completed':True,'quality_full64_equal_cases':sum(v['all_same'] for v in quality['sequence_comparisons'].values()),'quality_cases':len(quality['sequence_comparisons']),'all16_pure_ffn_path':True,'native_marker_restored':True,'next':'Physical copy/profile gate before formal service timing.'},indent=2)+'\n')
    print('functional_sequence_completed',flush=True)
