from pathlib import Path
import collections
import json
import subprocess
import sys
import time

repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2'
root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-moe-graph-live2-20261018/node33'
cid='084035e84399bd88c3788c80f184590f309fd6730f93cc308f460487c75af213'
def collect(script,label,out,args=(),timeout=600,node=33):
    dest=r/out
    existing=list(dest.glob('*/metrics.json')) if dest.exists() else []
    if existing:
        assert len(existing)==1
        x=json.loads(existing[0].read_text());assert not x.get('failure');return x
    q=subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/collect_node.py'),'--node',str(node),'--script',str(script),'--label',label,'--output',str(dest),'--control','/tmp/glm53-active-20261018.sock','--timeout',str(timeout)]+['--script-arg='+v for v in args],cwd=repo,capture_output=True,text=True,timeout=timeout+180)
    (r/(out+'-collector.txt')).write_text(q.stdout+q.stderr)
    assert q.returncode==0,q.stdout+q.stderr
    return json.loads(next(dest.glob('*/metrics.json')).read_text())

def run(phase):
    x=collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py','mtp_moe_graph_live2_'+phase.replace('-','_')+'_v1','mtp-moe-graph-live2-'+phase,('--execute','--phase',phase,'--probe-kind','mtp_moe_graph','--run-dir',root,'--expected-D-CID',cid,'--cases',root+'/prefill-padding-cases-v1.json'))
    assert x['completed'] and x['numeric_all_pass'],(phase,x.get('failure'))
    print(json.dumps({'phase':phase,'rows':len(x['rows']),'numeric_all':True}),flush=True)
    return x

if __name__=='__main__':
    start=time.monotonic()
    while not (r/'mtp-moe-graph-live2-startup-outcome.json').exists():
        assert time.monotonic()-start<1100;time.sleep(10)
    assert json.loads((r/'mtp-moe-graph-live2-startup-outcome.json').read_text())['both_ready']
    functional=run('functional')
    paths=collect(r/'read_mtp_moe_live2_paths.py','mtp_moe_graph_live2_paths_v1','mtp-moe-graph-live2-paths',timeout=90)
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
            for call in x['calls']:
                counters[x['rank']][call['action']]+=1
                shapes.add(tuple(call['shape']))
                if row['mode']=='adaptive':assert call['eligible'],call
        assert all(counters[i]==counters[0] for i in range(16)),counters
        assert sum(counters[0].values())>0, (prefix,'No native core invocation for full request')
        count=sum(v for k,v in counters[0].items() if 'replay' in k)
        assert (count>0)==(row['mode']=='adaptive'),(row['case_id'],row['mode'],counters)
        proof.append({'case':row['case_id'],'mode':row['mode'],'all16':True,'native_actions_per_rank':dict(counters[0]),'shapes':sorted(shapes)})
    (r/'mtp-moe-graph-live2-path-summary-v1.json').write_text(json.dumps({'all16':True,'rows':proof,'scope':'Actual native MTP MoE calls/guards/cache actions, not physical task or graph output correctness proof.'},indent=2)+'\n')
    print(json.dumps({'phase':'all16_pure_ffn_path','passed':True}),flush=True)
    inherited=json.loads((r/'mtp-moe-graph-functional-outcome.json').read_text());assert inherited['completed'] and inherited['quality_full64_equal_cases']==26
    proxy=collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/start_proxy.py','mtp_moe_graph_live2_official_proxy_v1','mtp-moe-graph-live2-official-proxy',('--run-dir','/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-moe-graph-live2-20261018/node32','--expected-P-CID','12234db89fa8430727bba41ac176e8e21a81ecdd82a80eb07b26fc5c5b95c975','--minimum-lifetime-seconds','900'),timeout=100,node=32)
    assert proxy['ready']
    print(json.dumps({'phase':'official_proxy','CID':proxy['CID']}),flush=True)
    c4 = run('c4-functional')
    paths2 = collect(r/'read_mtp_moe_live2_paths.py','mtp_moe_graph_live2_paths_v2','mtp-moe-graph-live2-paths-v2',timeout=90)
    c4proof=[]
    for row in c4['rows']:
        prefix=row['chunks'][0]['id']
        joined=[x for x in paths2['rows'] if any(z==prefix or z.startswith(prefix+'-') for z in x['ids'])]
        assert sorted(set(x['rank'] for x in joined))==list(range(16)),(prefix,len(joined))
        selected={rank:sum('replay' in c['action'] for x in joined if x['rank']==rank for c in x['calls']) for rank in range(16)}
        assert len(set(selected.values()))==1 and (selected[0]>0)==(row['mode']=='adaptive'),selected
        c4proof.append({'case':row['case_id'],'mode':row['mode'],'all16':True,'selected_per_rank':selected[0]})
    (r/'mtp-moe-graph-live2-c4-path-summary-v1.json').write_text(json.dumps(c4proof,indent=2)+'\n')
    (r/'mtp-moe-graph-live2-functional-outcome.json').write_text(json.dumps({'completed':True,'quality_reused_from_same_build_live1':True,'quality_full64_equal_cases':26,'quality_cases':26,'all16_pure_ffn_path':True,'native_marker_restored':True,'next':'Physical copy/profile gate before formal service timing.'},indent=2)+'\n')
    print('functional_sequence_completed',flush=True)
