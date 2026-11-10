"""Publish this diagnosis and the updated one-file view, preserving branch peers."""
from pathlib import Path
import base64,concurrent.futures,hashlib,json,subprocess

P=Path(__file__).resolve().parent;ROOT=P.parent.parent
REPO='repos/wio1997/Inference-Foundry';BRANCH='910c-deepseek-v4.1-flash'
BASE='projects/910c-deepseek-v4.1-framework-investigation/handoff/'
DEST=BASE+'evidence/idle-attribution-171/';BAL=BASE+'evidence/balanced-profile-171/'
NAMES=['STATUS.md','REPORT.md','OFFLINE-REPORT.md','findings.json','analyze_idle.py',
       'probe_trace.py','schema-probe.json','idle-attribution.json','submission-summary.json',
       'source-map.json','additional-source-map.json','summarize.py','publish.py',
       'source/model_runner_v1.py','source/dspark_proposer.py','source/llm_base_proposer.py',
       'source/ascend-block_table.py','source/ascend-dsa_cp.py','source/ascend-dsa_v41_cp.py',
       'source/vllm-block_table.py','source/vllm-gpu_input_batch.py','source/vllm-utils.py',
       'scoped-capture/collect_remote.py','scoped-capture/launch_capture.py',
       'scoped-capture/zcode-collect.task.txt','scoped-capture/zcode-start-receipt.json',
       'scoped-capture/launch.json','scoped-capture/launch-output.txt','scoped-capture/poll_capture.py',
       'scoped-capture/export_when_restored.py','scoped-capture/export-launch.json',
       'scoped-capture/export-state.json','scoped-capture/analyze_scopes.py',
       'scoped-capture/scope-analysis.json','scoped-capture/inspect_prepare_boundaries.py',
       'scoped-capture/prepare-boundary-analysis.json','scoped-capture/prepare-input-markers.json',
       'scoped-capture/raw-scope-label-check.json','scoped-capture/receipt.json',
       'scoped-capture/restored-smoke.json','scoped-capture/balanced-reference-0.json',
       'scoped-capture/balanced-steady.json']
BAL_NAMES=['README.md','REPORT.md','build_global_view.py','finalize_report.py',
           'global-view.js','idle-view-fragment.html','verify_global_view.mjs','view-validation.json']
def api(path,obj=None,method='POST'):
    cmd=['gh','api',REPO+'/'+path]+(['-X',method,'--input','-'] if obj is not None else [])
    return json.loads(subprocess.check_output(cmd,input=json.dumps(obj).encode() if obj is not None else None))

if __name__=='__main__':
    payload={DEST+n:(P/n).read_bytes() for n in NAMES}
    payload.update({BAL+n:(P.parent/'balanced-profile-171'/n).read_bytes() for n in BAL_NAMES})
    for name in ['README.md','REPORT.md']:
        payload[BAL+name]=payload[BAL+name].replace('../../171-框架Profile热点.html'.encode(),b'index.html')
    payload[BAL+'index.html']=(ROOT/'171-框架Profile热点.html').read_bytes().replace(b'audit/balanced-profile-171/',b'').replace(b'audit/idle-attribution-171/',b'../idle-attribution-171/')
    first=api('git/ref/heads/'+BRANCH)['object']['sha']
    manifest={p:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for p,v in payload.items() if p.startswith(DEST)}
    payload[DEST+'manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
    remote=api('contents/'+BAL+'manifest.json?ref='+first)
    bm=json.loads(base64.b64decode(remote['content']))
    bm.update({p:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for p,v in payload.items() if p.startswith(BAL)})
    payload[BAL+'manifest.json']=json.dumps(bm,ensure_ascii=False,indent=2).encode()
    entries=[]
    for path,content in payload.items():
        blob=api('git/blobs',{'content':base64.b64encode(content).decode(),'encoding':'base64'})['sha']
        entries.append({'path':path,'mode':'100644','type':'blob','sha':blob})
    for attempt in range(3):
        head=api('git/ref/heads/'+BRANCH)['object']['sha']
        remote=api('contents/'+BASE+'README.md?ref='+head);text=base64.b64decode(remote['content']).decode()
        tag='设备空档已定位到输入准备'
        if tag not in text:
            anchor='## 4. 优化地图与研究阶段\n';assert text.count(anchor)==1
            note='\n2026-11-10 '+tag+'：[报告与代码入口](evidence/idle-attribution-171/REPORT.md) · [全局图](evidence/balanced-profile-171/README.md)。一次原生阶段标签补采，16卡闭合11循环：prepare input占Free约96.21%，草稿1.89%；代表卡协调后到首个metadata提交前约31ms，其中约27ms空档、243次where。优先价已有metadata-host / BatchFrame共同构造与交付，具体实现和净收益待确认，不重复旧信用。原服务已恢复，新增优化实现0。\n'
            text=text.replace(anchor,anchor+note,1)
        readme=BASE+'README.md';blob=api('git/blobs',{'content':base64.b64encode(text.encode()).decode(),'encoding':'base64'})['sha']
        base=api('git/commits/'+head)['tree']['sha'];tree=api('git/trees',{'base_tree':base,'tree':entries+[{'path':readme,'mode':'100644','type':'blob','sha':blob}]})['sha']
        commit=api('git/commits',{'message':'定位171设备空档到输入准备并补31ms参数构造窗口证据','tree':tree,'parents':[head]})['sha']
        try:api('git/refs/heads/'+BRANCH,{'sha':commit,'force':False},'PATCH');break
        except subprocess.CalledProcessError:
            if attempt==2:raise
    def verify(row):
        content=base64.b64decode(api('git/blobs/'+row['sha'])['content'])
        expected=text.encode() if row['path']==readme else payload[row['path']]
        assert content==expected,row['path']
        return row['path']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified=list(pool.map(verify,entries+[{'path':readme,'sha':blob}]))
    receipt={'commit':commit,'parent':head,'files':verified,'readback_equal':True,'model_optimizations':0}
    (P/'publication.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps({'commit':commit,'files':len(verified),'readback_equal':True}))
