"""Publish an explicit evidence bundle with Git data APIs and non-forced CAS."""
from pathlib import Path
import base64,hashlib,json,subprocess,sys

P=Path(__file__).resolve().parent
ROOT=P.parent.parent
REPO='repos/wio1997/Inference-Foundry'
BRANCH='910c-deepseek-v4.1-flash'
HANDOFF='projects/910c-deepseek-v4.1-framework-investigation/handoff/'
DEST=HANDOFF+'evidence/balanced-profile-171/'
FILES=['README.md','STATUS.md','REPORT.md','findings.json','balanced-analysis.json',
       'kernel-pmu-analysis.json','kernel-full-inventory.json','activity-timeline.json',
       'steady-host-gaps.json','launch.json','receipt.json','restored-smoke.json',
       'balanced-reference-0.json','balanced-reference-1.json','balanced-reference-2.json',
       'balanced-early.json','balanced-steady.json','chain-view-data.json',
       'chain-template.html','plain-summary.html','global-view-fragment.html','global-view.css','global-view.js',
       'build_global_view.py','analyze_balanced.py','analyze_kernel_pmu.py',
       'analyze_host_gaps.py','export_full_kernel_inventory.py','export_activity_timeline.py',
       'finalize_report.py','collect_remote.py','view-validation.json']
def api(path,obj=None,method='POST'):
    args=['gh','api',REPO+'/'+path]+(['-X',method,'--input','-'] if obj is not None else [])
    return json.loads(subprocess.check_output(args,input=json.dumps(obj).encode() if obj is not None else None))

def refresh_view():
    names=['README.md','STATUS.md','REPORT.md','plain-summary.html','global-view.css',
           'global-view.js','build_global_view.py','finalize_report.py',
           'verify_global_view.mjs','view-validation.json','publish_workbench.py']
    payload={DEST+name:(P/name).read_bytes() for name in names}
    for name in ['README.md','STATUS.md','REPORT.md']:
        path=DEST+name
        payload[path]=payload[path].replace('../../171-框架Profile热点.html'.encode(),b'index.html')
    payload[DEST+'index.html']=(ROOT/'171-框架Profile热点.html').read_bytes().replace(b'audit/balanced-profile-171/',b'')
    head=api('git/ref/heads/'+BRANCH)['object']['sha']
    old=api('contents/'+DEST+'manifest.json?ref='+head)
    manifest=json.loads(base64.b64decode(old['content']))
    manifest.update({path:{'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()} for path,value in payload.items()})
    payload[DEST+'manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
    entries=[]
    for path,value in payload.items():
        sha=api('git/blobs',{'content':base64.b64encode(value).decode(),'encoding':'base64'})['sha']
        entries.append({'path':path,'mode':'100644','type':'blob','sha':sha})
    for attempt in range(3):
        head=api('git/ref/heads/'+BRANCH)['object']['sha']
        remote=api('contents/'+HANDOFF+'README.md?ref='+head)
        text=base64.b64decode(remote['content']).decode()
        tag='性能图首页改为时间账单'
        if tag not in text:
            anchor='## 4. 优化地图与研究阶段\n'
            assert text.count(anchor)==1
            text=text.replace(anchor,anchor+'\n2026-11-10 '+tag+'：[入口](evidence/balanced-profile-171/README.md)。稳态设备窗口中算子53%–55%、空档约38%、未重叠通信7%–9%；纯调度占比未分清。先看时间分布与MoE/DFC热点，再展开详细证据；本次仅展示更新，无新增采集。\n',1)
        readme=HANDOFF+'README.md'
        readme_blob=api('git/blobs',{'content':base64.b64encode(text.encode()).decode(),'encoding':'base64'})['sha']
        base=api('git/commits/'+head)['tree']['sha']
        tree=api('git/trees',{'base_tree':base,'tree':entries+[{'path':readme,'mode':'100644','type':'blob','sha':readme_blob}]})['sha']
        commit=api('git/commits',{'message':'简化171性能图首页，明确算子、空档、通信与调度证据','tree':tree,'parents':[head]})['sha']
        try:
            api('git/refs/heads/'+BRANCH,{'sha':commit,'force':False},'PATCH')
            break
        except subprocess.CalledProcessError:
            if attempt==2:raise
    for row in entries+[{'path':readme,'sha':readme_blob}]:
        blob=api('git/blobs/'+row['sha'])
        value=base64.b64decode(blob['content'])
        assert value==(text.encode() if row['path']==readme else payload[row['path']]),row['path']
    receipt={'commit':commit,'parent':head,'files':[readme]+list(payload),'readback_equal':True,'kind':'view-only; no new profiling'}
    (P/'view-publication.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps({'commit':commit,'files':len(receipt['files']),'readback_equal':True}))

if __name__=='__main__':
    if '--view-only' in sys.argv:
        refresh_view()
        sys.exit(0)
    payload={DEST+name:(P/name).read_bytes() for name in FILES}
    for name in ['README.md','STATUS.md','REPORT.md']:
        path=DEST+name
        payload[path]=payload[path].replace('../../171-框架Profile热点.html'.encode(),b'index.html')
    payload[DEST+'index.html']=(ROOT/'171-框架Profile热点.html').read_bytes().replace(b'audit/balanced-profile-171/',b'')
    manifest={path:{'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()} for path,value in payload.items()}
    payload[DEST+'manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
    # Blobs are immutable and may be reused if another writer advances the branch.
    entries=[]
    for path,value in payload.items():
        sha=api('git/blobs',{'content':base64.b64encode(value).decode(),'encoding':'base64'})['sha']
        entries.append({'path':path,'mode':'100644','type':'blob','sha':sha})
    for attempt in range(3):
        head=api('git/ref/heads/'+BRANCH)['object']['sha']
        remote=api('contents/'+HANDOFF+'README.md?ref='+head)
        text=base64.b64decode(remote['content']).decode()
        anchor='## 4. 优化地图与研究阶段\n'
        assert text.count(anchor)==1
        note='\n2026-11-10 配平复采完成：[框架→算子全局性能图与复用包](evidence/balanced-profile-171/README.md) · [报告](evidence/balanced-profile-171/REPORT.md)。DP各4镜像请求，参考中位16.091秒、509.11输出token/s；32worker/2前端trace，early88类/steady84类设备算子。稳态Gloo仅1.53%/2.21%，两组共有Free约38.4%，目标DFC最大。Free生产者及可省量待确认，新增优化实现0；灰色段仍待归因，不能标记全链路搜索完成。原服务已恢复。\n'
        assert '配平复采完成' not in text,'Already published; inspect instead of duplicating'
        old='DP1的Gloo等待与设备Free重叠中位约710ms，框架协调/组间进度为优先排查线索；具体可省量待确认。'
        text=text.replace(old,'旧首段DP1的Gloo与Free重叠中位约710ms；后续配平复采的steady两组Gloo均低，旧首段不可外推为稳态瓶颈。')
        text=text.replace(anchor,anchor+note,1)
        readme_path=HANDOFF+'README.md'
        readme_blob=api('git/blobs',{'content':base64.b64encode(text.encode()).decode(),'encoding':'base64'})['sha']
        base=api('git/commits/'+head)['tree']['sha']
        tree=api('git/trees',{'base_tree':base,'tree':entries+[{'path':readme_path,'mode':'100644','type':'blob','sha':readme_blob}]})['sha']
        commit=api('git/commits',{'message':'记录171配平profile与框架到算子全局性能图','tree':tree,'parents':[head]})['sha']
        try:
            api('git/refs/heads/'+BRANCH,{'sha':commit,'force':False},'PATCH')
            break
        except subprocess.CalledProcessError:
            if attempt==2:raise
    for row in entries+[{'path':readme_path,'sha':readme_blob}]:
        blob=api('git/blobs/'+row['sha'])
        value=base64.b64decode(blob['content'])
        expected=text.encode() if row['path']==readme_path else payload[row['path']]
        assert value==expected,row['path']
    final=api('git/ref/heads/'+BRANCH)['object']['sha']
    assert final==commit,'Branch advanced after readback; inspect ancestry'
    receipt={'commit':commit,'parent':head,'files':[readme_path]+list(payload),'readback_equal':True,'head_verified':True}
    (P/'publication.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps({'commit':commit,'files':len(receipt['files']),'readback_equal':True}))
