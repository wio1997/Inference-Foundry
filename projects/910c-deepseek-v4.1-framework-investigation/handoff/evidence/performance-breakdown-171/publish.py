"""Publish the detailed ledger and its view with a non-forced branch update."""
from pathlib import Path
import base64,concurrent.futures,hashlib,json,subprocess

P=Path(__file__).resolve().parent;ROOT=P.parent.parent
REPO='repos/wio1997/Inference-Foundry';BRANCH='910c-deepseek-v4.1-flash'
BASE='projects/910c-deepseek-v4.1-framework-investigation/handoff/'
DEST=BASE+'evidence/performance-breakdown-171/';BAL=BASE+'evidence/balanced-profile-171/'
FILES=['STATUS.md','REPORT.md','findings.json','rank-data.json','analyze_detailed.py','summarize.py','publish.py','operator-ranking.csv','host-breakdown.csv','host-operations.csv']
VIEW_FILES=['README.md','build_global_view.py','global-view.js','performance-fragment.html','performance-view.css','performance-view.js','verify_global_view.mjs','view-validation.json']
def api(path,obj=None,method='POST'):
    cmd=['gh','api',REPO+'/'+path]+(['-X',method,'--input','-'] if obj is not None else [])
    return json.loads(subprocess.check_output(cmd,input=json.dumps(obj).encode() if obj is not None else None))

if __name__=='__main__':
    payload={DEST+n:(P/n).read_bytes() for n in FILES}
    payload.update({BAL+n:(P.parent/'balanced-profile-171'/n).read_bytes() for n in VIEW_FILES})
    payload[BAL+'README.md']=payload[BAL+'README.md'].replace('../../171-框架Profile热点.html'.encode(),b'index.html')
    html=(ROOT/'171-框架Profile热点.html').read_bytes()
    html=html.replace(b'audit/balanced-profile-171/',b'').replace(b'audit/idle-attribution-171/',b'../idle-attribution-171/').replace(b'audit/performance-breakdown-171/',b'../performance-breakdown-171/')
    payload[BAL+'index.html']=html
    head=api('git/ref/heads/'+BRANCH)['object']['sha']
    manifest={p:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for p,v in payload.items() if p.startswith(DEST)}
    payload[DEST+'manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
    old=api('contents/'+BAL+'manifest.json?ref='+head)
    bm=json.loads(base64.b64decode(old['content']));bm.update({p:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for p,v in payload.items() if p.startswith(BAL)})
    payload[BAL+'manifest.json']=json.dumps(bm,ensure_ascii=False,indent=2).encode()
    entries=[]
    for path,content in payload.items():
        sha=api('git/blobs',{'content':base64.b64encode(content).decode(),'encoding':'base64'})['sha']
        entries.append({'path':path,'mode':'100644','type':'blob','sha':sha})
    for attempt in range(3):
        head=api('git/ref/heads/'+BRANCH)['object']['sha'];old=api('contents/'+BASE+'README.md?ref='+head)
        text=base64.b64decode(old['content']).decode();tag='同窗84类详细性能账单'
        if tag not in text:
            anchor='## 4. 优化地图与研究阶段\n';assert text.count(anchor)==1
            note='\n2026-11-10 '+tag+'：[完整排行/CSV/逐卡数据](evidence/performance-breakdown-171/REPORT.md) · [入口](evidence/balanced-profile-171/README.md)。同次阶段标记样本16卡各11完整轮，目标DFC144.54ms/17.22%、量化矩阵乘43.57ms/5.19%、MatMulV2 34.37ms/4.10%。输入准备的309.77ms空档继续按去嵌套CPU事件展开，其中155.35ms仍未定位到具体Python函数。79计算/辅助类+5通信全列，完整shape/图内分属可查；新NPU测试/实现0。\n'
            text=text.replace(anchor,anchor+note,1)
        readme=BASE+'README.md';blob=api('git/blobs',{'content':base64.b64encode(text.encode()).decode(),'encoding':'base64'})['sha']
        base=api('git/commits/'+head)['tree']['sha'];tree=api('git/trees',{'base_tree':base,'tree':entries+[{'path':readme,'mode':'100644','type':'blob','sha':blob}]})['sha']
        commit=api('git/commits',{'message':'细分171空档与全部84类算子同窗耗时并提供CSV','tree':tree,'parents':[head]})['sha']
        try:api('git/refs/heads/'+BRANCH,{'sha':commit,'force':False},'PATCH');break
        except subprocess.CalledProcessError:
            if attempt==2:raise
    def verify(row):
        data=base64.b64decode(api('git/blobs/'+row['sha'])['content'])
        assert data==(text.encode() if row['path']==readme else payload[row['path']]),row['path']
        return row['path']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified=list(pool.map(verify,entries+[{'path':readme,'sha':blob}]))
    receipt={'commit':commit,'parent':head,'files':verified,'readback_equal':True,'new_model_runs':0}
    (P/'publication.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps({'commit':commit,'files':len(verified),'readback_equal':True}))
