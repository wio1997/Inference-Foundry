"""Publish explicit token-ledger blobs; branch mutation uses the GitHub connector."""
from pathlib import Path
import base64,concurrent.futures,hashlib,json,re,subprocess,urllib.request,urllib.parse
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
REPO='repos/wio1997/Inference-Foundry/';BRANCH='910c-deepseek-v4.1-flash'
BASE='projects/910c-deepseek-v4.1-framework-investigation/handoff/'
DEST=BASE+'evidence/token-efficiency-171/';BAL=BASE+'evidence/balanced-profile-171/'
FILES=['README.md','REPORT.md','STATUS.md','build_data.py','source-evidence.json','view-data.json','draft-dp-waves.csv','validation.json','stage_publication.py','verify_publication.py']
VIEW=['README.md','build_global_view.py','token-fragment.html','token-view.css','token-view.js','verify_global_view.mjs','view-validation.json']
def api(path,data=None):
    return json.loads(subprocess.check_output(['gh','api',REPO+path]+(['-X','POST','--input','-'] if data is not None else []),input=json.dumps(data).encode() if data is not None else None,timeout=60))
def raw(path,commit):
    url='https://raw.githubusercontent.com/wio1997/Inference-Foundry/'+commit+'/'+urllib.parse.quote(path)
    with urllib.request.urlopen(url,timeout=35) as r:return r.read()
def upload(row):
    path,b=row;sha=api('git/blobs',{'content':base64.b64encode(b).decode(),'encoding':'base64'})['sha']
    assert sha==hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    return {'path':path,'sha':sha,'type':'blob','mode':'100644'}
def rewrite(b):
    for old,new in [('audit/balanced-profile-171/',''),('audit/idle-attribution-171/','../idle-attribution-171/'),('audit/performance-breakdown-171/','../performance-breakdown-171/'),('audit/framework-timing-171/','../framework-timing-171/'),('tools/framework-timing/','../framework-timing-171/tool/'),('audit/overlap-timeline-171/','../overlap-timeline-171/'),('audit/gap-dataflow-171/','../gap-dataflow-171/'),('audit/token-efficiency-171/','../token-efficiency-171/'),('audit/execution-search/','../execution-search/')]:b=b.replace(old.encode(),new.encode())
    return b
if __name__=='__main__':
    head=api('git/ref/heads/'+BRANCH)['object']['sha'];tree=api('git/commits/'+head)['tree']['sha']
    payload={DEST+n:(P/n).read_bytes() for n in FILES}
    payload[DEST+'README.md']=payload[DEST+'README.md'].replace('../../171-框架Profile热点.html'.encode(),b'../balanced-profile-171/index.html')
    payload.update({BAL+n:(P.parent/'balanced-profile-171'/n).read_bytes() for n in VIEW})
    payload[BAL+'README.md']=payload[BAL+'README.md'].replace('../../171-框架Profile热点.html'.encode(),b'index.html')
    html=rewrite((ROOT/'171-框架Profile热点.html').read_bytes()).decode()
    # Research entry asides are maintained by another chat; preserve its latest head.
    remote=raw(BAL+'index.html',head).decode()
    html=re.sub(r'<aside id="[^"]+-default-entry"[^>]*>.*?</aside>','',html,flags=re.S)
    for block,ident in re.findall(r'(<aside id="([^"]+)"[^>]*>.*?</aside>)',remote,re.S):
        if f'id="{ident}"' not in html:html=html.replace('<body>','<body>'+block,1)
    payload[BAL+'index.html']=html.encode()
    payload[DEST+'manifest.json']=json.dumps({p:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for p,b in payload.items() if p.startswith(DEST)},ensure_ascii=False,indent=2).encode()
    readme=raw(BASE+'README.md',head).decode();tag='有效Token与Padding分段账本'
    if tag not in readme:
        anchor='## 4. 优化地图与研究阶段\n';assert readme.count(anchor)==1
        note='\n2026-11-10 **'+tag+'**：[全局图新增页](evidence/balanced-profile-171/index.html) · [报告](evidence/token-efficiency-171/REPORT.md) · [接续断点](evidence/token-efficiency-171/STATUS.md) · [454组逐波CSV](evidence/token-efficiency-171/draft-dp-waves.csv)。新配平批次输出8192/草稿接受73.25%；原C8草稿body有效8840/物理10688行，额外17.29%，主要TP8对齐。两cohort和行/时间分开；当前全模型pad比例未知。Target若实际进24行桶，B1..4条件pad75/50/25/0%，桶人口未测。本轮新profile/模型请求/服务动作0，原全模型源码排查继续。\n'
        readme=readme.replace(anchor,anchor+note,1)
    payload[BASE+'README.md']=readme.encode()
    bm=json.loads(raw(BAL+'manifest.json',head))
    bm.update({p:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for p,b in payload.items() if p.startswith(BAL)})
    payload[BAL+'manifest.json']=json.dumps(bm,ensure_ascii=False,indent=2).encode()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(upload,payload.items()))
    (P/'publication-plan.json').write_text(json.dumps({'head':head,'base_tree':tree,'repository':'wio1997/Inference-Foundry','branch':BRANCH,'entries':entries},ensure_ascii=False,indent=2))
    (P/'publication-expected.json').write_text(json.dumps({p:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for p,b in payload.items()},ensure_ascii=False,indent=2))
    print(json.dumps({'head':head,'base_tree':tree,'files':len(entries),'bytes':sum(len(b) for b in payload.values())}))
