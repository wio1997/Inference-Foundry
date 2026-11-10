"""Stage explicit blobs only; connector performs tree/commit/CAS ref update."""
from pathlib import Path
import base64, concurrent.futures, hashlib, json, subprocess
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
REPO='repos/wio1997/Inference-Foundry/'
BRANCH='910c-deepseek-v4.1-flash'
BASE='projects/910c-deepseek-v4.1-framework-investigation/handoff/'
DEST=BASE+'evidence/gap-dataflow-171/'
BAL=BASE+'evidence/balanced-profile-171/'
FILES=['README.md','REPORT.md','STATUS.md','coverage.json','view-data.json','operations.csv','build_data.py',
       'source/indexer.py','source/compressor.py','stage_publication.py','verify_publication.py']
VIEW=['README.md','build_global_view.py','global-view.js','idle-view-fragment.html','dataflow-fragment.html',
      'dataflow-view.css','dataflow-view.js','verify_global_view.mjs','view-validation.json']
def api(path,data=None):
    return json.loads(subprocess.check_output(['gh','api',REPO+path]+(['-X','POST','--input','-'] if data is not None else []),
        input=json.dumps(data).encode() if data is not None else None))
def upload(row):
    path,content=row
    sha=api('git/blobs',{'content':base64.b64encode(content).decode(),'encoding':'base64'})['sha']
    return {'path':path,'sha':sha,'type':'blob','mode':'100644'}
def build_payload():
    payload={DEST+n:(P/n).read_bytes() for n in FILES}
    payload[DEST+'REPORT.md']=payload[DEST+'REPORT.md'].replace(b'](../../audit/',b'](../')
    payload[DEST+'README.md']=payload[DEST+'README.md'].replace('../../171-框架Profile热点.html'.encode(),b'../balanced-profile-171/index.html')
    payload.update({BAL+n:(P.parent/'balanced-profile-171'/n).read_bytes() for n in VIEW})
    payload[BAL+'README.md']=payload[BAL+'README.md'].replace('../../171-框架Profile热点.html'.encode(),b'index.html')
    html=(ROOT/'171-框架Profile热点.html').read_bytes()
    for old,new in [('audit/balanced-profile-171/',''),('audit/idle-attribution-171/','../idle-attribution-171/'),
                    ('audit/performance-breakdown-171/','../performance-breakdown-171/'),('audit/framework-timing-171/','../framework-timing-171/'),
                    ('tools/framework-timing/','../framework-timing-171/tool/'),('audit/overlap-timeline-171/','../overlap-timeline-171/'),
                    ('audit/gap-dataflow-171/','../gap-dataflow-171/')]:
        html=html.replace(old.encode(),new.encode())
    payload[BAL+'index.html']=html
    return payload
if __name__=='__main__':
    payload=build_payload()
    payload[DEST+'manifest.json']=json.dumps({p:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        for p,b in payload.items() if p.startswith(DEST)},ensure_ascii=False,indent=2).encode()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(upload,payload.items()))
    head=api('git/ref/heads/'+BRANCH)['object']['sha'];tree=api('git/commits/'+head)['tree']['sha']
    readme=base64.b64decode(api('contents/'+BASE+'README.md?ref='+head)['content']).decode()
    tag='空档操作、数据用途与真实消费者'
    if tag not in readme:
        anchor='## 4. 优化地图与研究阶段\n';assert readme.count(anchor)==1
        note='\n2026-11-10 **'+tag+'**：[全局图默认页](evidence/balanced-profile-171/index.html) · [逐项说明](evidence/gap-dataflow-171/REPORT.md) · [接续断点](evidence/gap-dataflow-171/STATUS.md) · [数据字典CSV](evidence/gap-dataflow-171/operations.csv)。28项/8阶段/9份冻结源；逐项给读取、生成、作用、消费者、提前依赖及代码。CP/base/local helper自身占细函数诊断准备空档65.10%；字段独立时间未测，同一函数不重复加价。真实KV数据读写与地址/metadata分开，附同窗算子shape；未认领缓存池容量为读流量。复用旧profile与loop046读者证据，本轮新采集/模型/服务动作0，全模型搜索仍未扫全。\n'
        readme=readme.replace(anchor,anchor+note,1)
    payload[BASE+'README.md']=readme.encode();entries.append(upload((BASE+'README.md',readme.encode())))
    bm=json.loads(base64.b64decode(api('contents/'+BAL+'manifest.json?ref='+head)['content']))
    bm.update({p:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for p,b in payload.items() if p.startswith(BAL)})
    payload[BAL+'manifest.json']=json.dumps(bm,ensure_ascii=False,indent=2).encode()
    entries.append(upload((BAL+'manifest.json',payload[BAL+'manifest.json'])))
    (P/'publication-plan.json').write_text(json.dumps({'head':head,'base_tree':tree,'repository':'wio1997/Inference-Foundry','branch':BRANCH,'entries':entries,'files':len(entries)},ensure_ascii=False,indent=2))
    (P/'publication-expected.json').write_text(json.dumps({p:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for p,b in payload.items()},ensure_ascii=False,indent=2))
    print(json.dumps({'head':head,'base_tree':tree,'files':len(entries),'bytes':sum(len(b) for b in payload.values())}))
