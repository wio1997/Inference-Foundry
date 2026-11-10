"""Upload explicit evidence blobs; final tree/ref is handled by connector."""
from pathlib import Path
import base64,concurrent.futures,hashlib,json,subprocess
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
REPO='repos/wio1997/Inference-Foundry/';BRANCH='910c-deepseek-v4.1-flash';BASE='projects/910c-deepseek-v4.1-framework-investigation/handoff/';DEST=BASE+'evidence/overlap-timeline-171/';BAL=BASE+'evidence/balanced-profile-171/'
FILES=['README.md','REPORT.md','STATUS.md','view-data.json','summary.json','candidate-coverage.csv','scenario-examples.json','model-validation.json','measured-timeline.json.gz','raw-archive.json','extract_remote.py','read_capture.py','patch_task_ids.py','build_data.py','write_report.py','verify_model.cjs','stage_publication.py','verify_publication.py']
SOURCES=['legacy-candidates.csv','project-review.csv','code-led-candidates.json','megakernel-cuts.json']
VIEW=['README.md','build_global_view.py','global-view.js','overlap-fragment.html','overlap-model.js','overlap-view.css','overlap-view.js','verify_global_view.mjs','view-validation.json']
def api(path,data=None):
 return json.loads(subprocess.check_output(['gh','api',REPO+path]+(['-X','POST','--input','-'] if data is not None else []),input=json.dumps(data).encode() if data is not None else None))
def build_payload():
 payload={DEST+n:(P/n).read_bytes() for n in FILES}
 payload.update({DEST+'sources/'+n:(P/'sources'/n).read_bytes() for n in SOURCES})
 payload.update({BAL+n:(P.parent/'balanced-profile-171'/n).read_bytes() for n in VIEW})
 payload[BAL+'README.md']=payload[BAL+'README.md'].replace('../../171-框架Profile热点.html'.encode(),b'index.html')
 html=(ROOT/'171-框架Profile热点.html').read_bytes()
 for old,new in [('audit/balanced-profile-171/',''),('audit/idle-attribution-171/','../idle-attribution-171/'),('audit/performance-breakdown-171/','../performance-breakdown-171/'),('audit/framework-timing-171/','../framework-timing-171/'),('tools/framework-timing/','../framework-timing-171/tool/'),('audit/overlap-timeline-171/','../overlap-timeline-171/')]:html=html.replace(old.encode(),new.encode())
 payload[BAL+'index.html']=html
 return payload

def upload(row):
 path,content=row;sha=api('git/blobs',{'content':base64.b64encode(content).decode(),'encoding':'base64'})['sha'];return {'path':path,'sha':sha,'type':'blob','mode':'100644'}
if __name__=='__main__':
 payload=build_payload();manifest={p:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for p,b in payload.items() if p.startswith(DEST)}
 payload[DEST+'manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(upload,payload.items()))
 head=api('git/ref/heads/'+BRANCH)['object']['sha'];tree=api('git/commits/'+head)['tree']['sha']
 readme=base64.b64decode(api('contents/'+BASE+'README.md?ref='+head)['content']).decode();tag='全量融合地图与空档覆盖条件模型'
 if tag not in readme:
  anchor='## 4. 优化地图与研究阶段\n';assert readme.count(anchor)==1
  note='\n2026-11-10 **'+tag+'**：[交互图](evidence/balanced-profile-171/index.html) · [分析/公式](evidence/overlap-timeline-171/REPORT.md) · [全量位置CSV](evidence/overlap-timeline-171/candidate-coverage.csv)。72条主台账历史记录+P01流水研究项、45条执行重审均映射到12模块；保留状态、scope和重复关系。复用16卡旧稳态样本：平均Free38.38%，全部消失的平均窗口容量界1.623×；另一轮固定CPU账本条件界1.047×，不是融合总上限。滑块展示数据释放/独立CPU/缓冲/新增成本的假设，默认新增覆盖0，不把历史百分比相加。本轮新NPU/请求/服务动作0。\n'
  readme=readme.replace(anchor,anchor+note,1)
 payload[BASE+'README.md']=readme.encode();entries.append(upload((BASE+'README.md',readme.encode())))
 bm=json.loads(base64.b64decode(api('contents/'+BAL+'manifest.json?ref='+head)['content']));bm.update({p:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for p,b in payload.items() if p.startswith(BAL)})
 payload[BAL+'manifest.json']=json.dumps(bm,ensure_ascii=False,indent=2).encode();entries.append(upload((BAL+'manifest.json',payload[BAL+'manifest.json'])))
 (P/'publication-plan.json').write_text(json.dumps({'head':head,'base_tree':tree,'repository':'wio1997/Inference-Foundry','branch':BRANCH,'entries':entries,'files':len(entries)},ensure_ascii=False,indent=2))
 (P/'publication-expected.json').write_text(json.dumps({p:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for p,b in payload.items()},ensure_ascii=False,indent=2))
 print(json.dumps({'head':head,'base_tree':tree,'files':len(entries),'bytes':sum(len(b) for b in payload.values())}))
