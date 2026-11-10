"""Upload explicit public blobs; GitHub connector creates tree/commit/ref separately."""
from pathlib import Path
import base64,concurrent.futures,hashlib,json,subprocess
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
REPO='repos/wio1997/Inference-Foundry/';BRANCH='910c-deepseek-v4.1-flash';BASE='projects/910c-deepseek-v4.1-framework-investigation/handoff/';DEST=BASE+'evidence/framework-timing-171/';BAL=BASE+'evidence/balanced-profile-171/'
FILES=['zcode-collect.task.txt','zcode-launch.json','REPORT.md','STATUS.md','COVERAGE.md','view-data.json','function-analysis.json','full-request-functions.json','detailed-performance.json','adapter.json','instrumented-functions.json','source-map.json','source-details.json','launch.json','launch-output.txt','receipt.json','restored-smoke.json','raw-inventory.json','historical-output-check.json','balanced-reference-0.json','balanced-reference-1.json','balanced-steady.json','scope-analysis.json','requests.csv','engine-core.csv','worker-full-request.csv','functions.csv','operators.csv','clock-alignment.csv','probe-costs.csv','collect_remote.py','launch_capture.py','prepare_adapter.py','read_sources.py','analyze_functions.py','analyze_detailed.py','analyze_scopes.py','build_report.py','build_source_details.py','fetch_evidence.py','export_when_restored.py','run_analysis.py','verify_restoration.py','stage_publication.py','analysis-state.json','function-body-state.json','export-state.json','export-attempt1-state.json','exporter-recovery.json','clock-diagnostic.json','unmatched-clock-diagnostic.json','captured-hook/ft_hook.py','captured-hook/sitecustomize.py','captured-hook/manifest.json']
SOURCE=['core.py','scheduler.py','gpu_model_runner.py','dsa_v41.py','device_metadata.py','multiproc_executor.py','shm_broadcast.py']
TOOLS=['README.md','TIMELINE.md','ft_hook.py','sitecustomize.py','verify_hook.py','adapter-template.json']
VIEW=['README.md','build_global_view.py','global-view.js','performance-view.css','function-fragment.html','function-view.js','verify_global_view.mjs','view-validation.json']
def api(path,data=None):
 cmd=['gh','api',REPO+path]+(['-X','POST','--input','-'] if data is not None else [])
 return json.loads(subprocess.check_output(cmd,input=json.dumps(data).encode() if data is not None else None))

def build_payload():
 payload={DEST+n:(P/n).read_bytes() for n in FILES}
 for n in SOURCE:payload[DEST+'source/'+n]=(P/'source'/n).read_bytes()
 for n in TOOLS:payload[DEST+'tool/'+n]=(ROOT/'tools/framework-timing'/n).read_bytes()
 for n in VIEW:payload[BAL+n]=(P.parent/'balanced-profile-171'/n).read_bytes()
 for n,b in list(payload.items()):
  if n.startswith(DEST) and n.endswith('.md'):payload[n]=b.replace(b'../../tools/framework-timing/',b'tool/')
 payload[BAL+'README.md']=payload[BAL+'README.md'].replace('../../171-框架Profile热点.html'.encode(),b'index.html')
 b=(ROOT/'171-框架Profile热点.html').read_bytes()
 for old,new in [('audit/balanced-profile-171/',''),('audit/idle-attribution-171/','../idle-attribution-171/'),('audit/performance-breakdown-171/','../performance-breakdown-171/'),('audit/framework-timing-171/','../framework-timing-171/'),('tools/framework-timing/','../framework-timing-171/tool/')]:b=b.replace(old.encode(),new.encode())
 payload[BAL+'index.html']=b
 return payload

def stage():
 payload=build_payload();head=api('git/ref/heads/'+BRANCH)['object']['sha'];commit=api('git/commits/'+head)
 readme=base64.b64decode(api('contents/'+BASE+'README.md?ref='+head)['content']).decode()
 tag='完整timeline逻辑与逐函数诊断'
 if tag not in readme:
  anchor='## 4. 优化地图与研究阶段\n';assert readme.count(anchor)==1
  note='\n2026-11-10 **'+tag+'**：[方法/验收](evidence/framework-timing-171/tool/TIMELINE.md) · [覆盖账](evidence/framework-timing-171/COVERAGE.md) · [函数到算子报告](evidence/framework-timing-171/REPORT.md) · [全景HTML](evidence/balanced-profile-171/index.html)。完整请求18进程轻量15.967999s/513.026输出token/s；16卡细窗42种已执行函数、84类设备/通信。主要Host候选为CP metadata重复构造；EngineCore调度/更新每组约172ms，等worker不能当同等调度成本。新增标记边界与时钟误差单列，保留API、批次关联、全请求连续设备及DFC内部缺口。一次有界补采已恢复health200/8-token/11 SHA不变，无新增性能实现。\n'
  readme=readme.replace(anchor,anchor+note,1)
 payload[BASE+'README.md']=readme.encode()
 manifest={path:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for path,b in payload.items() if path.startswith(DEST)}
 payload[DEST+'manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
 def upload(item):
  path,b=item;r=api('git/blobs',{'content':base64.b64encode(b).decode(),'encoding':'base64'})
  return {'path':path,'mode':'100644','type':'blob','sha':r['sha']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(upload,payload.items()))
 plan={'head':head,'base_tree':commit['tree']['sha'],'entries':entries,'files':len(entries),'payload_bytes':sum(len(b) for b in payload.values()),'repository':'wio1997/Inference-Foundry','branch':BRANCH}
 (P/'publication-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
 print(json.dumps({k:v for k,v in plan.items() if k!='entries'}))

if __name__=='__main__':stage()
