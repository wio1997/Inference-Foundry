"""Rebuild the standalone global view using only captured local evidence."""
from pathlib import Path
from collections import defaultdict
import json,statistics as st

P=Path(__file__).resolve().parent
ROOT=P.parent.parent
load=lambda name:json.loads((P/name).read_text())
findings=load('findings.json')
analysis=load('balanced-analysis.json')
inventory=load('kernel-full-inventory.json')
chain=load('chain-view-data.json')
chain['nodes']['idle']['title']='配平后两组共有的设备空档'
for key,node in chain['nodes'].items():
    if node.get('scope','').startswith('worker 前12'):
        node['scope']='early / steady 每窗最多12个worker迭代。默认显示steady；目标与草稿同类事件除DFC外尚未完整拆分。'

def family(name):
    if 'engram' in name or name in ['_hash_ids_kernel','_write_hash_cache_kernel']:return 'Engram'
    if name.startswith('DFC_') or name in ['MoeGatingTopKHash','DequantSwigluQuant']:return 'MoE'
    if name in ['HcPre','HcPost']:return 'mHC'
    if any(k in name for k in ['SparseFlashMla','LightningIndexer','indexer','Rotary','rotary']):return 'Attention'
    if 'dspark' in name or 'dflash' in name:return 'DSpark / 迭代'
    if 'slot_mapping' in name or name=='prepare_inputs_padded_kernel':return '输入 / cache'
    if 'rejection_' in name:return '校验 / 采样'
    if name.startswith('hcom_') or 'AicpuKernel' in name:return '通信'
    return '通用 / 位置待绑定'

workers=[r for r in analysis['files'] if r['dp'] is not None]
windows={(r['phase'],r['ep']):r['device_window_ms'] for r in workers}
groups={}
for phase in ['early','steady']:
    rows=[r for r in inventory['files'] if r['phase']==phase]
    names=sorted(set(o['operator_role'] for r in rows for o in r['operators']))
    groups[phase]={}
    for dp in [0,1]:
        ranks=[r for r in rows if r['dp']==dp]
        operators=[]
        for name in names:
            values=[];examples=[];seen=set()
            for rank in ranks:
                found=next((o for o in rank['operators'] if o['operator_role']==name),None)
                if not found:
                    values.append({'calls':0,'wall_sum_ms':0,'union_ms':0,'window_pct':0})
                    continue
                value={k:v for k,v in found.items() if k not in ['first_example','operator_role']}
                value['window_pct']=found['union_ms']/windows[(phase,rank['ep'])]*100
                values.append(value)
                ex=found['first_example']
                signature=tuple(ex.get(k) for k in ['Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Output Formats','Block Num','Mix Block Num'])
                if signature not in seen:
                    seen.add(signature);examples.append({'ep':rank['ep'],'csv':rank['path'],**ex})
            fields=set(k for v in values for k in v)
            med={k:st.median(v[k] for v in values if v.get(k) is not None) for k in fields if any(v.get(k) is not None for v in values)}
            operators.append({'name':name,'family':family(name),'m':med,'examples':examples})
        groups[phase]['dp'+str(dp)]={'operators':operators,'window_ms':st.median(windows[(phase,r['ep'])] for r in ranks)}

data={'findings':findings,'groups':groups,'ranks':workers,
      'gaps':load('steady-host-gaps.json')['representative_ranks'],
      'timeline':load('activity-timeline.json')}
detail_path=P.parent/'performance-breakdown-171'/'findings.json'
if detail_path.exists():data['detailed']=json.loads(detail_path.read_text())
idle_path=P.parent/'idle-attribution-171'/'findings.json'
if idle_path.exists():
    data['idle']=json.loads(idle_path.read_text())
    chain['nodes']['idle']['problem']='新增16卡阶段标记样本中，去掉首尾不完整循环后，约96%的设备空档与prepare input阶段重叠；完整窗口中也约94%。两组均一致。'
    chain['nodes']['idle']['next']='优先追下一轮输入/注意力metadata共同构造和交付。代表卡协调后至首个metadata提交前约31ms，其中约27ms与Free重叠，出现243次where事件。仍须给替代方案算净价。'
    chain['nodes']['idle']['evidence']='idle-attribution-171/REPORT.md与scoped-capture/scope-analysis.json。新阶段样本和本卡原Free数字分开记；阶段交集不等于全部可省。'
    chain['nodes']['metadata']['problem']='新增同负载阶段标记确认prepare input占约96%的Free。范围含请求状态、位置/长度、缓存映射、DP协调和_build_attention_metadata，不直接等于某个metadata kernel的耗时。'
    chain['nodes']['metadata']['next']='先查协调后、首次原生metadata提交前的参数构造窗口及CP各组公共数据，再核对可复用的同代计算、缓冲和批量发布。'
    chain['nodes']['metadata']['evidence']='model_runner_v1.py:2287–2530；legacy CPMetadataBuilder.build与_build_local_token_metadata；新样本代表EP0/8各11轮。'
(P/'global-view-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
html=(P/'chain-template.html').read_text().replace('__CHAIN_JSON__',json.dumps(chain,ensure_ascii=False))
html=html.replace('<title>171 · 推理全链路与卡点</title>','<title>171 · 推理时间花在哪里</title>')
html=html.replace('171 配平后的推理链路与卡点','171 · 推理时间花在哪里')
html=html.replace('DP0 / DP1 各4条镜像请求已验证 · 默认展示配平后的稳态热点','先看时间分布，再看最耗时的算子与待查原因')
html=html.replace('</style>',(P/'global-view.css').read_text()+(P/'performance-view.css').read_text()+'</style>',1)
html=html.replace('<section class="endpoint">','<div id="overview-panel" class="view-panel" hidden><section class="endpoint">',1)
html=html.replace('<footer class="footer">','</div>'+(P/'global-view-fragment.html').read_text()+(P/'idle-view-fragment.html').read_text()+(P/'performance-fragment.html').read_text()+'<footer class="footer">',1)
marker='<div id="overview-panel"'
nav='''<nav class="view-nav" aria-label="性能图层级"><button data-view="bill" aria-pressed="true">时间花在哪里</button><button data-view="overview">完整链路</button><button data-view="kernels">算子明细</button><button data-view="ranks">16卡时间线</button><button data-view="calls">提交与等待</button><button data-view="scope">数据范围</button></nav>'''
html=html.replace(marker,nav+(P/'plain-summary.html').read_text()+marker,1)
if 'idle' in data:
    html=html.replace('<button data-view="overview">','<button data-view="idle">空档在做什么</button><button data-view="overview">',1)
if 'detailed' in data:
    html=html.replace('<button data-view="bill" aria-pressed="true">时间花在哪里</button>','<button data-view="performance" aria-pressed="true">详细时间账单</button><button data-view="bill">原采样概览</button>',1)
    html=html.replace('171 · 推理时间花在哪里','171 · 推理性能明细')
    html=html.replace('先看时间分布，再看最耗时的算子与待查原因','同一份稳态采样：空档细分、CPU操作和全部设备算子')
    html=html.replace('<button data-view="kernels">算子明细</button>','<button data-view="kernels">原采样算子</button>',1)
html=html.replace('目标与草稿的同类设备事件尚未分开，当前合计值同时包含两者。','DFC已按目标/草稿分开；其余同类事件可同时包含两者，各节点预算可重叠。')
script='<script id="global-data" type="application/json">'+json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')+'</script><script>'+(P/'global-view.js').read_text()+'</script><script>'+(P/'performance-view.js').read_text()+'</script>'
html=html.replace('</body>',script+'</body>')
if P.parent.name=='evidence':
    html=html.replace('audit/balanced-profile-171/','').replace('audit/idle-attribution-171/','../idle-attribution-171/').replace('audit/performance-breakdown-171/','../performance-breakdown-171/')
    output=P/'index.html'
else:output=ROOT/'171-框架Profile热点.html'
output.write_text(html)
print('Built',len(html.encode()),'bytes;',len(groups['steady']['dp0']['operators']),'steady operator types')
