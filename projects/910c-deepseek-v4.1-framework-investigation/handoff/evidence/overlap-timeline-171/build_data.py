from pathlib import Path
from collections import Counter
import csv,gzip,hashlib,json,statistics as st,subprocess
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
x=json.loads((P/'measured-timeline.json').read_text());ranks=x['ranks']
review={r['原编号']:r for r in csv.DictReader(open(P.parent/'算子开发立项复核.csv',encoding='utf-8-sig'))}
base=list(csv.DictReader(open(P.parent/'去重候选表.csv',encoding='utf-8-sig')))
code=json.load(open(P.parent/'code-led/candidates-ranked.json'))['rows']
mega=json.load(open(P.parent/'megakernel/候选与门槛.json'))['rows']
modules=[]
for i in range(12):
 mid=f'M{i:02}';s=(P.parent/'modules'/f'{mid}.md').read_text();title=next(l for l in s.splitlines() if l.startswith('# ')).split(' ',2)[-1];modules.append({'id':mid,'title':title})
segment_modules={'S00':['M00'],'S01':['M00','M11'],'S02':['M00','M11'],'S03':['M00','M11'],'S04':['M01'],'S05':['M00','M02'],'S06':['M02','M05','M07'],'S07':['M03','M04'],'S08':['M05','M06','M07'],'S09':['M08','M09'],'S10':['M10'],'S11':['M09','M11']}
cm={'C01':['M00'],'C02':['M00','M11'],'C03':['M00','M11'],'C04':['M00','M10','M11'],'C05':['M02','M07','M08','M10'],'C06':['M01'],'C07':['M02','M05','M07'],'C08':['M03'],'C09':['M03'],'C10':['M03','M04'],'C11':['M03'],'C12':['M05','M06'],'C13':['M06','M07','M10'],'C14':['M07','M10'],'C15':['M09','M10','M11']}
km={'K00':['M06'],'K01':['M02','M03'],'K02':['M03','M04'],'K03':['M06'],'K04':['M10'],'K05':['M07','M10'],'K06':['M01'],'K07':['M00','M11'],'K08':['M10'],'K09':[f'M{i:02}' for i in range(12)],'K10':['M07','M08','M09']}
# Relation links identify overlapping scopes, not additional independent gains.
aliases={'F01':['C03','K07'],'F02':['C03','K07'],'F03':['C03','K07'],'F04':['C07','K01'],'F05':['C11'],'F07':['C09','K02'],'F08':['C10','K02'],'F09':['C13','K00'],'H01':['C03','K07'],'R06':['C08','K01'],'R08':['C10','K02'],'R19':['C02','C15'],'R23':['C14','K05'],'R24':['K08'],'U08':['K09'],'C03':['F01','F02','F03','H01','K07'],'C07':['F04','K01'],'C08':['K01'],'C09':['F07','K02'],'C10':['F08','K02'],'C13':['K00','K03','K04'],'C14':['K05'],'C15':['P01'],'K00':['K03','C13'],'K01':['C07','C08'],'K02':['F07','F08','C09','C10'],'K03':['C13','K00'],'K04':['C13','K09'],'K05':['C14'],'K06':['C06'],'K07':['F01','F02','C03'],'K08':['R24'],'K09':['U08','K04'],'K10':['C05']}
types=[{'id':'math','label':'数学改写 / 计算消除与共享'},{'id':'kernel','label':'kernel融合 / 批量发布'},{'id':'mega','label':'megakernel / tile生产消费流水'},{'id':'comm','label':'通信与布局交付重组'},{'id':'state','label':'cache / 状态 / 数据复用'},{'id':'schedule','label':'跨层 / 跨迭代调度重叠'}]
mt={'M00':['kernel','math','state'],'M01':['math','state','kernel'],'M02':['math','kernel'],'M03':['kernel','math','comm'],'M04':['mega','comm'],'M05':['math','kernel'],'M06':['kernel','mega','comm'],'M07':['kernel','schedule'],'M08':['math','kernel'],'M09':['kernel','state'],'M10':['kernel','schedule'],'M11':['state','schedule','kernel']}
manual_types={'C01':['state','kernel'],'C02':['math','kernel','state','schedule'],'C03':['math','kernel','state'],'C04':['math','comm','schedule'],'C05':['math'],'C06':['math','state','schedule'],'C07':['math','kernel','mega'],'C08':['math','comm'],'C09':['math','mega','state'],'C10':['mega','comm'],'C11':['math','kernel'],'C12':['math','kernel'],'C13':['mega','comm','schedule'],'C14':['math','comm','kernel'],'C15':['schedule','state'],'K00':['mega','comm'],'K01':['mega','math','comm'],'K02':['mega','math','comm'],'K03':['mega','schedule'],'K04':['mega','schedule','comm'],'K05':['mega','math','comm'],'K06':['mega','state'],'K07':['kernel','math'],'K08':['mega'],'K09':['mega','schedule','comm'],'K10':['math','mega'],'P01':['schedule','state']}
# Dedicated kernel types only; shared MatMul/RMS types are marked as broad binding.
module_ops={'M00':['slot_mapping_kernel','prepare_inputs_padded_kernel'],'M01':['engram','Engram'],'M02':['HcPre','HcPost','RmsNorm'],'M03':['VllmAscendQuantLightningIndexerV2','SparseFlashMla','ScatterNdUpdateSk','InplacePartialRotaryMul'],'M04':['TransposeBatchMatMul','MatMulV2','hcom_alltoall_'],'M05':['HcPre','RmsNorm','MoeGatingTopKHash'],'M06':['DispatchFFNCombine','MoeGatingTopKHash','DequantSwigluQuant'],'M07':['HcPost','Add'],'M08':['MatMulV2','MatMulV3','RmsNorm'],'M09':['rejection','ArgMax','TopK'],'M10':['DispatchFFNCombine','MatMulV2','RmsNorm','ArgMax'],'M11':['slot_mapping','prepare_inputs_padded']}
records=[]
for r in base:
 ident=r['ID'];rev=review[ident];plan=rev['当前排期判断'];status='已确认可实现' if plan=='已有实测' else '已停止' if plan=='不立项' else '待确认'
 records.append({'id':ident,'name':r['候选组'],'modules':[r['收益归属']],'status':status,'scheduling':plan,'scope':r['适用负载'],'saves':r['能省什么'],'value':r['收益/上界/敏感性'],'conditions':r['改变取舍的条件'],'relations_text':r['重叠关系'],'evidence':r['固定版本证据'],'origin':'历史全量表','source_file':'audit/去重候选表.csv','review_reason':rev['依据']})
for r in code:
 ident=r['id'];rev=review[ident];plan=rev['当前排期判断'];records.append({'id':ident,'name':r['mechanism'],'modules':cm[ident],'status':'已确认可实现' if plan=='已有实测' else '待确认','scheduling':plan,'scope':r['workload'],'saves':r['saves'],'value':r['gain'],'conditions':r['condition'],'relations_text':r['overlap'],'evidence':'audit/code-led/candidates-ranked.json','origin':'代码交付链重审','source_file':'audit/code-led/candidates-ranked.json','review_reason':rev['依据']})
for r in mega:
 ident=r['切口ID'];status='已确认可实现' if ident=='K00' else '已停止' if ident in ['K06','K08'] else '待确认'
 records.append({'id':ident,'name':r['较大执行段'],'modules':km[ident],'status':status,'scheduling':'已在基线' if ident=='K00' else '历史实现有重叠' if ident=='K07' else '仅限该具体切口','scope':r['适用负载'],'saves':r['可消工作'],'value':r['理论值/上界/门槛'],'conditions':r['决定性条件'],'relations_text':r['来源归属'],'evidence':'audit/megakernel/候选与门槛.json','origin':'Megakernel切口','source_file':'audit/megakernel/候选与门槛.json','review_reason':r['status']})
records.append({'id':'P01','name':'上一轮计算覆盖下一轮Host准备','modules':['M00','M09','M10','M11'],'status':'待确认','scheduling':'本轮条件模型，未开发','scope':'固定C8稳态decode；其它批次另判','saves':'将独立的CPU计划提前到上一轮设备计算；保持接受数、草稿ID与live输入bank的晚交付','value':'可提前比例、数据释放与额外CPU/缓冲未确认；界限和敏感性见当前时间图','conditions':'先分离可提前字段与结果依赖字段；私有不可变CPU快照；producer_done与consumer_done；额外CPU资源及争用；真实关键路径改善','relations_text':'与C02/C03/K07共同准备消除重叠；与C15结果/草稿分离相关；设备加速后隐藏窗口会变短','evidence':'tools/framework-timing/TIMELINE.md; audit/execution-search/metadata-batch-frame-protocol.md','origin':'本轮跨迭代流水模型','source_file':'audit/overlap-timeline-171/README.md','review_reason':'理论敏感性，不是已测收益'})
assert len(records)==73 and len({r['id'] for r in records})==73
for r in records:
 r['types']=manual_types.get(r['id'],list(dict.fromkeys(t for m in r['modules'] for t in mt[m])))
 r['related']=aliases.get(r['id'],[])
 r['kernel_patterns']=list(dict.fromkeys(t for m in r['modules'] for t in module_ops[m]))
 if r['id']=='P01':r['kernel_patterns']=[]
 r['binding']='阶段级映射；通用Type可能同时属于多个调用点，未当作逐次因果绑定'
 r['historical_only']=r['status']=='已确认可实现'
 r['prefill_only']=r['id'] in ['K01','K02','F08','C08']
# Exact activity only for the two displayed representatives. Raw 16-card
# extraction is retained locally; all 16 summaries and provenance are kept.
reps=[]
for r in ranks:
 if r['ep'] not in [0,8]:continue
 a=r['cycles'][3]['start_ms'];b=r['cycles'][5]['end_ms']
 clip=lambda rows:[[round(max(a,s),6),round(min(b,e),6)] for s,e in rows if e>a and s<b]
 names=sorted({t['name'] for t in r['tasks']});ids={n:i for i,n in enumerate(names)}
 reps.append({'ep':r['ep'],'dp':r['dp'],'origin_us':r['origin_us'],'start_ms':a,'end_ms':b,'window_ms':b-a,'activity':{k:clip(v) for k,v in r['activity'].items()},'host':[h for h in r['host'] if h['end_ms']>a and h['start_ms']<b],'cycles':r['cycles'][3:6],'task_names':names,'tasks':[[ids[t['name']],round(t['start_ms'],6),round(t['end_ms'],6),int(t['target_graph']),t.get('stream'),t.get('task')] for t in r['tasks']]})
light=json.load(open(P.parent/'framework-timing-171/full-request-functions.json'));ref=json.load(open(P.parent/'framework-timing-171/balanced-reference-1.json'));cpu=[]
for p in light['processes']:
 rs=[r for r in p['rows'] if r['label']=='balanced-reference-1' and r['name'].split(':')[-1] in ['NPUModelRunner.execute_model','NPUModelRunner.sample_tokens']]
 if len(rs)==2:cpu.append({'pid':p['pid'],'thread_cpu_ms':sum(r['thread_cpu_ms'] for r in rs),'wall_ms':sum(r['duration_ms'] for r in rs)})
assert len(cpu)==16
avg={k:st.mean(r['totals'][k] for r in ranks) for k in ranks[0]['totals']};avg['window_ms']=st.mean(r['window_ms'] for r in ranks);avg['capacity_speedup']=avg['window_ms']/(avg['window_ms']-avg['free_ms']);avg['capacity_reduction_pct']=avg['free_ms']/avg['window_ms']*100
cpu_bound={'batch_ms':ref['batch_seconds']*1000,'mean_worker_cpu_ms':st.mean(r['thread_cpu_ms'] for r in cpu),'max_worker_cpu_ms':max(r['thread_cpu_ms'] for r in cpu),'workers':cpu};cpu_bound['fixed_cpu_demand_speedup_cap']=cpu_bound['batch_ms']/cpu_bound['max_worker_cpu_ms'];cpu_bound['fixed_cpu_demand_reduction_pct']=(1-cpu_bound['max_worker_cpu_ms']/cpu_bound['batch_ms'])*100
snapshots={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P.parent/'去重候选表.csv',P.parent/'算子开发立项复核.csv',P.parent/'code-led/candidates-ranked.json',P.parent/'megakernel/候选与门槛.json',P.parent/'framework-timing-171/full-request-functions.json']}
loop_modules={1:[2,3],2:[3],3:[2,3],4:[3],5:[4,5],6:[5,6,7],7:[6],8:[6,7],9:[3],10:[1,11],11:[6],12:[1,2,7],13:[9,10,11],14:[8,9,11],15:[0,11],16:[3,6,8],17:[0,10],18:[10],19:[0,10,11],20:[6,10],21:[0,11],22:[0,9,11],23:[9],24:[0],25:[0],26:[0],27:[0,10],28:[0],29:[0],30:[3],31:[3],32:[6],33:[6,11],34:[3],35:[3],36:[1],37:[3],38:[3],39:[3,4],40:[0,11],41:[4,5,7],42:[2,5,7],43:[1,2],44:[6],45:[0,11]}
loop_names={20:'非活跃DP与真实专家工作',21:'Prefix缓存及异常退役',22:'请求身份与缓存代次',23:'输出文本/SSE交付',24:'长prompt初始token生成',25:'输入渲染与tokenization队列',26:'稳定prefix tokenization共享',27:'启动预热与MM容量',28:'实际tokenizer分派与复用',29:'启动workspace与MM预算',30:'QLI量化query与共享成员发布',31:'IndexK/longKV多读者发布',32:'DFC GEMM tile与跨核激活',33:'原生workspace与末读退役',34:'SMLA metadata与schedule packet',35:'Selected KV到QK/PV L1流水',36:'Engram压缩scale与UVA寿命',37:'RMS/RoPE/量化/Scatter联合交付',38:'SWA/WKV/Q重叠与join',39:'head布局贯穿SMLA/O投影',40:'同代metadata与资源读者',41:'O投影/SP/残差到FFN',42:'HC双输出与premix提前交付',43:'Engram全局owner/WKV/gate/HC',44:'DFC输入行量化/路由到GMM',45:'有函数归因的共同metadata frame'}
research=[]
for folder in sorted((ROOT/'tasks/execution-derived-fusion-search/loops').glob('loop-*')):
 state_path=folder/'state.json';decision_path=folder/'decision.json';stt=json.loads(state_path.read_text());dec=json.loads(decision_path.read_text());num=int(folder.name.split('-')[-1]);report=(folder/'report.md').read_text()
 research.append({'id':folder.name,'title':loop_names.get(num,stt['title']),'original_title':stt['title'],'modules':[f'M{i:02}' for i in loop_modules.get(num,range(12))],
 'research_state':stt['status'],'verdict':dec.get('result'),'reason':dec.get('reason'),'next_action':stt.get('next_action'),'updated_at':stt.get('updated_at'),
 'source_path':str(state_path.relative_to(ROOT)),'state_sha256':hashlib.sha256(state_path.read_bytes()).hexdigest(),'report_sha256':hashlib.sha256(report.encode()).hexdigest(),
 'evidence_paths':dec.get('evidence_paths',[])})
# Snapshot primary registries so every displayed historical row has a stable source.
source_folder=P/'sources';source_folder.mkdir(exist_ok=True)
for source,name in [(P.parent/'去重候选表.csv','legacy-candidates.csv'),(P.parent/'算子开发立项复核.csv','project-review.csv'),(P.parent/'code-led/candidates-ranked.json','code-led-candidates.json'),(P.parent/'megakernel/候选与门槛.json','megakernel-cuts.json')]: (source_folder/name).write_bytes(source.read_bytes())
data={'schema':1,'modules':modules,'research':research,'types':types,'candidates':records,'representatives':reps,'global':avg,'cpu_bound':cpu_bound,'sources':snapshots,'snapshotted_git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),'source_capture':x['source_capture'],'new_npu_runs':0,'counts':{'historical_records':72,'current_pipeline_model':1,'total_records':73,'research_records':len(research),'independent_projects':'not implied','states':dict(Counter(r['status'] for r in records))},'rank_summaries':[{k:v for k,v in r.items() if k in ['ep','dp','source','sha256','totals','window_ms','cycles']} for r in ranks]}
(P/'view-data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
with (P/'candidate-coverage.csv').open('w',encoding='utf-8-sig',newline='') as f:
 fields=['id','name','modules','types','status','scheduling','scope','saves','value','conditions','related','evidence'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows({k:';'.join(r[k]) if isinstance(r[k],list) else r[k] for k in fields} for r in records)
(P/'summary.json').write_text(json.dumps({'global':avg,'cpu_bound':cpu_bound,'counts':data['counts'],'sources':snapshots,'raw_trace_read_only':True},ensure_ascii=False,indent=2))
print(json.dumps({'bytes':(P/'view-data.json').stat().st_size,'counts':data['counts'],'global':avg,'same_cpu_cap':cpu_bound['fixed_cpu_demand_speedup_cap']}))
