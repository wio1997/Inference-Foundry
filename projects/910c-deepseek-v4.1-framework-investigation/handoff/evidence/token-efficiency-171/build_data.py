"""Reconcile logical tokens, speculative acceptance, and observed DSpark row extents.

Uses existing snapshots only. Counts are kept in separate cohorts and units.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json

P=Path(__file__).resolve().parent
ROOT=P.parents[1]
def evidence(path):
    return P.parent/path.removeprefix('audit/') if P.parent.name=='evidence' else ROOT/path
def load(path):return json.loads(evidence(path).read_text())
def digest(path):return hashlib.sha256(evidence(path).read_bytes()).hexdigest()
def total(counter):return sum(counter.values())

def main():
    new_path='audit/framework-timing-171/balanced-reference-1.json'
    j=load(new_path);c=j['counter_deltas']
    request_input=sum(r['usage']['prompt_tokens'] for r in j['requests'])
    output=sum(r['usage']['completion_tokens'] for r in j['requests'])
    assert len(j['requests'])==8 and total(c['request_success_total'])==8
    assert request_input==total(c['prompt_tokens_total']) and output==total(c['generation_tokens_total'])
    drafts=total(c['spec_decode_num_draft_tokens_total']);accepted=total(c['spec_decode_num_accepted_tokens_total'])
    assert 0<=accepted<=drafts
    requests={'cohort':'20261110T090101Z / balanced-reference-1；同轮细函数采集的轻量参考批次',
              'requests':8,'dp_requests':dict(Counter(str(r['dp_rank']) for r in j['requests'])),
              'prompt_tokens':request_input,'delivered_output_tokens':output,
              'prefix_queries':total(c['prefix_cache_queries_total']),'prefix_hits':total(c['prefix_cache_hits_total']),
              'draft_tokens':drafts,'accepted_draft_tokens':accepted,'not_accepted_draft_tokens':drafts-accepted,
              'draft_accept_pct':100*accepted/drafts,'batch_seconds':j['batch_seconds'],
              'output_tokens_s':j['output_tokens_per_second'],'whole_model_scheduled_tokens':None,'whole_model_pad_pct':None,
              'notes':['请求输入量包含命中的prefix，不等于重新计算的prompt量。',
                       '未接受草稿在target校验时属于真实校验工作；不能一律当shape padding或可删工作。',
                       'accepted+draft轮数不强行等于交付输出数，终止/截断与统计边界另核。',
                       '本记录不含每轮actual/padded字段，无法统计当前全框架padding比例。']}
    requests['prefix_query_minus_hit']=requests['prefix_queries']-requests['prefix_hits']

    old_path='audit/execution-search/dspark-three-body-frontier.json'
    f=load(old_path);cols=f['body_row_columns'];groups=defaultdict(list)
    for raw in f['body_rows']:
        row=dict(zip(cols,raw))
        if row['wave_active']:groups[(row['wave'],row['rank']//8)].append(row)
    waves=defaultdict(dict)
    for (wave,dp),rows in groups.items():
        assert len(rows)==8
        assert len({r['rank_active'] for r in rows})==1
        assert len({r['head_M'] for r in rows})==1
        assert len({r['local_SP_M'] for r in rows})==1
        active=bool(rows[0]['rank_active']);head=rows[0]['head_M'];assert head%5==0
        waves[wave][dp]={'rows':rows,'own_B':head//5 if active else 0,'synthetic':not active}
    records=[];families=Counter();summary=Counter();native_rank_windows=0
    for wave,pair in sorted(waves.items()):
        assert set(pair)=={0,1}
        max_b=max(x['own_B'] for x in pair.values());assert 1<=max_b<=4
        for dp,x in sorted(pair.items()):
            rows=x['rows'];m=rows[0]['local_SP_M'];physical=8*m;live=5*x['own_B'];uniform=5*max_b
            assert physical==8*((uniform+7)//8)
            alignment=physical-uniform
            dp_pad=uniform-live if not x['synthetic'] else 0
            synthetic=uniform if x['synthetic'] else 0
            assert live+alignment+dp_pad+synthetic==physical
            row={'wave':wave,'dp':dp,'own_B':x['own_B'],'max_B':max_b,'synthetic_dummy':x['synthetic'],
                 'live_query_rows':live,'dp_uniform_rows':uniform,'local_SP_M':m,'physical_rows':physical,
                 'alignment_pad_rows':alignment,'dp_extra_rows':dp_pad,'synthetic_rows':synthetic,
                 'overhead_pct':100*(physical-live)/physical,'first_HcPre_tasks':{str(r['rank']):r['first_HcPre_TASK_ROW'] for r in rows}}
            records.append(row);families[(x['own_B'],max_b,physical)]+=1
            summary.update(live_query_rows=live,physical_rows=physical,alignment_pad_rows=alignment,
                           dp_extra_rows=dp_pad,synthetic_rows=synthetic)
            native_rank_windows+=len(rows)
    assert len(waves)==227 and len(records)==454 and native_rank_windows==3632
    for key in ['live_query_rows','alignment_pad_rows','dp_extra_rows','synthetic_rows']:
        summary[key+'_pct']=100*summary[key]/summary['physical_rows']
    summary['overhead_rows']=summary['physical_rows']-summary['live_query_rows']
    summary['overhead_pct']=100*summary['overhead_rows']/summary['physical_rows']
    old={'cohort':'原C8 / 20261102T1901 profile，8条原请求；与镜像配平新批次分开',
         'unit':'每层SP行域、按DP组跨8卡合并；不重复计TP副本，不是最终生成token数或FLOPs',
         'waves':227,'dp_wave_groups':454,'native_rank_windows':native_rank_windows,'summary':dict(summary),
         'families':[{'own_B':b,'max_B':bm,'physical_rows':p,'dp_wave_groups':n,'live_rows':5*b,'overhead_pct':100*(p-5*b)/p} for (b,bm,p),n in families.most_common()],
         'records':records,'three_layer_factor':3,
         'qualification':'原profile的HcPre/SP/head形状和调用家族，经固定源码还原行域；不证明各算子都执行了全部pad行，不能把17.29%当延迟收益。',
         'all_model_pad_pct':None}
    design=[]
    for b in range(1,5):
        for kind,width in [('target_uniform',6),('draft_balanced',5)]:
            n=width*b;padded=8*((n+7)//8)
            design.append({'path':kind,'B':b,'logical_rows':n,'physical_rows':padded,'pad_rows':padded-n,'pad_pct':100*(padded-n)/padded,
                           'qualification':'固定TP8、uniform宽度下的源码推导；不作为本批次逐轮实测人口'})
    layers=[
      {'name':'请求与输出','useful':'prompt / 实际交付output','extent':'请求usage与全局计数器','status':'新批次已统计','finding':f'8请求，输入{request_input:,}、输出{output:,}；prefix命中另列。'},
      {'name':'Target输入与图档位','useful':'scheduler.total_num_scheduled_tokens（含有效投机校验位置）','extent':'SP补齐→graph bucket→DP统一后num_tokens_padded','status':'源码已核，逐轮人口缺失','finding':'TP8：6/12/18/24行至少变为8/16/24/24行；最终图档位还要按实际dispatcher记。'},
      {'name':'DSpark body / dense行域','useful':'5×本DP真实请求数','extent':'8×ceil(5×DPmaxB/8)','status':'原C8已还原454个DP波','finding':f'有效{summary["live_query_rows"]:,}/物理{summary["physical_rows"]:,}行；额外行{summary["overhead_pct"]:.2f}%。'},
      {'name':'Attention / KV读写','useful':'CP实际query、seq_lens、有效slot和稀疏索引','extent':'Q tensor / 分页cache池 / graph padding容量','status':'有效域保护已核；实际读流量未统计','finding':'CP裁剪actual token、无效slot=-1、长度/索引限制读取；cache池shape不等于每次读全部。'},
      {'name':'MoE专家','useful':'mask后sender、实际recv及每expert行数','extent':'SP carrier / top-k路由扩展 / GEMM tile','status':'原生计数历史证据可复用，当前全批尚未合账','finding':'本卡query=0仍可能接收远端expert任务；top-k复制不是padding；空expert GMM已有guard。'},
      {'name':'拒绝草稿与context占位','useful':'当前阶段仍被真实消费者需要的行','extent':'prepare_inputs_padded保留旧QSL并用selector过滤','status':'新批次接受率已统计，占位传播量未测','finding':'拒绝草稿在校验时是真实工作；之后可作为padding保留，必须按阶段追踪，不能一律把num_actual_tokens当业务有效数。'},
      {'name':'预分配与硬件tile','useful':'有效字节/逻辑M,N,K','extent':'池容量、persistent buffer、tile对齐extent','status':'应分存储与执行单独统计','finding':'大buffer不表示全量计算；tile尾块占比也不等于耗时占比。'},
    ]
    sources=['audit/framework-timing-171/balanced-reference-1.json',old_path,'audit/execution-search/worker-capture-family-frontier.json',
             'audit/execution-search/worker-capture-source-review.json']
    conditional_graph=[{'B':b,'scheduled':6*b,'bucket':24,'pad_pct':100*(24-6*b)/24} for b in range(1,5)]
    d={'title':'有效Token与Padding账本','date':'2026-11-10','requests':requests,'historical_draft':old,'design_cases':design,'layers':layers,
       'conditional_graph24':conditional_graph,
       'source_evidence':json.loads((P/'source-evidence.json').read_text()),'inputs':[{'path':s,'sha256':digest(s)} for s in sources],
       'capture_contract':{'per_iteration':['cohort / dp / iteration / real-or-dummy / prefill-or-decode / target-or-draft',
           'real request count / scheduled tokens / DP-max token count / graph mode & bucket',
           'stage-specific live count / tensor extent / validity predicate / valid cache slots',
           'valid MoE senders / actual receiver rows / rejected-context placeholders'],
           'collection':'在计数已位于CPU或原有日志时直接记录；设备mask用设备端归约累积、批末异步读一次，避免每轮.item()/synchronize()。预填/解码、target/draft、dummy分别合账。',
           'aggregation':'先按DP组/交付点去重；逐阶段sum(live)/sum(extent)，不平均单轮百分比，不把不同阶段或不同采样相加。'},
       'new_profiles':0,'new_model_requests':0,'service_actions':0,'whole_model_complete':False}
    (P/'view-data.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    with (P/'draft-dp-waves.csv').open('w') as out:
        keys=[k for k in records[0] if k!='first_HcPre_tasks'];w=csv.DictWriter(out,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows({k:r[k] for k in keys} for r in records)
    write_report(d)
    (P/'validation.json').write_text(json.dumps({'requests_match_counters':True,'request_count':8,'all_wave_groups_have_8_ranks':True,
        'two_dp_groups_per_active_wave':True,'all_454_row_partitions_reconcile':True,'source_summary_matches':len(records)==454,
        'native_rank_windows':native_rank_windows,'live_rows':summary['live_query_rows'],'physical_rows':summary['physical_rows'],
        'unknown_whole_model_percentage_kept_null':requests['whole_model_pad_pct'] is None},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output_tokens':output,'draft_tokens':drafts,'accepted_draft':accepted,'accept_pct':requests['draft_accept_pct'],
        'historical_draft_live_rows':summary['live_query_rows'],'historical_draft_physical_rows':summary['physical_rows'],'row_overhead_pct':summary['overhead_pct']},ensure_ascii=False))

def write_report(d):
    r=d['requests'];h=d['historical_draft'];s=h['summary']
    out=['# 有效Token与Padding：统计口径与已有结果','',
         '**存在padding，但必须按执行阶段统计。** 已还原的原C8草稿body行域中，额外行占17.29%；最新配平批次的完整框架padding比例尚未记录，保留未知。两组数据不能互换。','',
         '## 1. 三种“有效”分开记','',
         '- 业务产出：最终交付给用户的token。',
         '- 算法所需位置：当前阶段要处理的真实query/校验/状态位置；被拒绝草稿在被校验时仍属于真实工作。',
         '- 执行行域：tensor或SP行域实际传给算子的行数，可能含DP统一、对齐、图档位和占位。硬件tile与buffer容量再另列。','',
         '`行域利用率 = Σ当前阶段有效行 / Σ该阶段执行行域`；`草稿接受率 = 接受草稿 / 提出草稿`；`输出吞吐 = 实际交付token / 请求批次时间`。三者不是同一个指标。','',
         '## 2. 新批次：实际输入、输出与草稿计数','',r['cohort'],'',
         '|项目|token数|口径|','|---|---:|---|',
         f'|请求输入|{r["prompt_tokens"]:,}|8×32768，含命中的prefix|',
         f'|实际输出|{r["delivered_output_tokens"]:,}|8×1024；TP8副本不重复计|',
         f'|prefix命中|{r["prefix_hits"]:,}|查询{r["prefix_queries"]:,}；查询减命中{r["prefix_query_minus_hit"]:,}不直接冒充GPU计算token数|',
         f'|提出草稿|{r["draft_tokens"]:,.0f}|全局计数器DP0+DP1|',
         f'|接受草稿|{r["accepted_draft_tokens"]:,.0f}|接受率{r["draft_accept_pct"]:.2f}%|',
         f'|未接受草稿|{r["not_accepted_draft_tokens"]:,.0f}|属于投机算法成本；不是全部shape padding|','',
         '请求usage与全局prompt/generation计数严格核对。当前完整scheduled tokens、逐轮actual/padded以及每阶段validity未记录，不能由这些计数推出全模型padding占比。','',
         '## 3. 原C8草稿body：由已有profile还原的行域','',h['cohort'],'',
         '227个活跃波×2个DP=454组，关联3632个rank窗。以每层行域计：真实query为`5×ownB`，DP统一为`5×maxB`，TP8载体为`8×ceil(5×maxB/8)`。三层相同scope可都乘3，比例不变；这不是8840个最终输出。','',
         '|分类|行数/每层|占物理行域|','|---|---:|---:|']
    for name,key in [('真实草稿query','live_query_rows'),('TP8对齐补齐','alignment_pad_rows'),('DP统一额外行','dp_extra_rows'),('无请求侧synthetic行','synthetic_rows')]:
        out.append(f'|{name}|{s[key]:,}|{s[key+"_pct"]:.4f}%|')
    out += [f'|合计物理行域|{s["physical_rows"]:,}|100%|',f'|额外行合计|{s["overhead_rows"]:,}|{s["overhead_pct"]:.4f}%|','',
            '**17.29%是该草稿body行域的额外行比例，不是全框架无效计算比例、FLOPs比例或可兑现时间收益。** CP已经裁剪部分query；MoE已经按mask过滤部分sender；dummy rank仍可能为远端真实token处理专家。各算子执行哪些额外行，需要逐消费者核对。','',
            '典型例：满批草稿20→24行，4行补齐，占16.67%；B1草稿5→8，占37.5%；非对称DP中ownB1/maxB2为5→16，占68.75%。这些分别包含不同的对齐与DP成本。','',
            '|真实own B|DP max B|物理行域|DP波组数|本组额外行比例|','|---:|---:|---:|---:|---:|']
    for x in h['families']:out.append(f'|{x["own_B"]}|{x["max_B"]}|{x["physical_rows"]}|{x["dp_wave_groups"]}|{x["overhead_pct"]:.2f}%|')
    out+=['','逐波原始关联及对账：[draft-dp-waves.csv](draft-dp-waves.csv)；Task ID保存在JSON。','',
          '## 4. 每一段要查什么','', '|阶段|有效域 / 执行域|当前结果|','|---|---|---|']
    for x in d['layers']:out.append(f'|{x["name"]}|{x["useful"]} / {x["extent"]}|{x["status"]}：{x["finding"]}|')
    out += ['', '额外说明：`prepare_inputs_padded`显式保留被拒绝位置作为后续context占位，QSL不压缩，用selector选择有效输出。因此字段名`num_actual_tokens`也可能描述“保留位置数”，业务有效域须沿读者追。不能把本阶段拒绝率和另一阶段padding率相加。','',
            '## 5. 固定配置下的设计量级，不冒充逐轮人口','',
            '以下先只计TP8对齐。target uniform宽度6；draft宽度5且两DP均衡。Target仍可能受实际graph档位/DP重派发影响。','',
            '|路径|请求B|逻辑行|TP8行域|补齐占行域|','|---|---:|---:|---:|---:|']
    for x in d['design_cases']:out.append(f'|{x["path"]}|{x["B"]}|{x["logical_rows"]}|{x["physical_rows"]}|{x["pad_pct"]:.2f}%|')
    out += ['', '**FULL桶须另算。** 冻结runner对FULL decode执行`LCM(query_len=6,TP=8)=24`捕获尺寸对齐。若本轮实际走24行图桶，B1/B2/B3/B4的scheduled行是6/12/18/24，图桶padding分别为75%/50%/25%/0%。这是有条件的行数计算，尚无本批次逐轮桶命中人口，不认作实测比例。','']
    out += ['', '## 6. 后续统一采集契约','',
            '每轮按target/draft、prefill/decode、real/dummy、DP和阶段记录：真实请求数、scheduled数量、DP统一数量、graph档位、stage-specific有效数量、tensor行域、slot/mask有效量、MoE真实收发量。有效域的定义必须同时记录，避免误用actual字段。','',
            d['capture_contract']['collection'],'',d['capture_contract']['aggregation'],'',
            '优先优化“确实执行了无效行、且占关键时间”的阶段。仅缩buffer shape但硬件tile不变，或算子已经按valid mask跳过，未必改善性能；新增compact/索引/通信成本必须扣除。','',
            '## 7. 源码与范围','']
    for x in d['source_evidence']['entries']:
        out += [f'- `{x["file"]}:{x["line"]}–{x["end_line"]}`；SHA256 `{x["sha256"]}`。']
    out += ['', '源片段完整保存在[source-evidence.json](source-evidence.json)。[机器账本](view-data.json) · [统计校验](validation.json)。本轮完全本机复用既有请求、原生记录和冻结源码；新增NPU/profile/模型请求/服务操作均为0，未重核live环境。全模型逐阶段有效行账尚未完成，不用历史17.29%填补当前缺口。','']
    (P/'REPORT.md').write_text('\n'.join(out))

if __name__=='__main__':main()
