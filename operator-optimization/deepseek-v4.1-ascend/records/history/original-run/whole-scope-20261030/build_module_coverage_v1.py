#!/usr/bin/env python3
"""Create explicit whole-pipeline review surfaces without closing unknowns."""
import fnmatch
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
ledger=ROOT/'coverage-ledger'
files=json.loads((ledger/'source-files.json').read_text())
units=json.loads((ledger/'source-units.json').read_text())
branches=json.loads((ledger/'branch-sites.json').read_text())

# Matching a source file is a discovery index only. Shared files intentionally
# associate with multiple surfaces; counts must not be summed as coverage.
definitions=[
('M00','注册、平台替换、模型与算子选择',['*/models/__init__.py','*/platform.py','*/patch/*','*/model_executor/custom_op.py','*/utils.py'],[],['真实模型/LogitsProcessor/LMHead类及权重分片模式','vllm.models与legacy DSpark导入重定向','已注册与实际使用的算子/硬件能力']),
('M01','请求、tokenization与采样契约',['*/entrypoints/openai/completion/*'],[],['冻结请求到SamplingParams/metadata的逐字段映射','server/model默认值和条件processor']),
('M02','前缀复用、KV分配与回收',['*/v1/core/kv_cache*','*/v1/core/block_pool.py','*/v1/core/single_type_kv_cache_manager.py'],[],['cache命中后真实计算域','块共享/回收/跨步所有权与hash状态']),
('M03','EngineCore、调度与异步队列',['*/v1/core/sched/*','*/v1/engine/core.py'],[],['提交/回复/准备的实际因果边','队列容量、同次batch身份和阻塞点']),
('M04','worker输入准备与持久batch状态',['*/worker/model_runner_v1.py','*/v1/worker/gpu_input_batch.py'],['F33'],['6956个scope的具体生产者/消费者','上一批读取完成与下一批覆盖','未隐藏准备与等待的拆解']),
('M05','embedding、模型wrapper和图像条件',['*/models/deepseek_v41/vl_model.py','*/models/deepseek_v41/vision.py','*/ops/vocab_parallel_embedding.py'],['F02','F28'],['文本请求的vision执行/驻留条件','embedding分片及token行域','wrapper到language模型的完整值链']),
('M06','mHC初始化、pre/post与跨层残差',['*/models/deepseek_v41/model.py','*/device/device_op.py'],['F01','F02','F03','F04'],['系数/残差的全部消费者与alias','精度、归约与跨层状态依赖','内部复合核的primitive/effect展开']),
('M07','Q/KV投影、norm、量化与RoPE',['*/attention/dsa_v41.py','*/attention/context_parallel/dsa_v41_cp.py'],['F05','F06','F07','F08','F32'],['query/cache行域与权重/scale所有权','所有indexer分叉与持久cache消费者','流/事件/原位更新身份']),
('M08','compressor、ring状态与latent分叉',['*/models/deepseek_v41/compressor.py','*/attention/dsa_v41.py','*/ops/triton/compressor/*'],['F17','F18','F19'],['跨迭代ring写读与C2有效性','两个latent消费者及BF16舍入','不同压缩比/空行/边界分支']),
('M09','Indexer评分、候选与跨层索引',['*/models/deepseek_v41/indexer.py','*/attention/dsa_v41.py'],['F09','F10','F11','F12'],['真实key长度/候选覆盖率/mode','精确TopK、tie、sentinel与回退','层20到后续所有索引消费者']),
('M10','SWA、压缩KV与稀疏attention',['*/attention/dsa_v41.py','*/models/deepseek_v41/mixed_quant_attention.py','*/ops/dsa.py'],['F13'],['所有cache平面和mask/sink','A3与packed等条件分支的适用证据','kernel内部及跨流完整资源账']),
('M11','attention输出、head恢复、两级投影与归约',['*/attention/context_parallel/dsa_cp.py','*/attention/context_parallel/dsa_v41_cp.py'],['F14','F15','F16'],['行/head/token恢复与padding','块ready是否可发布及collective顺序','两级投影精度与全体消费者']),
('M12','MoE路由、dispatch、专家计算和combine',['*/ops/fused_moe/*','*/layers/fused_moe/*','*/mc2/dispatch_ffn_combine/*'],['F20','F21'],['同调用专家行数/容量/mask','内部组级ready与共享workspace','EP组、二进制身份及通信暴露']),
('M13','共享专家、路由分支汇合与scaling',['*/models/deepseek_v41/model.py','*/ops/fused_moe/*'],['F22','F23'],['shared/routed的量化域和输入版本','归约域、合并舍入与资源竞争','fused MC2选路与双分支并行条件']),
('M14','Engram hash、host/UVA lookup、投影与门控',['*/models/deepseek_v41/engram/*'],['F24','F25'],['完整host/UVA访问与ready链','跨rank表布局、scale和无效ID','投影/gate/残差与跨步hash状态']),
('M15','最终mHC collapse、norm及MTP辅助状态',['*/models/deepseek_v41/model.py'],['F26'],['最终hidden与aux/MTP缓冲的全部消费者','归一化与舍入、SP/PP恢复','跨迭代覆盖与图地址生命周期']),
('M16','目标/草稿词表投影与跨卡重建',['*/layers/logits_processor.py','*/layers/vocab_parallel_embedding.py','*/ops/vocab_parallel_embedding.py'],['F26','F30','WS-VOCAB-01'],['STANDARD/FINE_GRAINED/PCP_X_TP真实分支','全部logits消费者与处理顺序','局部选择的全局tie、padding和ID映射']),
('M17','目标采样、拒绝/接受、bonus与logprobs',['*/sample/*','*/ops/triton/reject_sample.py'],['F30','WS-REJECT-01'],['全部greedy/random/synthetic/block/entropy路径','bonus与target原值副本的条件消费者','实际metadata、RNG、输出token/回退状态']),
('M18','DSpark上下文投影、共享KV及输入发布',['*/models/deepseek_v41/dspark.py','*/spec_decode/dspark_proposer.py'],['F27','F29'],['context与query独立行域/分片','历史KV与当前请求身份','publish/overwrite事件和跨卡信息']),
('M19','DSpark decoder、mHC及共享计算',['*/models/deepseek_v41/dspark.py'],['F27','F28'],['三个draft block身份与配置fallback','与目标层复用的所有语义差异','跨block隐藏状态/残差和buffer']),
('M20','DSpark base logits、Markov和confidence',['*/models/deepseek_v41/dspark.py','*/spec_decode/llm_base_proposer.py'],['F30','WS-VOCAB-01'],['seed与逐位置Markov依赖','raw_logits的所有原位/概率消费者','词表分块与跨位置可交错的边界']),
('M21','draft/target metadata、slot与索引准备',['*/attention/dsa_v41.py','*/attention/context_parallel/dsa_v41_cp.py','*/spec_decode/dspark_proposer.py'],['F31','F33'],['global/local请求长度与设备权威状态','持久地址的全体消费者','target/draft、空行、padding与skip-update']),
('M22','graph捕获、replay、更新与多流生命周期',['*/compilation/acl_graph.py','*/worker/model_runner_v1.py'],[],['每次graph/replay到任务/请求映射','静态地址、更新handle与event身份','跨流收敛、异常和回退条件']),
('M23','DP/TP/EP/PCP通信与分区所有权',['*/distributed/*','*/models/common/ops/*','*/ops/vocab_parallel_embedding.py'],['F15','F16','F21','F32','WS-VOCAB-01'],['参与组/序号/真实传输量','全流全卡暴露关键路径','重排/分块后的顺序与共享资源']),
('M24','跨迭代状态、接受回滚、预取和重用',['*/worker/model_runner_v1.py','*/spec_decode/*','*/models/deepseek_v41/engram/*','*/v1/core/*'],[],['NEXT_INVOCATION依赖与request重排','KV/概率/hash/RNG的版本和生存期','可提前的准备与必须等待的状态']),
('M25','采样结果回传、响应、流式输出与终止',['*/v1/engine/output_processor.py','*/entrypoints/openai/completion/serving.py'],[],['设备结果到客户端chunk/输出token归一化','stop/EOS/ignore_eos和1K长度','主机队列、异步D2H与输出消费者'])]

gates=['scope_and_dispatch','all_relevant_branches','value_and_all_consumers','alias_and_state_lifetime',
       'cross_stream_rank_iteration_dependencies','fusion_split_regroup_rules','resource_performance_and_bounds','evidence_and_literature_reanalysis']
surfaces=[]
for ident,name,patterns,candidates,gaps in definitions:
    matched={x['path'] for x in files if any(fnmatch.fnmatch(x['path'],p) for p in patterns)}
    us=[x['id'] for x in units if x['source'] in matched]
    bs=[x['id'] for x in branches if x['source'] in matched]
    surfaces.append({'id':ident,'name':name,'source_selectors':patterns,'indexed_source_paths':sorted(matched),
      'indexed_unit_ids':us,'indexed_branch_ids':bs,'associations_are_many_to_many':True,
      'candidate_crosswalk':candidates,'status':'OPEN_NEW_REVIEW' if ident in ('M00','M16','M17') else 'OPEN_EXISTING_EVIDENCE_NEEDS_FULL_CROSSWALK',
      'mandatory_gates':{g:{'status':'OPEN','evidence':[]} for g in gates},
      'conclusion_changing_gaps':gaps,'complete':False})

rules=[('R01','纵向两算子/三算子/更长链融合，不设四节点完成上限'),('R02','共享输入/生产者的横向合并与批处理'),
 ('R03','布局、转换、缩放/投影的代数重排'),('R04','复合算子的语义分解与重新归组'),
 ('R05','分块ready交付与生产消费流水'),('R06','跨卡分片、计算与通信所有权重组'),
 ('R07','跨层持久状态、重复工作与重算/复用'),('R08','跨迭代准备、预取、状态发布与回滚'),
 ('R09','保持精度/排序/RNG语义的算法替换或特化'),('R10','主机/设备、队列、图更新与提交边界重组')]
rule_coverage=[{'surface':x['id'],'rule':r,'description':name,'status':'OPEN','evidence':[],'exclusion_requires_proof':True} for x in surfaces for r,name in rules]
out={'goal_scope_closed':False,'overall_complete':False,'surface_universe':'PROVISIONAL_EXPANDING; source import/call and primitive closure remains open',
 'no_overall_completion_percentage':True,'surfaces':surfaces,'rule_coverage':rule_coverage,
 'limits':['Source associations overlap; counts cannot be summed as coverage.','Every semantic gate starts OPEN; source indexing does not satisfy it.','Other-model helpers in source snapshots are not declared executed or applicable.','No prior completed report or bounded graph enumeration closes any gate.']}
(ROOT/'module-coverage-ledger.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
lines=['# 整体覆盖台账入口','', '**整体未完成；范围闭包仍在扩展。** 文件/条件条目是定位索引，不是已审阅或运行覆盖。','',
 '| 模块 | 已关联源码数 | 分支站点关联数（可重复） | 结论性缺口 |','|---|---:|---:|---|']
for x in surfaces:lines.append(f"| {x['id']} {x['name']} | {len(x['indexed_source_paths'])} | {len(x['indexed_branch_ids'])} | {'；'.join(x['conclusion_changing_gaps'])} |")
lines.extend(['','每个模块必须逐项关闭范围/dispatch、分支、值与全部消费者、alias/状态、跨流/卡/迭代依赖、变换规则、资源/性能/极限与文献复析八类门槛。',
 '[机器可读台账](module-coverage-ledger.json)包含站点ID与260个规则检查槽；它们目前均未关闭。',
 '[源码底账](coverage-ledger/summary.json)覆盖选取的冻结源，不宣称完整传递依赖闭包。'])
(ROOT/'COVERAGE-LEDGER.md').write_text('\n'.join(lines)+'\n')

# Compact measured family evidence from the same capture, separate from source claims.
data=json.loads((ROOT/'head-sampler-exports.json').read_text())
selected=[('target_head_compatible','MatMulV2','"24,5120;16160,5120"'),
 ('draft_base_head_compatible','MatMulV2','"20,5120;16160,5120"'),
 ('markov_head_compatible','MatMulV2','"4,256;129280,256"'),
 ('target_row_gather_compatible','Index','"24,129280;1;2;20"'),
 ('target_fp32_copy_compatible','TensorMove','"20,129280"'),
 ('bonus_argmax_compatible','ArgMaxV2','"4,129280;"'),
 ('target_argmax_compatible','ArgMaxV2','"20,129280;"'),
 ('greedy_rejection','rejection_greedy_sample_triton','"4;20;20;4,1"')]
families=[]
for label,typ,shape in selected:
    rr=[]
    for source in data['rows']:
        match=[g for g in source['groups'] if g['type']==typ and g['input_shapes']==shape and (typ!='ArgMaxV2' or g['input_dtypes']=='FLOAT;INT64')]
        assert len(match)==1,(label,source['rank'],len(match))
        g=match[0];rr.append({'rank':source['rank'],'count':g['count'],'median_ns':g['median_ns'],'sum_ns':g['sum_ns'],'csv_sha256':source['csv_sha256']})
    families.append({'label':label,'type':typ,'input_shapes':shape,'per_rank':rr,'rank_count':len(rr),
                     'rank_median_duration_ns':statistics.median(x['median_ns'] for x in rr),
                     'count_per_rank_values':sorted({x['count'] for x in rr}),'binding':'SHAPE_AND_DTYPE_COMPATIBLE_NOT_TENSOR_IDENTITY'})
(ROOT/'head-sampler-family-summary.json').write_text(json.dumps({'scope':'official later window all16rank; per-rank medians then rank median','families':families,'no_critical_path_or_gain_claim':True},indent=2)+'\n')
print(json.dumps({'module_surfaces':len(surfaces),'rule_slots':len(rule_coverage),'all_surfaces_complete':False,
 'head_sampler_families':[{'name':x['label'],'median_us':x['rank_median_duration_ns']/1000,'counts_per_rank':x['count_per_rank_values']} for x in families]},ensure_ascii=False,indent=2))
