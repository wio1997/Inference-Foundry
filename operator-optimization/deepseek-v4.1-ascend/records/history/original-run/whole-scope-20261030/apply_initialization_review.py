#!/usr/bin/env python3
"""Attach local semantic/exclusion contracts and persist an incomplete checkpoint."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shlex

W=Path(__file__).resolve().parent;R=W.parent
read=lambda p:json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
files=read(W/'coverage-ledger-v3/source-files.json');byfile={x['path']:x for x in files}
assert len(byfile)==len(files)==573
for f in files:assert hashlib.sha256(Path(f['local_path']).read_bytes()).hexdigest()==f['sha256']
summary=read(W/'coverage-ledger-v3/summary.json');assert summary['parse_errors']==[] and not summary['scope_closed']
counts={}
for kind in ['source-units','branch-sites','call-sites','import-sites']:
    before=read(W/('coverage-ledger-v2/'+kind+'.json'));after=read(W/('coverage-ledger-v3/'+kind+'.json'))
    ids={x['id'] for x in after};assert len(ids)==len(after) and {x['id'] for x in before}<=ids
    assert all(x['source_sha256']==byfile[x['source']]['sha256'] for x in after)
    counts[kind]=len(after)
sites=read(W/'coverage-ledger-v3/branch-sites.json')
specs=[
 ('vllm','transformers_utils/configs/deepseek_v41.py','DeepseekV41Config.__init__',35,'model_type不从text覆盖顶层类型；其它键继续检查属性。','冻结text.model_type与顶层不同，跳过该键有意义'),
 ('vllm','transformers_utils/configs/deepseek_v41.py','DeepseekV41Config.__init__',38,'property字段跳过直接setattr；其它text字段设置到root实例。','不得无条件认为全部text键均以同样方式写入'),
 ('vllm','transformers_utils/configs/deepseek_v41.py','DeepseekV41Config.__init__',43,'存在rope_scaling时在super之后恢复rope_parameters；否则保留base处理。','固定text有rope_scaling，普通初始化取true'),
 ('vllm','transformers_utils/configs/deepseek_v41.py','DeepseekV41Config.__init__',49,'缺expert_dtype且quant_cfg提供时补字段；否则不补。','固定quantization_config仅quant_method，条件false'),
 ('vllm','config/multimodal.py','MultiModalConfig.get_limit_per_prompt',483,'language_model_only时返回0；否则查显式limit。','冻结未开language_model_only，普通默认false'),
 ('vllm','config/multimodal.py','MultiModalConfig.get_limit_per_prompt',488,'未指定modality limit时返回999；否则返回显式count。','冻结image limit未指定，普通默认true'),
 ('vllm-ascend','models/deepseek_v41/vl_model.py','AscendDeepseekV41ForCausalLM.__init__',69,'vision层数>0时设置mm-prefix标志；false不设置。','冻结vision32层，构造与请求执行必须分开'),
 ('vllm-ascend','models/deepseek_v41/vl_model.py','AscendDeepseekV41ForCausalLM.__init__',82,'enabled时构造ViT/aligner/image向量；否则保留None。','普通固定配置推导true；实际驻留设备尚未测量'),
 ('vllm-ascend','models/deepseek_v41/vl_model.py','AscendDeepseekV41ForCausalLM._parse_and_validate_image_input',105,'patches缺失返回None；存在时组装image参数。','固定文本completions请求无patches，REQUEST_TEXT域推导true'),
 ('vllm-ascend','models/deepseek_v41/vl_model.py','AscendDeepseekV41ForCausalLM.embed_multimodal',181,'无image_input或无vision时返回空；否则执行图像处理。','固定REQUEST_TEXT域input=None，数学vision forward不执行；startup/dummy另审'),
 ('vllm-ascend','models/deepseek_v41/vl_model.py','AscendDeepseekV41ForCausalLM.embed_input_ids',203,'无multimodal embeds时仅返回language embedding；否则合并。','固定文本域采用前者，embedding其它分片条件不随之关闭'),
 ('vllm-ascend','worker/model_runner_v1.py','NPUModelRunner._get_eagle3_aux_layers_from_config',789,'上游显式eagle/dflash层配置存在则优先返回；否则进入DSpark扩展。','固定JSON缺少覆盖字段，普通配置取false'),
 ('vllm-ascend','worker/model_runner_v1.py','NPUModelRunner._get_eagle3_aux_layers_from_config',791,'DSpark时查询其层号；非DSpark不进入扩展。','固定method推导true'),
 ('vllm-ascend','worker/model_runner_v1.py','NPUModelRunner._get_eagle3_aux_layers_from_config',796,'dspark_target_layer_ids非空时+1返回；否则查legacy target_layer_ids。','固定[37,38,39]变[38,39,40]，无需default layer fallback'),
 ('vllm-ascend','worker/model_runner_v1.py','NPUModelRunner._get_eagle3_aux_layers_from_config',800,'legacy target_layer_ids非空时+1返回；否则返回None。','此前DSpark专有字段已命中，该fallback在固定正常路径不达'),
 ('vllm-ascend','models/deepseek_v41/model.py','DeepseekV41Model.forward',1560,'命中aux集合时先mean HC/按需gather再append；否则无该aux生产。','正常配置生产点为进入零基37/38/39层前，非层输出'),
 ('vllm-ascend','models/deepseek_v41/model.py','DeepseekV41Model.forward',1595,'buffer非None时可gather完整HC并copy；None时不保存完整HC。','use_eagle包括dspark，普通PP末组会构造；保存不等于后续需要数值'),
 ('vllm-ascend','models/deepseek_v41/model.py','DeepseekV41Model.forward',1603,'SP且无buffer时在collapse后gather；否则不走该gather。','条件重组可复用已有结构；不能无条件删除恢复token行的通信'),
 ('vllm-ascend','worker/model_runner_v1.py','NPUModelRunner.propose_draft_token_ids',2059,'mtp且buffer非None时替换hidden_states；否则不使用该getter返回值。','固定DSpark该条件false；其它模式/调试/反射消费者保持OPEN'),
 ('vllm','v1/sample/logits_processor/builtin.py','MinTokensLogitsProcessor.add_request',199,'min_tokens为0或已满足时返回None，不建禁用stop token状态；否则建立约束。','固定CompletionRequest.min_tokens默认0，普通配置推导true'),
]
reviews=[]
for repo,suffix,owner,line,effects,applicable in specs:
    matches=[x for x in sites if x['source'].startswith(repo+'/') and x['source'].endswith(suffix) and x['owner']==owner and x['line']==line and x['kind']=='If']
    assert len(matches)==1,(repo,suffix,owner,line,len(matches))
    s=matches[0]
    reviews.append({k:s[k] for k in ['id','source','source_sha256','line','owner','kind','condition']}|{
        'both_arm_effect_summary':effects,'applicability':applicable,'fact_kind':'DOCUMENTED_AND_CONDITIONAL_DERIVED',
        'runtime_directly_observed':False,'all_consumers_closed':False,'whole_module_closed':False,
        'evidence':'INITIALIZATION-AND-TAIL-REVIEW.md'})
assert len({x['id'] for x in reviews})==len(reviews)==20

# The exclusion proof is phase-specific. Decorators, constructors, loading and startup remain open.
vision_source='vllm-ascend/vllm_ascend/models/deepseek_v41/vision.py'
vision_units=[x for x in read(W/'coverage-ledger-v3/source-units.json') if x['source']==vision_source and x['kind']=='function' and x['name']!='__init__']
excluded_ids=[x['id'] for x in sites if x['source']==vision_source and any(u['line']<=x['line']<=u['end_line'] for u in vision_units)]
phase_exclusion={'id':'M05-VISION-REQUEST-TEXT','phase':'FIXED_TEXT_REQUEST_FORWARD_ONLY',
    'premises':['saved completions body has text prompt and no patches/media payload','wrapper patches=None→image_input=None→no vision forward'],
    'excluded_function_ids':[x['id'] for x in vision_units],'excluded_branch_ids':excluded_ids,
    'not_excluded':['decorator/module initialization','constructors and weight loading/residency','startup dummy/profile/warmup','other image requests and broader service capability'],
    'runtime_directly_observed':False,'whole_module_closed':False}
save(W/'initialization-branch-review-delta.json',{'reviews':reviews,'phase_specific_exclusions':[phase_exclusion],
    'overall_complete':False,'source_ledger':'coverage-ledger-v3'})

evidence=read(W/'initialization-tail-evidence.json')
assert evidence['mtp_buffer']['max_capacity_mib_if_bf16']==320 and evidence['symbol_search_match_count']==90
assert all(len(x['per_rank'])==16 and all(r['count']==12 for r in x['per_rank']) for x in evidence['same_capture_families'])
external=read(W/'installed-transformers-contract.json');assert external['version']=='5.14.1'
for path,item in external['files'].items():
    b=(W/'installed-transformers-source'/path).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
model=read(R/'evidence/model-config.json')
generation_fields={'repetition_penalty','temperature','top_k','top_p','min_p','max_new_tokens'}
assert not generation_fields & model.keys() and not generation_fields & model['text_config'].keys()
assert not {'eagle_aux_hidden_state_layer_ids','eagle_config','dflash_config'} & model.keys()
assert not {'eagle_aux_hidden_state_layer_ids','eagle_config','dflash_config'} & model['text_config'].keys()
serve=R/'evidence/frozen-serve.sh';text=serve.read_text();tokens=shlex.split(next(x for x in text.splitlines() if x.startswith('exec ')))
assert 'export VLLM_USE_V2_MODEL_RUNNER=0' in text
assert not {'--language-model-only','--limit-mm-per-prompt','--generation-config','--override-generation-config'} & set(tokens)
derivation={'status':'CONDITIONAL_FROZEN_INITIALIZATION_NOT_LIVE_RESOLVED_OBJECT_PROOF',
    'source_ledger':'coverage-ledger-v3','current_installed_transformers':external['version'],
    'checkpoint_serialization_transformers_version':model['transformers_version'],
    'aux_layer_mapping':{'checkpoint_zero_based_ids':[37,38,39],'interface_one_based_ids':[38,39,40],'producer':'HC mean entering zero-based layers37/38/39'},
    'generation_fallback':'no six-whitelist field in saved model config; source fallback from_model_config/to_diff_dict→CompletionRequest defaults under normal frozen initialization',
    'vision_phases':{'ordinary_construction':'DERIVED_ENABLED','fixed_text_request_math_forward':'DERIVED_INAPPLICABLE','startup_profile':'OPEN','actual_resident_device_bytes':'UNKNOWN'},
    'mtp_buffer':'DSpark includes use_eagle and allocates; V1 getter value numerically used only under method=mtp in reviewed caller; all consumers/observers remain open',
    'configuration_sha256':hashlib.sha256((R/'evidence/model-config.json').read_bytes()).hexdigest(),
    'serve_sha256':hashlib.sha256(serve.read_bytes()).hexdigest(),'overall_complete':False}
save(W/'initialization-applicability.json',derivation)

backup=W/'before-initialization-review-records';backup.mkdir(exist_ok=False)
for name in ['STATE.json','PROGRESS.md','COVERAGE-LEDGER.md','module-coverage-ledger.json']:
    (backup/name).write_bytes((W/name).read_bytes())
module=read(W/'module-coverage-ledger.json')
for m in module['surfaces']:
    extra=[]
    if m['id']=='M00':extra=['vllm/vllm/transformers_utils/config.py','vllm/vllm/transformers_utils/configs/__init__.py','vllm/vllm/transformers_utils/configs/deepseek_v41.py']
    if m['id']=='M01':extra=[x['path'] for x in files if '/entrypoints/generate/base/' in x['path'] or x['path'].endswith('/entrypoints/serve/engine/protocol.py')]
    if m['id']=='M05':extra=['vllm/vllm/config/multimodal.py','vllm/vllm/transformers_utils/configs/deepseek_v41.py']
    selected=set(m['indexed_source_paths'])|set(extra)
    m['indexed_source_paths']=sorted(selected);m['indexed_branch_ids']=[x['id'] for x in sites if x['source'] in selected]
    if m['id'] in ('M00','M01','M05','M15','M18'):
        m.setdefault('local_review_evidence',[]).extend(['INITIALIZATION-AND-TAIL-REVIEW.md','initialization-applicability.json','initialization-branch-review-delta.json','initialization-tail-evidence.json'])
        m['mandatory_gates']['scope_and_dispatch']['status']='PARTIAL_INITIALIZATION_AND_PHASE_DERIVATION; GLOBAL_OPEN'
    if m['id']=='M05':m['phase_specific_exclusions']=[phase_exclusion['id']]
    if m['id']=='M15':
        m['mandatory_gates']['value_and_all_consumers']['status']='PARTIAL_V1_DSPARK_MTP_BUFFER_LIVENESS; GLOBAL_OBSERVERS_OPEN'
        m['mandatory_gates']['resource_performance_and_bounds']['status']='PARTIAL_SAME_CAPTURE_HC_FAMILIES_AND_LOGICAL_BYTES; RUNTIME_EXPOSURE_OPEN'
    assert not m['complete']
for slot in module['rule_coverage']:
    if slot['surface'] in ('M05','M15','M18') and slot['rule'] in ('R03','R04','R05','R06','R07','R09'):
        slot['status']='CONDITIONAL_LOCAL_INITIALIZATION_TAIL_REVIEW; OTHER_BOUNDARIES_OPEN'
        slot.setdefault('evidence',[]).append('INITIALIZATION-AND-TAIL-REVIEW.md')
module['source_ledger']='coverage-ledger-v3'
module.setdefault('review_delta',[]).append('initialization-branch-review-delta.json')
assert not module['overall_complete'] and not module['goal_scope_closed']
save(W/'module-coverage-ledger.json',module)

state=read(W/'STATE.json');now=datetime.datetime.now(datetime.timezone.utc).isoformat()
state.update(updated_at_utc=now,source_ledger='coverage-ledger-v3/summary.json',source_files_indexed=573,
    selected_local_branch_contracts=43,initialization_review='INITIALIZATION-AND-TAIL-REVIEW.md',
    previous_goal_turn_classification='PROGRESS:19 versioned state edges,23 local contracts and real checkpoint outputs inspected before continuing',
    current_turn_progress='7 source dependencies,2 installed library sources,exact-symbol audit,20 local branches,phase-specific exclusion,2 conditional directions,16rank HC families',
    next_action='继续完整运行配置/平台替换及全消费者闭包；优先补MoE同调用有效专家行/容量、Indexer实际长度/mode和请求关键路径。对vision驻留、HC缓冲/gather仅报告条件账与相容测量；不得据报告成稿关闭整体。')
state['candidate_local_reviews'].extend(['WS-VISION-RESIDENT-01','WS-TAIL-STATE-01 (F26 refinement)'])
state['conditional_derivations'].extend(['vision ordinary construction enabled but fixed-text request forward excluded; startup/residency separate','aux entering layers37/38/39; generation fallback uses installed5.14.1 source, not checkpoint version metadata','MTP buffer local V1 DSpark numeric liveness contract; strict getter-observer domain retained'])
assert state['goal_status']=='active' and not state['overall_complete'] and state['whole_modules_closed']==0
save(W/'STATE.json',state)

lines=['# 整体优化机会分析进度','','**整体未完成，目标active；26个暂定模块中0个通过全部覆盖门槛。** 不提供缺少依据的完成百分比。报告成稿、索引规模和检查点PASS均不代表整体验收。','',
    '当前已落地：','',
    '- 573份固定commit源码的版本化站点索引；另存当前安装Transformers5.14.1的2个配置源。原558/566文件索引保留。',
    '- 26个暂定pipeline审阅面、260个变换规则槽与每模块8类门槛；范围仍扩展，不设四节点完成上限。',
    '- 43个选定分支的局部效果合同、19条局部版本化状态边；固定文本请求的vision forward有阶段限定排除依据，构造/驻留/启动profile继续保留。',
    '- 同次16rank CSV中8类词表族及4类完整HC兼容族，原SHA、dtype、次数/直方图与算术已核对。形状相容不能替代tensor/事件身份。',
    '- 补固定请求/默认回退、辅助层编号、Markov/拒绝及跨迭代消费者；纠正DSpark导入误读与checkpoint元数据版本不能代表安装库版本。',
    '- 新审阅vision驻留特化与DSpark下MTP缓冲/尾部gather的条件重组，含参数/接口逻辑字节账。所有候选仍有条件与重叠，未实现、未测模型收益。','',
    '决定性缺口仍包括：运行配置/动态替换/全部消费者与primitive闭包；全部相关分支、alias和跨流/卡/层/迭代状态；完整请求因果与关键路径；MoE有效专家行/容量、Indexer实际长度/mode、Engram host/UVA成本；分层访存与适用资源上限。','',
    '[最新初始化/尾部审阅](INITIALIZATION-AND-TAIL-REVIEW.md)；[输出头](TAIL-HEAD-SAMPLER-REVIEW.md)；[DSpark](DSPARK-CONTRACT-REVIEW.md)；[跨迭代状态](CROSS-ITERATION-REVIEW.md)；[覆盖台账](COVERAGE-LEDGER.md)；[STATE](STATE.json)。','',
    '本goal仅只读取证与离线分析，未修改目标、重启服务或启动新的推理/NPU负载。前一turn已核实为实质进展，本turn继续产出新证据；无外部阻塞，整体保持active。']
(W/'PROGRESS.md').write_text('\n'.join(lines)+'\n')
coverage=W/'COVERAGE-LEDGER.md';text=coverage.read_text()
text+='\n最新源码底账为[573文件v3](coverage-ledger-v3/summary.json)，累计43个选定局部分支合同；新增[初始化delta](initialization-branch-review-delta.json)与[阶段适用依据](initialization-applicability.json)。旧表关联数是先前索引快照，最新数量以机器台账为准；所有整模块门槛仍未关闭。\n'
coverage.write_text(text)

outer=read(R/'STATE.json');outer['last_update']=now;outer['next_read_only_action']=state['next_action']
outer['whole_scope_current']['latest_semantic_review']='whole-scope-20261030/INITIALIZATION-AND-TAIL-REVIEW.md'
save(R/'STATE.json',outer)
catalog=read(R/'RESEARCH-CATALOG.json');existing={x['relative_path']:x for x in catalog['indexed_files']}
changed=[R/'STATE.json']+[W/x for x in ['STATE.json','PROGRESS.md','COVERAGE-LEDGER.md','module-coverage-ledger.json','INITIALIZATION-AND-TAIL-REVIEW.md','initialization-applicability.json','initialization-branch-review-delta.json','initialization-tail-evidence.json','coverage-ledger-v3/summary.json']]
for path in changed:
    relative=str(path.relative_to(R));b=path.read_bytes();item=existing.get(relative,{})
    item.update(relative_path=relative,local_path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),current_local_record=True,
        server_scope='Updated local checkpoint; no claim of remote/public synchronization.')
    item.pop('server_mirror_path',None);item.pop('stage_mirror_path',None);existing[relative]=item
catalog['indexed_files']=list(existing.values());catalog['continuation_updated_at_utc']=now
save(R/'RESEARCH-CATALOG.json',catalog)
result={'status':'PASS_LOCAL_INITIALIZATION_REFERENCE_CHECKPOINT_ONLY','source_files':573,'unique_source_sites':counts,
    'new_selected_branch_contracts':20,'total_selected_branch_contracts':43,'phase_excluded_vision_branch_sites':len(excluded_ids),
    'local_state_edges_unchanged':19,'whole_modules_closed':0,'overall_goal_complete':False}
save(W/'initialization-checkpoint-verification.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
