#!/usr/bin/env python3
"""Persist reviewed conditional contracts, identities and explicit open gates."""
import ast
import fnmatch
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parent
read = lambda relative: json.loads((ROOT/relative).read_text())
def save(relative, data):
    (ROOT/relative).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')

files = read('coverage-ledger-v2/source-files.json')
byfile = {x['path']: x for x in files}
assert len(files) == len(byfile)
for x in files:
    assert hashlib.sha256(Path(x['local_path']).read_bytes()).hexdigest() == x['sha256']
counts = {}
for kind in ('source-units','branch-sites','call-sites','import-sites'):
    old = read('coverage-ledger/'+kind+'.json')
    new = read('coverage-ledger-v2/'+kind+'.json')
    old_ids = {x['id'] for x in old}
    new_ids = {x['id'] for x in new}
    assert len(new_ids) == len(new) and old_ids <= new_ids
    assert all(x['source_sha256'] == byfile[x['source']]['sha256'] for x in new)
    counts[kind] = len(new)
assert read('coverage-ledger-v2/summary.json')['parse_errors'] == []

def source(suffix):
    found = [x for x in files if x['path'].endswith(suffix)]
    assert len(found) == 1, suffix
    return found[0]
def class_default(suffix, class_name, field):
    meta = source(suffix)
    tree = ast.parse(Path(meta['local_path']).read_text())
    cls = next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name == class_name)
    node = next(x for x in cls.body if isinstance(x,ast.AnnAssign) and isinstance(x.target,ast.Name) and x.target.id == field)
    value = node.value
    if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id == 'Field':
        value = next(x.value for x in value.keywords if x.arg == 'default')
    return {'value':ast.literal_eval(value),'source':meta['path'],'source_sha256':meta['sha256'],'line':node.lineno}

serve = ROOT.parent/'evidence/frozen-serve.sh'
tokens = shlex.split(next(line for line in serve.read_text().splitlines() if line.startswith('exec ')))
flag = lambda name: tokens[tokens.index(name)+1]
additional = json.loads(flag('--additional-config'))
speculative = json.loads(flag('--speculative-config'))
assert flag('--tensor-parallel-size') == '8' and flag('--data-parallel-size') == '2'
assert not {'--prefill-context-parallel-size','--decode-context-parallel-size'} & set(tokens)
assert not {'finegrained_tp_config','rejection_sampler_config','dynamic_spec_config'} & additional.keys()
assert speculative == {'method':'dspark','num_speculative_tokens':5,'enforce_eager':True}

client = ROOT.parent/'diagnostic-20261029/official_profile_client.py'
tree = ast.parse(client.read_text())
function = next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name == 'run_request')
assignment = next(x for x in function.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in x.targets))
keys = [ast.literal_eval(x) for x in assignment.value.keys]
literal_fields = {}
for key,value in zip(keys,assignment.value.values):
    try: literal_fields[key] = ast.literal_eval(value)
    except (ValueError,TypeError): pass
assert literal_fields['temperature'] == 0 and literal_fields['ignore_eos'] is True
assert not {'logprobs','prompt_logprobs','logprob_token_ids'} & set(keys)
defaults = {}
for name,suffix,cls,field in [
    ('lmhead_finegrained_size','vllm_ascend/ascend_config.py','FinegrainedTPConfig','lmhead_tensor_parallel_size'),
    ('entropy','vllm_ascend/ascend_config.py','RejectionSamplerConfig','enable_entropy_verify'),
    ('block','vllm_ascend/ascend_config.py','RejectionSamplerConfig','enable_block_verify'),
    ('dynamic_spec_method','vllm_ascend/ascend_config.py','DynamicSpecConfig','method'),
    ('draft_sample_method','vllm/config/speculative.py','SpeculativeConfig','draft_sample_method'),
    ('pcp_size','vllm/config/parallel.py','ParallelConfig','prefill_context_parallel_size'),
    ('dcp_size','vllm/config/parallel.py','ParallelConfig','decode_context_parallel_size'),
]: defaults[name] = class_default(suffix,cls,field)
assert {k:v['value'] for k,v in defaults.items()} == {'lmhead_finegrained_size':0,'entropy':False,'block':False,'dynamic_spec_method':None,'draft_sample_method':'greedy','pcp_size':1,'dcp_size':1}

config_file = ROOT.parent/'evidence/model-config.json'
config = json.loads(config_file.read_text())['text_config']
model_fields = {key:config[key] for key in ['num_hidden_layers','n_routed_experts','num_experts_per_tok','num_nextn_predict_layers','dspark_block_size','dspark_target_layer_ids','dspark_markov_rank','dspark_n_routed_experts','dspark_num_experts_per_tok']}
metadata = read('fixed-contract-metadata.json')
build = metadata['files']['/vllm-workspace/vllm-ascend/vllm_ascend/_build_info.py']
assert "__device_type__ = 'A3'" in build['text']
evidence = {'status':'DERIVED_CONFIGURATION_CONTRACT_NOT_DIRECT_LOADED_OBJECT_PROOF',
    'source_ledger':'coverage-ledger-v2/summary.json','defaults':defaults,
    'serve':{'sha256':hashlib.sha256(serve.read_bytes()).hexdigest(),'additional_config':additional,'speculative_config':speculative,'tp':8,'dp':2},
    'client':{'sha256':hashlib.sha256(client.read_bytes()).hexdigest(),'literal_fields':literal_fields,'omitted_logprob_fields':True},
    'model_config':{'sha256':hashlib.sha256(config_file.read_bytes()).hexdigest(),'fields':model_fields},
    'current_build_metadata':{'value':'A3','sha256':build['sha256'],'historical_loaded_object_directly_observed':False},
    'conclusions':{'standard_target_draft_base_vocab_plan':'DERIVED_UNDER_NORMAL_FROZEN_INITIALIZATION',
        'replicated_markov_plan':'DERIVED_FROM_DISABLE_TP_TRUE_FIRST_BRANCH',
        'target_greedy_and_no_requested_logprobs':'DERIVED_FOR_SAVED_CLIENT_REQUESTS',
        'entropy_block_confidence_probability_paths':'DERIVED_DISABLED_WITH_FROZEN_DEFAULTS',
        'a5_packed_cache_execution':'DERIVED_INAPPLICABLE_FOR_A3_STANDARD_ADAPTOR'},
    'open':['historical resolved objects, mutation/patch closure and same-call IDs','complete processor and frontend default/fallback chain','missing generation_config.json does not prove neutral defaults','full state/alias/event/collective consumers','primitive internal work and critical path'],
    'overall_complete':False}
save('contract-applicability.json',evidence)

sites = read('coverage-ledger-v2/branch-sites.json')
specs = [
    ('ops/vocab_parallel_embedding.py','_resolve_vocab_parallel_plan',100,'If','disable_tp','true优先REPLICATED；false进入PCP/fine-grained/standard。','Markov true；普通head false，均为构造合同推导'),
    ('ops/vocab_parallel_embedding.py','_resolve_vocab_parallel_plan',111,'If','use_pcp_sharding','true使用TP×PCP所有权；false继续fine/standard。','固定PCP=1推导false'),
    ('ops/vocab_parallel_embedding.py','_resolve_vocab_parallel_plan',135,'If','is_lm_head and lmhead_tp_enable()','true独立LMHead-TP；false继续embedding或standard。','固定fine-grained大小0推导false'),
    ('spec_decode/dspark_proposer.py','AscendDSparkProposer.__init__',40,'If','self.sample_from_anchor','true每请求K行，false K+1行；最大buffer和位置容量随之改变。','配置getattr默认true；真实归一化字段与padding仍需闭合'),
    ('spec_decode/dspark_proposer.py','AscendDSparkProposer.__init__',64,'If',"dynamic_spec_config.method == 'dspark'",'true构造动态scheduler，false保留None。','固定默认None推导false'),
    ('spec_decode/llm_base_proposer.py','AscendSpecDecodeBaseProposer._run_merged_draft',1550,'If',"self.method == 'dspark'",'true基础logits后运行Markov循环；false普通逐token drafter路径。','固定method推导true，仍保留外层selector分派待证'),
    ('spec_decode/llm_base_proposer.py','AscendSpecDecodeBaseProposer._run_merged_draft',1586,'For','idx in range(self.num_speculative_tokens)','每个位置读取上个token、写下一token；零次不进入，异常中止；K=5正常完成5次。','直接实现族的跨位置依赖，不当作所有算法串行下界'),
    ('spec_decode/llm_base_proposer.py','AscendSpecDecodeBaseProposer._run_merged_draft',1590,'If','use_probabilistic','true采样并保留概率；false argmax并可能映射ID。','固定draft_sample_method greedy推导false'),
    ('spec_decode/llm_base_proposer.py','AscendSpecDecodeBaseProposer._run_merged_draft',1600,'If','dspark_has_vocab_mapping','true把draft ID映到target ID供返回/下步embedding；false保持ID。','实际mapping对象未直接观测，OPEN'),
    ('spec_decode/llm_base_proposer.py','AscendSpecDecodeBaseProposer._run_merged_draft',1603,'If','use_probabilistic and dspark_probs_list','true stack/contiguous全概率；false不发布概率tensor。','greedy合同推导false，不能将该未执行物化计为节省'),
    ('spec_decode/llm_base_proposer.py','AscendSpecDecodeBaseProposer._run_merged_draft',1611,'If','self.dynamic_spec is not None','true执行confidence动态更新；false跳过。','固定动态开关推导false；构建head不等于执行'),
    ('spec_decode/llm_base_proposer.py','AscendSpecDecodeBaseProposer._run_merged_draft',1637,'If','self.num_speculative_tokens == 1 or self.parallel_drafting','true返回整块tokens；false继续多次decoder。','dspark parallel_drafting推导true；不把后续K-1 decoder算入该域'),
    ('models/deepseek_v41/dspark.py','DeepseekV41DSparkModel.precompute_and_store_context_kv',301,'If','context_states.numel() == 0 or context_slot_mapping is None','true直接返回；false逐层投影/写context。','空context和无slot路径有意义，运行行域OPEN'),
    ('models/deepseek_v41/dspark.py','DeepseekV41DSparkModel.precompute_and_store_context_kv',305,'If','context_positions.numel() == 0','true不再写context；false使用本层wkv/norm/RoPE。','不同层权重/缓存独立，shared命名不证明值复用'),
]
reviews = []
for suffix,owner,line,kind,condition,arms,binding in specs:
    found = [x for x in sites if x['source'].startswith('vllm-ascend/') and x['source'].endswith(suffix) and x['owner']==owner and x['line']==line and x['kind']==kind]
    assert len(found)==1 and found[0]['condition']==condition,(suffix,line,condition,found)
    site = found[0]
    reviews.append({key:site[key] for key in ('id','source','source_sha256','line','owner','kind','condition')} | {
        'both_arm_or_loop_effect_summary':arms,'applicability':binding,
        'fact_kind':'DOCUMENTED_LOCAL_EFFECTS_AND_CONDITIONAL_DERIVATION','runtime_directly_observed':False,
        'all_consumers_closed':False,'whole_module_closed':False,'evidence':'DSPARK-CONTRACT-REVIEW.md'})
save('dspark-branch-review-delta.json',{'reviews':reviews,'overall_complete':False,'source_ledger':'coverage-ledger-v2'})

path = ROOT/'module-coverage-ledger.json'
backup = ROOT/'module-coverage-ledger-before-contract-review.json'
assert not backup.exists()
backup.write_bytes(path.read_bytes())
module = json.loads(path.read_text())
for m in module['surfaces']:
    if m['id']=='M00':
        m['conclusion_changing_gaps'] = [x for x in m['conclusion_changing_gaps'] if '导入重定向' not in x]
        m['conclusion_changing_gaps'].append('DSpark legacy直接导入已核正；真实resolved类与初始化闭包仍待证')
    extra = []
    if m['id']=='M01':extra=['*/sampling_params.py','*/config/model.py']
    if m['id']=='M20':extra=['*/model_executor/models/qwen3_dspark.py']
    current = set(m['indexed_source_paths'])
    current.update(x['path'] for x in files if any(fnmatch.fnmatch(x['path'],pat) for pat in extra))
    m['indexed_source_paths'] = sorted(current)
    m['indexed_branch_ids'] = [x['id'] for x in sites if x['source'] in current]
    if m['id'] in ('M00','M01','M05','M16','M17','M18','M19','M20','M21','M23','M24'):
        m.setdefault('local_review_evidence',[]).extend(['DSPARK-CONTRACT-REVIEW.md','contract-applicability.json','dspark-branch-review-delta.json'])
        m['mandatory_gates']['scope_and_dispatch']['status']='PARTIAL_FROZEN_CONFIGURATION_DERIVATION; RUNTIME_AND_CLOSURE_OPEN'
    assert not m['complete']
module['source_ledger'] = 'coverage-ledger-v2'
module['review_delta'] = ['tail-branch-review-delta.json','dspark-branch-review-delta.json']
assert not module['goal_scope_closed'] and not module['overall_complete']
save('module-coverage-ledger.json',module)

summary = {'status':'PASS_LOCAL_CONTRACT_CHECKPOINT_ONLY','source_files':len(files),'unique_source_sites':counts,
    'prior_site_ids_retained':True,'new_local_branch_contracts':len(reviews),'total_selected_local_branch_contracts':len(reviews)+9,
    'module_surfaces':len(module['surfaces']),'whole_modules_closed':0,'overall_goal_complete':False}
save('contract-checkpoint-verification.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
