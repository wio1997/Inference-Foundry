#!/usr/bin/env python3
"""Attach human-reviewed local branch contracts; never close whole modules."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sites=json.loads((ROOT/'coverage-ledger/branch-sites.json').read_text())
reviews=[
('vocab_parallel_embedding.py','AscendLogitsProcessor._get_logits','skip_gather',
 'true返回未重建/未截断的local head结果；false继续按parallel_mode处理。','skip_gather已有接口，调用者须承担原词表重建/截断及全部消费者合同。'),
('vocab_parallel_embedding.py','AscendLogitsProcessor._get_logits','lm_head.parallel_mode is VocabParallelMode.PCP_X_TP',
 'true通过PCP再TP恢复词表；false继续细粒度或普通分支。','PCP×TP不能直接用单个标准TP组的candidate gather替换。'),
('vocab_parallel_embedding.py','AscendLogitsProcessor._get_logits','lm_head.parallel_mode is VocabParallelMode.FINE_GRAINED',
 'true先汇集hidden token行并all-to-all回局部token完整词表；false普通/复制头路径。','所有权重组须保留token行恢复与fine-grained组，不能只减少词表量。'),
('vocab_parallel_embedding.py','AscendLogitsProcessor._get_logits_normal','lm_head.tp_size > 1',
 'true重建词表；false复制头不参加词表gather。','Markov复制头与target/draft base头分开计账，不重复提出已有的skip-gather。'),
('sample/rejection_sampler.py','AscendRejectionSampler.forward','not self.is_processed_logprobs_mode',
 'true clone使raw和processed存储分离；false两名最初alias，后续原位processor也改变raw名所指值。','不能跨模式假定raw始终是相同版本。entropy关闭且max_num_logprobs=None时，true分支raw快照在本函数局部无数值消费者。'),
('sample/rejection_sampler.py','AscendRejectionSampler.forward','sampling_metadata.max_num_logprobs is not None',
 'true提取accepted/raw或processed及bonus内部结果；false返回不含该logprobs提取。','bonus整行结果和raw保留不能在true分支无条件删除；None分支提供局部物化收敛点。'),
('sample/rejection_sampler.py','rejection_sample','using_entropy_verify and ori_target_logits is not None',
 'true对原值参数做FP32 softmax供entropy；false令ori_target_probs=None。','原值副本删除必须按这个guard检查，不能只看请求temperature。'),
('sample/rejection_sampler.py','rejection_sample','sampling_metadata.all_greedy',
 '此函数包含两个同条件站点：前者设置is_greedy=None；后者在greedy验证后提前返回，跳过后续random分布及恢复。','请求temperature=0与greedy kernel为支持线索；完整metadata/合成与entropy条件仍分别保留待证。')]
delta=[]
for suffix,owner,condition,arms,implication in reviews:
    found=[s for s in sites if s['source'].startswith('vllm-ascend/') and s['source'].endswith(suffix) and s['owner']==owner and s['kind']=='If' and s['condition']==condition]
    assert found,(suffix,owner,condition)
    for s in found:delta.append({'branch_id':s['id'],'source':s['source'],'source_sha256':s['source_sha256'],'line':s['line'],
        'condition':condition,'both_arm_effect_summary':arms,'optimization_implication':implication,
        'semantic_status':'LOCAL_CONDITIONAL_EFFECTS_REVIEWED','runtime_applicability':'OPEN_NOT_A_DIRECT_WORKER_METADATA_SNAPSHOT',
        'all_consumers_closed':False,'whole_module_closed':False,'evidence':'TAIL-HEAD-SAMPLER-REVIEW.md'})
assert len({x['branch_id'] for x in delta})==len(delta)
(ROOT/'tail-branch-review-delta.json').write_text(json.dumps({'scope':'selected local branches only; original AST ledger preserved','reviews':delta,'overall_complete':False},ensure_ascii=False,indent=2)+'\n')
path=ROOT/'module-coverage-ledger.json';data=json.loads(path.read_text())
for m in data['surfaces']:
    if m['id'] in ('M16','M17'):
        m['local_review_evidence']=['TAIL-HEAD-SAMPLER-REVIEW.md','tail-branch-review-delta.json','head-sampler-family-summary.json']
        m['mandatory_gates']['value_and_all_consumers']['status']='PARTIAL_LOCAL_CONSUMER_CONTRACT; GLOBAL_OPEN'
        m['mandatory_gates']['resource_performance_and_bounds']['status']='PARTIAL_SHAPE_DTYPE_MEASUREMENTS; EXPOSURE_AND_BOUNDS_OPEN'
for r in data['rule_coverage']:
    if (r['surface'],r['rule']) in (('M16','R06'),('M17','R01')):
        r['status']='CONDITIONAL_REVIEW_PRESENT_OTHER_BOUNDARIES_OPEN';r['evidence']=['TAIL-HEAD-SAMPLER-REVIEW.md']
assert not data['overall_complete'] and all(not m['complete'] for m in data['surfaces'])
path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'local_branch_contracts':len(delta),'whole_modules_closed':0,'overall_complete':False}))
