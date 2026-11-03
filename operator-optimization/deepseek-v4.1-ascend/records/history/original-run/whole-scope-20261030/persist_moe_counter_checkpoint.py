#!/usr/bin/env python3
"""Persist counter producer semantics and same-capture interfaces, with open value gaps."""
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

W=Path(__file__).resolve().parent;R=W.parent
read=lambda p:json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
files=read(W/'coverage-ledger-v3/source-files.json')
def source(suffix,line):
    found=[x for x in files if x['path'].startswith('vllm-ascend/') and x['path'].endswith(suffix)]
    assert len(found)==1,(suffix,len(found))
    f=found[0];assert hashlib.sha256(Path(f['local_path']).read_bytes()).hexdigest()==f['sha256']
    return {'source':f['path'],'source_sha256':f['sha256'],'line':line}
rows=read(W/'moe-interface-exports.json')['rows'];assert len(rows)==16
old=read(R/'diagnostic-20261029/official-analysis/operators.json')
hashes={x['source_dir']:x['sha256']['kernel_details.csv'] for x in old['rows']}
families={key:[] for key in ['target_compatible','draft_compatible']}
for row in rows:
    assert row['csv_sha256']==hashes[row['source_dir']]
    found=set()
    for g in row['groups']:
        parts=g['input_shapes'].strip('"').split(';')
        assert g['type']=='DispatchFFNCombine' and parts[0]=='3,5120'
        experts=int(parts[1].split(',')[0]);topk=int(parts[3].split(',')[1])
        label='target_compatible' if (experts,topk)==(24,6) else 'draft_compatible' if (experts,topk)==(8,3) else None
        assert label and label not in found,(row['rank'],experts,topk);found.add(label)
        assert g['input_dtypes']=='DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT;BOOL'
        assert g['output_shapes']=='"3,5120;'+str(experts)+'"' and g['output_dtypes']=='DT_BF16;INT32'
        values=[int(n) for n,c in g['duration_histogram_ns'].items() for _ in range(c)]
        expected=480 if experts==24 else 36
        assert len(values)==g['count']==expected and sum(values)==g['sum_ns'] and statistics.median(values)==g['median_ns']
        families[label].append({'rank':row['rank'],'input_rows':3,'local_experts':experts,'topk':topk,
            'counter_output_dtype':'INT32','counter_output_shape':[experts],'count':g['count'],'median_ns':g['median_ns']})
    assert len(found)==2
for label,items in families.items():assert {x['rank'] for x in items}==set(range(16))
facts=[
    {'id':'MC-ROLLBACK','fact_kind':'DOCUMENTED','claim':'user enable=1 sets global MegaMoe-supported false; only user2 tests CANN then normalizes effective enable to1',
     'evidence':source('vllm_ascend/ascend_config.py',670)},
    {'id':'MC-RESULT','fact_kind':'DOCUMENTED','claim':'custom branch passes preallocated counter as op output; CANN branch receives fresh tensor; result object carries actual counter',
     'evidence':source('ops/fused_moe/moe_comm_method.py',581)},
    {'id':'MC-COUNT-AXIS','fact_kind':'DOCUMENTED_SOURCE_SEMANTICS','claim':'GetCumsumForMMAIV accumulates each expert across source ranks; last rank row is per-expert total, not expert-axis prefix',
     'evidence':source('dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp',402)},
    {'id':'MC-COUNT-WRITE','fact_kind':'DOCUMENTED_SOURCE_EFFECT','claim':'core0 copies last source-rank row to INT32 expert output before later maxOutputSize clipping',
     'evidence':source('dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp',825)},
    {'id':'MC-CAPACITY','fact_kind':'DOCUMENTED_SOURCE_EFFECT','claim':'subsequent rowStart/row count and later expert processing may be clipped to maxOutputSize; raw counts are not processed-work identity',
     'evidence':source('dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp',848)},
    {'id':'MC-GROUP-TYPE','fact_kind':'DOCUMENTED','claim':'FusedExpertsResult defaults group_list_type=1; EPLB consumer treats1 as counts and0 as expert-axis prefix',
     'evidence':source('ops/fused_moe/moe_comm_method.py',89)},
]
record={'status':'PARTIAL_COUNTER_CONTRACT_AND_MEASURED_INTERFACES','facts':facts,
    'same_capture_families':{label:{'per_rank':items,'rank_median_call_median_ns':statistics.median(x['median_ns'] for x in items)} for label,items in families.items()},
    'actual_counter_values':None,'actual_active_mask_values':None,'actual_max_output_size_same_call':None,'actual_ep_group_identity_same_call':None,
    'conditional_later_bucket_bounds':{'premises':'all16 ranks form the same EP group, each at most3 physical rows; valid top-k IDs and receiver expertise; same bound for all ranks',
        'target_max_requested_receive_rows':16*3*min(6,24),'draft_max_requested_receive_rows':16*3*min(3,8),
        'not_a_necessary_work_lower_bound':True,'not_valid_for_prefill_or_other_buckets_without_rebinding':True},
    'finite_capture_requirements':['actual resolved branch and kernel/binary identity','same request/step, phase and target/draft/layer','EP membership and expert ownership',
        'input physical/valid rows and active mask','counter values copied after production before shared buffer overwrite','same-call capacity and before/after clipping meaning',
        'capture/replay-aware observation; do not report capture-time zeros as replay results','reference run, overhead and exact owned-environment restoration'],
    'all_consumers_closed':False,'whole_module_closed':False,'overall_complete':False,'review':'MOE-COUNTER-CONTRACT.md'}
save(W/'moe-counter-contract.json',record)

sites=read(W/'coverage-ledger-v3/branch-sites.json')
defs=[
    ('vllm_ascend/ascend_config.py','AscendConfig._validate_user_input_ranges','self.enable_fused_mc2 in (0, 1)',
        'true关闭MegaMoe支持；false继续测试用户值2。','固定用户值1推导true；不把包存在当执行MegaMoe'),
    ('vllm_ascend/ascend_config.py','AscendConfig._validate_user_input_ranges','self.enable_fused_mc2 == 2',
        'true检查CANN包并将effective enable改1；false不做该覆盖。','固定用户值1此前已命中首分支，本elif不执行'),
    ('ops/fused_moe/moe_comm_method.py','FusedMC2CommImpl.fused_experts','self.enable_fused_mc2 == 1',
        'true进入fused实现选择；false抛错。','固定effective值1推导true'),
    ('ops/fused_moe/moe_comm_method.py','FusedMC2CommImpl.fused_experts','_EXTRA_CTX.use_mega_moe',
        'true取得fresh CANN counter；false调用custom op写预分配counter。','普通固定初始化推导false；实际二进制和调用身份仍待绑定'),
]
reviews=[]
for suffix,owner,condition,effects,applies in defs:
    found=[x for x in sites if x['source'].startswith('vllm-ascend/') and x['source'].endswith(suffix) and x['owner']==owner and x['kind']=='If' and x['condition']==condition]
    assert len(found)==1,(owner,condition,len(found));x=found[0]
    reviews.append({k:x[k] for k in ['id','source','source_sha256','line','owner','kind','condition']}|{
        'both_arm_effect_summary':effects,'applicability':applies,'fact_kind':'DOCUMENTED_AND_CONDITIONAL_DERIVED',
        'runtime_directly_observed':False,'all_consumers_closed':False,'whole_module_closed':False,'evidence':'MOE-COUNTER-CONTRACT.md'})
save(W/'moe-branch-review-delta.json',{'reviews':reviews,'overall_complete':False,'source_ledger':'coverage-ledger-v3'})
backup=W/'before-moe-counter-review-records';backup.mkdir(exist_ok=False)
for name in ['STATE.json','PROGRESS.md','module-coverage-ledger.json']:(backup/name).write_bytes((W/name).read_bytes())
module=read(W/'module-coverage-ledger.json')
for m in module['surfaces']:
    if m['id'] in ('M12','M13','M23'):
        m.setdefault('local_review_evidence',[]).extend(['MOE-COUNTER-CONTRACT.md','moe-counter-contract.json','moe-branch-review-delta.json'])
        m['mandatory_gates']['scope_and_dispatch']['status']='PARTIAL_FUSED_BRANCH_ROLLBACK_AND_COUNTER_SOURCE_CONTRACT; RUNTIME_OPEN'
        m['mandatory_gates']['resource_performance_and_bounds']['status']='PARTIAL_PHYSICAL_INTERFACES; ACTUAL_COUNTS_MASK_CAPACITY_AND_EXPOSURE_OPEN'
    if m['id']=='M12':m['primitive_effect_reviews']=[x for x in facts if x['id'].startswith('MC-COUNT') or x['id']=='MC-CAPACITY']
    assert not m['complete']
module.setdefault('review_delta',[]).append('moe-branch-review-delta.json');assert not module['overall_complete']
save(W/'module-coverage-ledger.json',module)
state=read(W/'STATE.json');now=datetime.datetime.now(datetime.timezone.utc).isoformat()
state.update(updated_at_utc=now,selected_local_branch_contracts=47,moe_counter_review='MOE-COUNTER-CONTRACT.md',
    next_action='准备并执行有界、graph replay感知的同调用专家计数/mask/容量补采；先核实本人169实际占用和既有诊断脚本的恢复机制。禁止把capture零值、共享buffer最后层值或尺寸当计数。占用不允许时继续Indexer长度/mode及同次关键路径离线证据。')
state['conditional_derivations'].append('enable_fused_mc2 user1 validator disables CANN MegaMoe; custom counter sums source ranks per expert before capacity clipping')
assert state['goal_status']=='active' and not state['overall_complete'] and state['whole_modules_closed']==0
save(W/'STATE.json',state)
progress=(W/'PROGRESS.md').read_text().replace('43个选定分支','47个选定分支')
progress+='\nMoE新增实质证据：[计数合同](MOE-COUNTER-CONTRACT.md)明确用户值1的回退分支、逐专家计数轴及裁剪前语义。原later窗口16rank均有目标相容480次（输出INT32[24]）、draft相容36次（输出INT32[8]），但没有计数值。下一步须补同调用数值/mask/容量并防止graph capture与跨层覆盖误读；整体仍未完成。\n'
(W/'PROGRESS.md').write_text(progress)
outer=read(R/'STATE.json');outer['last_update']=now;outer['next_read_only_action']=state['next_action'];outer['whole_scope_current']['latest_semantic_review']='whole-scope-20261030/MOE-COUNTER-CONTRACT.md';save(R/'STATE.json',outer)

# Record this actual evidence checkpoint as a research Run, not numerical/model acceptance.
task=W/'tasks/whole-opportunity-coverage';loop=task/'loops/loop-001';ctl='/Users/wio/.codex/skills/operator-task-loop/scripts/taskctl.py'
commands=[]
def execute(args):
    result=subprocess.run(['python3',ctl]+args,text=True,capture_output=True,check=False)
    commands.append({'command':['python3',ctl]+args,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    save(W/'taskctl-initialization-moe-execution.json',{'commands':commands})
    assert result.returncode==0,commands[-1]
    return result.stdout.strip()
run=Path(execute(['new-run','--loop-dir',str(loop),'--case-key','c8-input32k-output1k-prefix90','--kind','review',
    '--command','Pinned initialization sources and local contract deltas; original16rank tail/MoE CSV analysis; C++ counter-axis/capacity review',
    '--code-ref','whole-scope-20261030 immutable snapshots and partial local review records; target unchanged']))
assert run.is_dir() and loop.resolve() in run.resolve().parents
args=['finish-run','--run-dir',str(run),'--status','pass','--correctness','not-applicable',
    '--summary','Local research checkpoint:573 source files,47 selected conditional branch contracts; fixed-text vision exclusion is phase-only; MoE output dtype/shape and producer semantics improved, but counter/mask/capacity values and complete coverage remain open. No model speedup/correctness or overall completion claim.']
for name in ['INITIALIZATION-AND-TAIL-REVIEW.md','initialization-checkpoint-verification.json','MOE-COUNTER-CONTRACT.md','moe-counter-contract.json','moe-interface-exports.json']:
    args+=['--artifact',str(W/name)]
execute(args)
index=read(task/'index.json');index['next_action']=state['next_action'];index['updated_at']=now;save(task/'index.json',index);execute(['compact','--task-dir',str(task)])

catalog=read(R/'RESEARCH-CATALOG.json');existing={x['relative_path']:x for x in catalog['indexed_files']}
changed=[R/'STATE.json']+[W/name for name in ['STATE.json','PROGRESS.md','module-coverage-ledger.json','MOE-COUNTER-CONTRACT.md','moe-counter-contract.json','moe-branch-review-delta.json']]
for p in changed:
    b=p.read_bytes();relative=str(p.relative_to(R));item=existing.get(relative,{})
    item.update(relative_path=relative,local_path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),current_local_record=True,
        server_scope='Updated local checkpoint; no claim of remote/public synchronization.')
    item.pop('server_mirror_path',None);item.pop('stage_mirror_path',None);existing[relative]=item
catalog['indexed_files']=list(existing.values());catalog['continuation_updated_at_utc']=now;save(R/'RESEARCH-CATALOG.json',catalog)
checkpoint={'status':'PASS_MOE_INTERFACES_AND_SOURCE_REFERENCES_ONLY','rank_csv_hashes_cross_checked':16,
    'new_selected_branch_contracts':4,'total_selected_branch_contracts':47,'actual_counter_values_observed':False,
    'family_rank_median_us':{label:value['rank_median_call_median_ns']/1000 for label,value in record['same_capture_families'].items()},
    'whole_modules_closed':0,'overall_goal_complete':False,'run':str(run)}
save(W/'moe-counter-checkpoint-verification.json',checkpoint)
print(json.dumps(checkpoint,ensure_ascii=False,indent=2))
