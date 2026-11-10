#!/usr/bin/env python3
"""Record exact source walks, guarded parent obligations and callee frontier."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import ast,json
from build_recursive_source_worklist import MANIFESTS

R=Path('audit/execution-search')
REPORT='目标词表到下一轮调度的完整输出与递归缺口判断.md'
TITLE='2026-11-11 目标采样到下一轮：完整score读者与递归callee分账'
SUMMARY='介绍：沿原target词表→bonus/拒绝采样→完整ID/count/logprob/NaN输出→异步复制、RPC与Engine下一轮消费者。到了哪里：27当前源、12原Python协议/有限反例；两代表卡24完整target head/reject/seed链，另立全模型递归callee工作表。结果：逐位门槛不变，完整score有条件才能消掉；旧固定投影小方案继续停止，更大owner/Markov交付保留待确认。122/34只说明原输入谓词，全链路仍未闭合，新原语/框架准入0。'
DETAIL='''已停止：把“greedy draft”当作target全greedy；给原bonus Sampler.forward(max_num_logprobs=-1)换成仅IDs而漏掉其全score输出；只因argmax结束便删raw logits、NaN、logprobs或调度状态；用本卡全-inf替代MinTokens的全词表恢复条件；把thinking写1e9当作强制ID；在随机/混合/合成路径删RNG或忽略discard回退offset。冻结DSpark的processor factory只建MinTokens，custom processor报错，不能把未选中的MinP/LogitBias当当前耗时；普通无spec metadata的首步仍用同一个spec配置factory。

到了真实消费者：AscendRejectionSampler先取bonus私有行，强制max=-1返回raw/processed全scores；target私有选行转FP32，raw模式再clone，之后penalty/allowed/badwords/MinTokens/thinking与采样constraints。外层max_num_logprobs非None才合并target+bonus全scores并取logprobs；source getter仍有完整词表/排名读者。原bonus输出本身不可缩成IDs；只有把更大的完整拒绝交付作为替换原语、并证明对应score结果在该guard下死掉，才能消除内部全score副本。保留所有仍活的BF16/FP32边界、输入变异和输出。

合法条件分支：target all_greedy是每批sampling metadata，不能从draft策略推出。全greedy constraints直接返回；mixed/random会缩放、top-k/p、softmax、uniform与recovered exponential noise，按原per-request generator顺序执行；discard还回退offset4。synthetic+block明确抛错。entropy原softmax在greedy早返前计算，若只看该函数返回则该概率在全greedy分支死掉；冻结默认未启用，不给这条未匹配负载的新分支计价。MinTokens structured恢复先看屏蔽后全行isneginf，再只恢复有限stop值；分片要全局all-inf与原stop值。thinking状态按bonus→target顺序更新/改force_index与bonus_token_forced，并写FP32 1e9，inf/NaN仍可能改变winner。这些路径保留原实现或扩大摘要，不能静默略过。

交付一直追到下一代：sample_tokens先清12字段暂态，grammar可能把完整logits往返CPU，bookkeeping在采样后读原logits的NaN，并保留prompt-logprobs所需hidden。ID INT32[B,K+1]含bonus/placeholder，count/seed/padded、draft与后续状态都读它。AsyncGPUModelRunnerOutput持有device refs，复制流等生产流、复制ID/logprob/NaN再record event；get_output先sync再parse/发布。MP的WorkerProc.enqueue_output实际调用get_output，失败转FAILURE；future按队列顺序交给Engine，先处理abort再scheduler.update_from_output。InputBatch另持CPU/event，在下一次_sample前修复重排后的输出尾部，spec修复还会原地截短draft CPU列表。parse_output是(!=-1)&(<V)逐位置过滤，不能改成>=0或遇首-1即停止；logprobs过滤/丢弃顺序也保留。

下一轮状态不能吞掉：Scheduler先处理inflight/stale与延迟free，按原request→output索引取ID，扣rejected computed/placeholders，逐token append/check_stop并在stop截尾；AsyncScheduler对非stale扣已交付placeholder并cache_blocks。首拒绝/全接受/零draft、discard、abort/完成、stale preemption与structured接受分别有分支。摘要buffer不能复用原output未完成的槽；最省证明的边界是继续保留原fresh ID输出、Fill/reject/count/seed/D2H/RPC/Engine过程，仅替换闭合guard下的score交付。

完整价格分列：旧配平C8 EP0/8、M24/B4/D20/V129280的24链，target Post-ready→reject完成均0.825173%该窗，到seed/padded完成0.882463%，含必需GEMM/数学与等待，不能全算可省。保留原local GEMM/Fill/reject/seed和全部后续，只给已匹配score gather/transpose/选行/cast/clone/argmax族全免预算均0.352432%。新增local reduce+小通信+merge总40us/call，条件净2.778–2.852ms/0.298852–0.303327%该窗；100us降0.221403–0.226747%，240us只0.040690–0.048060%，350us转负。都是收费情景，不是新kernel实测。原C8全16卡另有112周期：head-end→seed整段全免最多0.608161%该周期，旧固定投影小配方不重开；不把此子段上界迁给更大的HC/LMHead布局/Markov/context共同父段。CPU完成/请求与六普通曝光未绑定，新增请求信用0。

待确认：较大词表owner/Markov/下一代交付可继续合并判断。最窄安全起点需同时满足all_greedy、standard、无grammar/full-score/logprob/NaN额外读者、无生效penalty/allowed/badword/MinTokens/thinking，或显式保留各读者。原local GEMM的M/N/K、dtype、舍入不动；native tie/NaN/all-inf/pad与global ID、完整输出/可见变异、capture/别名、collective及summary临时租约、实际eligible人口与新增成本仍缺。任何末位差异都不准入；12协议控制没有数值验收资格。

递归覆盖分账：工作表把模型原17个选中/待角色绑定root（排除冻结PCP）与16条C3/alias绑定、target输出消费者一起展开；80份带来源的Python源，319个(receiver,body)上下文、2739处静态callsite。这里只对可解析direct/self/super调用继续展开，字段receiver、闭包、构造、外部import与native边界逐条挂起；AST展开不是走读。34未定输入谓词仍单列，不能当34条未扫源码，也不能据319/2739宣布扫全。当前明确走读和旧SHA原文相交证据另列，未闭合边由后续逐项归并/核实。'''
NEXT='loop047下一Run：优先收敛recursive-source-worklist/disposition里的已选模型字段receiver、闭包与import/native边界，先按exact-SHA旧原文证据去重，再选第一个没有完整走读凭证的真实大交付；逐条记录分支leg、实际消费者、可省工作/原语全部输出与价格，不重做旧固定投影/clone/argmax/count或相同native fixture。34未定输入与未走读callee分账；add-HC/sharedDFC、metadata/gid12、Engram owner、完整Markov继续留具体缺口。每替换原语逐位一致；源先定范围再匹配旧C8/六普通；本源码Run无新Torch/NPU/模型/profile/请求/服务/远端写入。'

WALKS={
 'vllm/vllm/v1/sample/sampler.py':{'Sampler':['forward','apply_temperature','sample','compute_logprobs','gather_logprobs','_combine_outputs_with_spec_tokens','apply_logits_processors','apply_penalties']},
 'vllm-ascend/vllm_ascend/sample/sampler.py':{'AscendSampler':['__init__','greedy_sample','apply_penalties'],'AscendTopKTopPSampler':['forward_native'],None:['random_sample','_apply_top_k_top_p_pytorch','_apply_top_k_top_p_torch_npu']},
 'vllm-ascend/vllm_ascend/sample/rejection_sampler.py':{'AscendRejectionSampler':['__init__','forward','apply_logits_processors','apply_penalties','_log_rejection_sampler_entry','_log_rejection_sampler_exit'],None:['apply_sampling_constraints','rejection_sample','sample_recovered_tokens']},
 'vllm/vllm/v1/sample/rejection_sampler.py':{'RejectionSampler':['_get_logprobs_tensors','parse_output','_combine_outputs_with_spec_tokens'],None:['generate_uniform_probs']},
 'vllm/vllm/v1/sample/logits_processor/builtin.py':{'MinTokensLogitsProcessor':['_mask_stop_token_logits','apply','apply_with_spec_decode']},
 'vllm/vllm/v1/sample/logits_processor/__init__.py':{None:['build_logitsprocs']},
 'vllm/vllm/v1/sample/ops/bad_words.py':{None:['_apply_bad_words_single_batch','apply_bad_words','apply_bad_words_with_drafts']},
 'vllm-ascend/vllm_ascend/sample/penalties.py':{None:['apply_all_penalties']},
 'vllm/vllm/v1/sample/thinking_budget_state.py':{'ThinkingBudgetStateHolder':['has_tracked_requests','update_state','apply_to_logits','_apply_forcing_to_logits']},
 'vllm/vllm/v1/worker/gpu_input_batch.py':{'InputBatch':['set_async_sampled_token_ids','update_async_output_token_ids','update_async_spec_token_ids']},
 'vllm/vllm/v1/worker/gpu_model_runner.py':{'AsyncGPUModelRunnerOutput':['__init__','get_output']},
 'vllm-ascend/vllm_ascend/worker/model_runner_v1.py':{'NPUModelRunner':['sample_tokens','_sample','_get_nans_in_logits','_bookkeeping_sync']},
 'vllm/vllm/v1/engine/core.py':{'EngineCore':['step','post_step','step_with_batch_queue']},
 'vllm/vllm/v1/executor/multiproc_executor.py':{'WorkerProc':['enqueue_output','handle_output','async_output_busy_loop'],'FutureWrapper':['result','_wait_for_response']},
 'vllm/vllm/v1/core/sched/scheduler.py':{'Scheduler':['update_from_output','_update_request_with_output','update_draft_token_ids','update_draft_token_ids_in_output']},
 'vllm/vllm/v1/core/sched/async_scheduler.py':{'AsyncScheduler':['_update_after_schedule','_update_request_with_output']},
 'vllm/vllm/v1/core/sched/utils.py':{None:['check_stop']},
}

def main():
    pins={s['runtime_path']:s for n in ['target-output-dependencies.json','target-output-aux-dependencies.json'] for s in json.loads((R/n).read_text())['sources']}
    excerpts=[]
    for name,owners in WALKS.items():
        pin=pins['/vllm-workspace/'+name];raw=Path(pin['local_path']).read_bytes();assert sha256(raw).hexdigest()==pin['runtime_sha256'];tree=ast.parse(raw);lines=raw.decode().splitlines()
        for owner,names in owners.items():
            body=tree.body if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner).body
            for name in names:
                f=next(n for n in body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
                excerpts.append(dict(runtime_path=pin['runtime_path'],sha256=pin['runtime_sha256'],owner=owner,name=name,first=f.lineno,last=f.end_lineno,lines=lines[f.lineno-1:f.end_lineno],scope='Named body source walk only; native, field-receiver, state-helper and legality/population obligations remain on the recursive frontier.'))
    (R/'target-output-source-excerpts.json').write_text(json.dumps(dict(excerpts=excerpts,qualification='Exact listed bodies, not27 whole files or a full recursive/numerical proof.'),ensure_ascii=False,indent=2)+'\n')
    work=json.loads((R/'recursive-source-worklist.json').read_text());reviewed=[];prior=[]
    # Only byte-matching source-excerpt records qualify as prior range evidence.
    candidates=[]
    def collect(value,origin):
        if isinstance(value,list):
            for x in value:collect(x,origin)
        elif isinstance(value,dict):
            p=value.get('runtime_path');h=value.get('sha256');a=value.get('first',value.get('first_line'));b=value.get('last',value.get('last_line'));ls=value.get('lines')
            if p and h and isinstance(a,int) and isinstance(b,int) and isinstance(ls,list) and all(isinstance(x,str) for x in ls):
                candidates.append(dict(runtime_path=p,sha256=h,first=a,last=b,lines=ls,artifact=origin))
            for k,x in value.items():
                if k!='lines' and isinstance(x,(list,dict)):collect(x,origin)
    prior_inputs=[]
    for p in sorted(R.glob('*source-excerpts.json')):
        if p.name=='target-output-source-excerpts.json':continue
        prior_inputs.append(dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()))
        collect(json.loads(p.read_text()),str(p))
    known={}
    for name in MANIFESTS:
        for p in json.loads((R/name).read_text())['sources']:
            h=p.get('runtime_sha256',p.get('sha256'))
            if p['runtime_path'].endswith('.py'):known[p['runtime_path'],h]=Path(p['local_path'])
    valid=[];unmatched=[]
    for e in candidates:
        p=known.get((e['runtime_path'],e['sha256']))
        if p is None:continue
        if e['lines']==p.read_text().splitlines()[e['first']-1:e['last']]:valid.append(e)
        else:unmatched.append({k:v for k,v in e.items() if k!='lines'})
    candidates=valid
    for n in work['nodes']:
        own=[e for e in excerpts if (e['runtime_path'],e['sha256'],e['first'],e['last'])==(n['runtime_path'],n['sha256'],n['first'],n['last'])]
        if own:reviewed.append(dict(node=n['id'],body_source_walk='CURRENT_BOUNDED_SOURCE_WALK',all_recursive_callees_closed=False,branch_legs='Body alternatives inspected; predicate observations/native/value evidence separate.'))
        matches=[]
        for e in candidates:
            if e['runtime_path']!=n['runtime_path'] or e['sha256']!=n['sha256'] or e['last']<n['first'] or e['first']>n['last']:continue
            matches.append(dict(artifact=e['artifact'],first=e['first'],last=e['last'],covers_entire_body=e['first']<=n['first'] and e['last']>=n['last']))
        if matches:prior.append(dict(node=n['id'],ranges=matches,qualification='Exact source SHA + saved excerpt ranges, not automatic recursive/branch/consumer coverage.'))
    own_nodes={p['node'] for p in reviewed}
    no_whole=[dict(node=n['id'],runtime_path=n['runtime_path'],first=n['first'],last=n['last']) for n in work['nodes'] if n['id'] not in own_nodes and not any(x['node']==n['id'] and any(e['covers_entire_body'] for e in x['ranges']) for x in prior)]
    disposition=dict(current_body_source_walks=reviewed,prior_exact_SHA_excerpt_intersections=prior,prior_excerpt_inputs=prior_inputs,worklist_sha256=sha256((R/'recursive-source-worklist.json').read_bytes()).hexdigest(),unmatched_saved_excerpt_ranges=unmatched,no_entire_body_excerpt_or_current_walk=no_whole,unresolved_edges=[dict(from_node=e['from_node'],line=e['line'],callee=e['callee'],state=e['state']) for e in work['edges'] if e['to_node'] is None and e['state']!='PYTHON_LANGUAGE_BOUNDARY_RETAINED'],qualification='Source text presence is not proof of fusion searching. Current source walks preserve native/opaque/state helper boundaries; older intersections require identical SHA and exact saved line bytes and only avoid re-reading already recorded ranges. All remaining receiver/import/closure/native edges require explicit semantic disposition.',whole_chain_complete=False)
    (R/'recursive-source-disposition.json').write_text(json.dumps(disposition,ensure_ascii=False,indent=2)+'\n')
    report=dict(status='TARGET_COMPLETE_OUTPUT_READERS_AND_RECURSIVE_FRONTIER_BOUNDED_SOURCE_REVIEW',current_source_pins=27,original_protocol_controls=12,source_body_walks=len(excerpts),classification={'已确认可实现':['本轮新增0。既有封闭CPU原型1按原资格保留；源码/协议不算native逐位准入。'],'已停止':['独立IDs-only替换原bonus Sampler(max=-1)','原固定GEMM之外的clone/cast/argmax/count及固定摘要旧小配方，沿用原预算；不重跑原native fixture','局部all-inf恢复、thinking直接ID、greedy draft等于target all-greedy和丢RNG/输出状态的错误化简'],'待确认':['更大词表owner/Markov/完整下一代交付；guarded score DCE需在完整原语内闭合读者','319body/2739callsite保守递归前沿的字段receiver/import/闭包/native与合法leg逐条收敛','保留add-HC/sharedDFC、metadata/gid12和Engram owner已有较大候选']},minimal_summary_guard=['target all_greedy, standard rejection; no selected grammar/full-score/logprob/NaN/entropy reader unless explicitly preserved','no effective allowed/badword/penalty/MinTokens/thinking processor unless exact sharded behavior and state transitions are implemented','preserve original local GEMM shape/dtype/math and all complete output IDs/placeholders/bonus, aliases/capture/lifetimes; original Fill/reject/seed/D2H/RPC/Engine unchanged in priced child'],semantic_corrections=['Original bonus Sampler returns full raw scores even when outer request asks none; outputs can become dead only inside the larger guarded complete rejection primitive.','MinTokens restoration needs global all(-inf) after masking and finite saved stop values; 1e9 thinking writes are arithmetic, not an unconditional winner.','Greedy draft distribution is distinct from target sampling metadata. Mixed/random/synthetic preserve uniform/exponential draws, seeded overwrite order and discard generator offset rollback.','Parser filters every (!=-1)&(<V) position; next-worker repair trims at first sentinel and mutates spec CPU lists. Preserve both original contracts, request ordering and stale/abort/KV retirement.'],recursive_frontier=dict(sources=work['source_count'],nodes=work['node_count'],calls=work['edge_count'],current_bounded_nodes=len(reviewed),prior_excerpt_nodes=len(prior),no_entire_body_excerpt_or_current_walk=len(no_whole),old_predicate_bound=122,old_unbound_inputs=34,all_recursive_callees_closed=False),remaining=['Stateful thinking _update_think_state and penalty Triton/native numerical bodies stay original/opaque for this guarded child; broader fusion requires their separate source/primitive proof.','Actual installed executor platform/monkeypatch dispatch is referenced by older exact-source receipt; current base queue/get_output bodies pinned, no newly loaded-object inspection.','Random/top-p/logprobs/grammar branches have score readers, retained baseline; not admitted as summary-only. Their native kernels are explicit worklist boundaries, not declared swept.','Whole model field receivers/constructor choices, closures and custom ops remain explicit; source worklist is an overapproximation including optional/unselected bodies.','Sixordinary/all16rank/client exposure and native argmax+packet/lease/paid costs remain unknown; request credit0.'],every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],next_action=NEXT,inputs=[dict(path=str(R/n),sha256=sha256((R/n).read_bytes()).hexdigest()) for n in ['target-output-dependencies.json','target-output-aux-dependencies.json','target-output-parent-primary.json','target-output-controls.json','target-output-price.json','target-output-source-excerpts.json','recursive-source-worklist.json','recursive-source-disposition.json']],reviewer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (R/'target-output-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (R/REPORT).write_text('# '+TITLE+'\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一条待扫：'+NEXT+'\n')
    print(json.dumps(dict(current_sources=27,controls=12,body_source_walks=len(excerpts),current_bounded_nodes=len(reviewed),prior_excerpt_nodes=len(prior),no_entire_body_excerpt_or_current_walk=len(no_whole),whole_chain_complete=False)))
if __name__=='__main__':main()
