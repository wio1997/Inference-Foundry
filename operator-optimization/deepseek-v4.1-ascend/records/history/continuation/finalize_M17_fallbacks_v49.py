"""Persist a concrete finite fallback-source advance, retaining original whole-scope gaps."""
from copy import deepcopy
from pathlib import Path

from finalize_origin_coverage_v9 import N, O, read, write
from persist_profile_refinement_v3 import persist_refinement


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v49.json').exists()
    contract = read(O / 'M17-pytorch-fallback-contract-v49.json')
    proof = read(O / 'M17-block-backend-formula-counterexample-v49.json')
    cases = read(O / 'M17-fallback-local-guard-cases-v49.json')
    variants = read(O / 'M17-fallback-conditional-variants-v49.json')
    assert contract['local_body_review_complete'] and not contract['all_sampler_branches_or_consumers_closed']
    assert cases['functions'] == 4 and cases['if_sites'] == 10 and cases['if_arms_reviewed'] == 20
    assert proof['finite_checks']['cases'] == 6174 and proof['finite_checks']['differing_last_accepted'] == 214
    assert proof['witness']['triton_logical_output'] != proof['witness']['pytorch_logical_output']
    assert len(variants['variants']) == 3 and variants['new_main_candidates'] == 0
    evidence = ['M17-pytorch-fallback-source-v49.json', 'M17-pytorch-fallback-contract-v49.json',
                'M17-block-backend-formula-counterexample-v49.json', 'M17-fallback-local-guard-cases-v49.json',
                'M17-fallback-conditional-variants-v49.json', 'M17-probability-branch-ledger-v49.json']
    review = dict(contract=str(O / evidence[1]), counterexample=str(O / evidence[2]),
                  guard_cases=str(O / evidence[3]), variants=str(O / evidence[4]),
                  implementation=False, new_performance_result=False)
    candidate_updates = {
        'F35': dict(review_v49=dict(review,
            relation='Fourdirectfallbackbodies are now locally analyzed,not merely interface-bound. PyTorchblockproduct(min(r,1)) is not Tritonprefixmin(prefix*r,1);validtwo-vocabulary witness changes emitted length. Recoveryblockwise mode is independently selected. Existinglazy-recovery/entropy variants refined,not a newF36.'),
            profile_review_state='FALLBACK_ACCEPTANCE_AND_RECOVERY_BODIES_ANALYZED;BACKEND_DISTINCTIONS_CONDITIONAL',
            conclusion_changing_gap='Remaining whole sampler branches/score/raw/processor consumers and actual flag/alias/input/bank/RNG generations remain;current profile does not attest stochastic/block/fallback occurrence. Native parity and measured gain are separate from conditional semantic analysis completion.',
            next_profile_or_contract_action='Do not repeat the fourfallback bodies as wholly unread or unify block formulas. Advance remaining finite sampler/processor/consumer branches from the pinned profile selectors;read exact original saved metadata when access is restored. Preserve existingRNGdraws when considering selected-row recovery.'),
    }
    framework_updates = {
        'H02': dict(review_v49=dict(review,
            relation='Conditional fallback structure-buffer reuse now grounded in exact CPUconstant.to(device)sites,interval owner bitmap and separate blockwise recoverymax().item(). Counts are source sites,not measured executions/time. Keep B/S/device/dtype/generation/capture keys and all RNG draws;no currentfallback host saving asserted.')),
    }
    module_updates = {
        'M17': dict(
            finding='BothPyTorchacceptance fallback bodies andboth directlydispatched recoverybodies reviewed under exact historical166sourcehash.10ifsites/20arms locally interpreted. Backendblockclampformulas are generallydifferent;explicitrecoveryselector and F64uniform/QzeroInf/placeholder/output-retention boundaries documented. ConditionalF35/H02refinements registered,not native currentworkload findings.',
            evidence=evidence, candidates=['F35'],
            gap='WholeM17branchuniverse remains nonexhaustive;all score/raw/processor/logprob consumers,actual guard/input/alias/generation/capture/native/physicalcost evidence are not closed. This source advance does not require a new server login and does not restore oldrawdata.'),
    }
    remaining = deepcopy(read(O / 'module-coverage-gaps-v48.json')['remaining'])
    replacements = [
        ('fullnon-greedybody', 'remaining sampler/processor/non-greedy branches beyond the four v49reviewed fallback bodies'),
        ('fallbackbodiesnotfullyreviewed.', 'v49nowreviewsbothPyTorchacceptancebodiesandbothdirectrecoverybodies,includingbackend-specificclampandindependentrecoverydispatch;remainingwholesamplerbranches/consumersstayopen.'),
    ]
    changed = []
    for i, text in enumerate(remaining):
        after = text
        for old, new in replacements:
            after = after.replace(old, new)
        if after != text:
            changed.append(dict(index=i, before=text, after=after))
            remaining[i] = after
    assert len(changed) == 2
    write(O / 'M17-coverage-gap-reconciliation-v49.json', dict(
        parent_version=48, changes=changed,
        reason='Retire only the now-answered four body questions. Preserve whole-module and actual consumer/guard/numerical/resource gaps;implementation/native benchmark absence alone is not an unanalysed source question.',
        all_module_gaps_closed=False, overall_complete=False))
    remaining.append('M17v49:fourdirectPyTorchfallbackbodies now source-reviewed under historical166SHA,10ifsites/20arms;6174exactrationalchecks include214differentlastacceptpositionsbetweenTritonrunning-prefixclampandPyTorchper-factorclamp. A valid2-vocabularywitness changes outputlength;not a currentworkloaddefect claim. q=1/nonplaceholder/normalizedfinitep<=1gives real-prefixequalityonly;F64/F32/placeholder/native distinctions retained. Standard/blockverificationdoesnotchooseblockwiserecovery;separateuse_block_verifydefaultFalseatroot. OrdinaryresidualP-Dvsblockwisep_i*P-D,exclusiveprefix,QzeroInfhandlingnowbound. ExistingF35lazyrecovery/entropyandH02structure-buffer variants refinedconditionally;noF36/no implementation/newdevice/remote work. WholeM17remainingbranches/consumers and originalrawroute/generation/process/resource gaps stayopen.')
    action = (
        '继续原Goal。v49是实际有限分支进展：先读M17-pytorch-fallback-contract-v49.json、M17-block-backend-formula-counterexample-v49.json、M17-fallback-local-guard-cases-v49.json、M17-fallback-conditional-variants-v49.json及module-coverage-gaps-v49.json。'
        '四个原先接口级fallback函数已定点读完，exact历史166SHA bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda，10ifsite20arms是局部静态覆盖，不是完整M17分支数/执行频次。'
        'PyTorchblock先min(r,1)后cumprod，Triton每步min(prefix*r,1)，不可一般合并；有效两词表反例P=[[1/4,3/4],[0,1]],D=[[1/2,1/2],[1/2,1/2]],draft=[0,1],u=[.9,.9]，prefixT=[.5,1],P=[.5,.5]，last1vs-1。q=1+finite normalized/no-placeholder条件只给实数前缀等价，uniformF64/F32/placeholder/native仍不同。'
        '根调用573没有use_block_verify，默认False；选择blockverification不自动用blockwiserecovery。后者exclusivep_i使residual=max(p_i*P-D,0)，不是普通P-D；Q零/Inf防护、RNGdraw顺序/seeded零草稿副作用、greedybank保留必须守住。F35lazyrecovery/entropy摘要与H02结构buffer仅条件变体，不增加主候选。'
        '继续其余有限sampler/processor/score/raw消费者或其它原模块的明确未读条件，勿重读这些已答四函数或以未实现/nativebenchmark本身当未分析。当前执行profile840API/1144TASK/14chain/72greedyTriton不证明stochastic/block/fallback发生。'
        '原profile路由仍待用户提供：193root/hostname已验，同账号原166/实际105asset-no-match且rc0是假成功；193已查路径没资料不等于全盘无。不要重试旧密码<REDACTED_CREDENTIAL>put/serve哈希按需读取，不启动模型/torch/NPU/health/process/kills。'
        '缓存16device24Cube48Vector、参考1800→1.8GHz与286101组一致性已知；28PMU字段全读，849缺组、16zero-sidecar-candidate phase-unbound graphs、3136模块归属歧义执行、2082regimes、26theoryINCOMPLETE保持。35模型7框架0整模块；M21十八族268site、M22十六族1101site110coreguard保持。整体Goal仍active，未完成范围不以报告成稿消除。'
    )
    report = (
        '本轮从profile已定位的采样模块继续补齐明确代码缺口：只读两个PyTorch接受函数及其直接依赖的两个恢复函数，四个函数均与历史166源码SHA一致。4个函数、10个if位置及20个分支臂完成局部源码解释；这不是全M17覆盖，也不是运行频次。\n\n'
        '发现block接受公式的后端差异：PyTorch先逐项min(r,1)再cumprod，Triton每步min(prefix*r,1)。合法两词表反例的概率比分别为1/2、2，uniform均为0.9；Triton累计值为0.5、1，PyTorch为0.5、0.5，最后接受位置分别1与-1，因而发出的token数量不同。6174个精确有理数例中214个最后接受位置不同。这是源码公式的非等价证据，不是当前负载实测故障，也没有执行Torch/Triton或设备测试。\n\n'
        '若无draft_probs、q=1且概率有限归一化、有效段没有placeholder，两式在实数域可相同；仍须保留F64/F32随机数、placeholder、归约顺序及native差异，不能扩大为两后端无条件等价。\n\n'
        '恢复路径也有独立选择条件。根调用未传use_block_verify，默认False；启用block验证不会自动启用blockwise恢复。普通恢复使用max(P-D,0)，单独的blockwise路径使用排他前缀p_i后的max(p_i*P-D,0)。原有request共享指数随机数、零/Inf处理、greedy原输出保留和全部随机数draw顺序均已写入契约。\n\n'
        '这些证据细化了现有F35的先决策再恢复、逐行entropy摘要，以及H02的结构buffer复用条件。没有新增主候选，没有实现优化或声称时延收益。当前profile只提供840条采样API、1144条TASK、14条链及72次greedy Triton任务，不能证明非贪心、block或fallback分支发生。\n\n'
        '原profile访问入口仍未恢复；本轮没有新连服务器或启动模型、设备、服务/进程检查。其余采样条件、全部score/raw/processor消费者、跨层/卡/迭代关系和资源分母仍需继续。35模型候选、7框架候选、0整模块关闭；整体目标保持进行中。\n'
    )
    jobs = [
        dict(kind='review', status='pass', description='Profile-gated exact local source read:twoacceptance andtwo direct recovery helpers matchhistorical166SHA;noimport/device/remote work.',
             artifacts=[O / evidence[0], N / 'read_M17_pytorch_fallbacks_v49.py']),
        dict(kind='design-check', status='pass', description='Derivebackendclampnonidentitywithvalidprobabilitywitnessand6174exactrationalcases;214last-positiondifferences. Proofscopeissourceformula,notnativebackendexecution.',
             artifacts=[O / evidence[2], N / 'analyze_M17_fallback_contracts_v49.py']),
        dict(kind='review', status='pass', description='Documentfourfallbackbodies/10ifsites20arms,independentrecoverydispatch,residual/RNG/precision/retainedoutputcontracts;refineexistingF35/H02conditionalvariantsandretireonlyansweredbodygaps.',
             artifacts=[O / name for name in evidence[1:]] + [O / 'M17-coverage-gap-reconciliation-v49.json']),
    ]
    persist_refinement(version=49, previous=48,
        title='M17 finite PyTorch fallback and backend semantic contracts',
        summary=dict(filename='M17-fallback-summary-v49.json',
            previous_turn_classification='PROGRESS:v48profile_evidence_capability',
            selected_local_functions=4, static_if_sites=10, static_if_arms=20,
            rational_cases=6174, differing_last_accept=214,
            new_main_candidates=0, native_or_device_test=False,
            target_implementation_changed=False, original_profile_access_restored=False,
            remote_job_status='No remote operation or background job launched in v49. Allthree local source/algebra/contract jobs finished;current server health/workload not polled.'),
        changes=['Fourpreviouslyunreviewedfallbackbodiesnowlocallyanalyzedunderhistorical166SHA;10ifsites/20arms scoped.',
                 'Backendblockclampnonidentityhasvalidnormalizeddistributioncounterexample;6174exactrationalchecks,214differentlastpositions. Independentblockwiserecoveryselector/residualnowexplicit.',
                 'ExistingF35lazyrecovery/entropyandH02structurebufferconditionsrefined;no newmaincandidate,implementation/nativegainorwholemoduleclosure.'],
        candidate_updates=candidate_updates, framework_updates=framework_updates,
        module_updates=module_updates, remaining=remaining, action=action, jobs=jobs,
        report=report, read_first=evidence[1:] + ['original-profile-access-status-v47.json'],
        publication_rows={}, gate_updates=dict(overall_complete=False),
        scripts=[N / 'read_M17_pytorch_fallbacks_v49.py', N / 'analyze_M17_fallback_contracts_v49.py'],
        executor_prefix='zcode-M17-fallback', main_script=Path(__file__), source_read=True)


if __name__ == '__main__':
    main()
