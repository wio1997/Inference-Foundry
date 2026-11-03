"""Preservev80evidence and publisha verifiedsuccessoraftertoolnamespace repair."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v81.json').exists()
    success=read(O/'A3-external-verifier-success-v81.json');assert success['success']and success['validated_checkpoint']==80
    history=read(O/'A3-datapath-validation-history-v81.json');assert len(history['errors'])==2
    history['recovery_policy']='V80analysis/verifier/manifestfilesremainimmutable. All85currentmanifestentrieswereverifiedbeforeordinaryversion81shared-state/publishertransition;sharedSTATEchangesareexpectedversionedcontinuationmutations.'
    write(O/'A3-datapath-validation-history-v81.json',history)
    for base in ['candidate-relation-clauses','candidate-relation-coverage','operator-analysis-coverage-ledger']:
        old=O/(base+'-v80.json');new=O/(base+'-v81.json');new.write_bytes(old.read_bytes())
    gate=deepcopy(read(O/'PROFILE-CODE-QUESTION-GATE-v80.json'));gate.update(verification_recovery='Correctedverifierweightbytespathnamespacenowvalidatesallv80source/data/statechecks;v80sourceunchanged,new81ledgercopiesanalysis.',new_remote_source_read=False,overall_complete=False);write(O/'PROFILE-CODE-QUESTION-GATE-v81.json',gate)
    previous=read(O/'REVIEW-CHECKPOINT-v80.json')
    remaining=list(read(O/'module-coverage-gaps-v80.json')['remaining'])
    action=('继续原Goal,v81验证修复checkpoint(PROGRESS，实质分析源v80，上一Goalturnv79亦PROGRESS)。'
        '先读A3-datapath-verification-recovery-summary-v81、PROFILE-FINDINGS-v81和A3-datapath-validation-history-v81；随后读v80的6primaryarchitecture页证据、97typed/30030PMU周期模型、Markov520、jointingresscut和CV/CC/VV接口重估。'
        'v80最终verifier局部W权重整数覆盖W目录，最后sharedstate路径检查TypeError；前面数据检查已完成。旧v80verifier及85artifactmanifest未编辑；新verify_A3_datapath_ceiling_v81.main(80)已完整通过并保存A3-external-verifier-success-v81。'
        '当前v81沿用v80所有算子/理论分析，只修复核验与版本化状态。verify_A3_datapath_ceiling_v81.py默认验证81应PASS。History保留prewriteelse2语法错误和postcheckpointW遮蔽错误，两个均resolved；新注册1error+2pass，旧Runs保留。不要执行旧badverifier80或回放finalizer80。'
        '下一实质问题仍是F23 A3Fixpipe/nativeatomic BF16round-before-add、onewriter与routedinputalias是否能保原DownBF16/mergeBF16两次舍入，只查primaryAPI/SDK定点证据，不实施/跑NPU；随后批量sourcegraph/candidatefootprint关系汇入。'
        '保持全部35模型+7历史框架/26模块/342分支记录24semantic、46pairclauses11groups/549中央未汇入。2073/285598conditionalreview9/1352pending、849PMU16phase3136ambiguity、0wholeclosure，actual最低模块MFU/全typedcapacity未建立。'
        '用户要求profile优先，代码只补算子融合/理论约束，不研究框架。无remote/server/model/Torch/NPU/newcapture/service/process/implementation，Goalactive，不因报告或核验绿灯complete。')
    report=(O/'PROFILE-FINDINGS-v80.md').read_text().replace('本轮补上','当前已补上').replace('本轮对97类','当前对97类')
    report+='\n核验恢复已完成：旧v80校验器的局部权重变量覆盖了目录变量，导致最后一组共享状态路径检查失败。新校验器保留旧文件和失败记录，已完整验证原检查点；当前v81沿用上述分析并修复核验流程，没有新增算子、设备执行或提速结论。\n'
    jobs=[
        dict(kind='review',status='error',description='Resolvedv80read-onlyverifierTypeError:integerweightWshadowedWdirectoryatrecoverystatepathcheck. V80source/manifestpreservedandnewnamespacedverifiercreated;noanalysisevidencechangedbyfailure.',artifacts=[O/'A3-datapath-validation-history-v81.json',N/'verify_A3_datapath_ceiling_v81.before-weight-shadow-fix.txt']),
        dict(kind='design-check',status='pass',description='Correctedverify_A3_datapath_ceiling_v81.main(80)fullypassed85manifesthashes,6primarydocs,97typed/30030PMU,Markov3/520,allsemantic/candidate/globalstatechecks beforeversion81transition. Exactsuccessreceiptretained.',artifacts=[O/'A3-external-verifier-success-v81.json',O/'checkpoint-manifest-v80.json',N/'verify_A3_datapath_ceiling_v81.py']),
        dict(kind='review',status='pass',description='Version81copiesv80relations/operatorregimes unchanged;no newcandidates/branches/source/remote/devicework. All342records24semanticmodules/46relations11groups/549pending/globaluncertaintyandGoalACTIVE preserved.',artifacts=[O/(b+'-v81.json')for b in ['candidate-relation-clauses','candidate-relation-coverage','operator-analysis-coverage-ledger']]+[O/'A3-datapath-validation-history-v81.json'])]
    persist_refinement(version=81,previous=80,title='Verified A3 datapath checkpoint recovery after local verifier namespace repair',
        summary=dict(filename='A3-datapath-verification-recovery-summary-v81.json',previous_turn_classification='PROGRESS:v80architecture/cycleanalysisverifiedafterlocaltoolrepair;priorgoalturnv79progress',
            analysis_source_version=80,new_operator_analysis_in_this_recovery=0,old_v80_immutable_analysis_files_changed=False,
            corrected_verifier_success_receipt='A3-external-verifier-success-v81.json',retained_resolved_local_verifier_errors=2,
            primary_document_snapshots_reused=6,typed_matrix_regimes=97,typed_matrix_executions=30030,Markov_regimes=3,Markov_executions=520,
            retained_global_records=342,retained_semantic_modules=24,relation_clauses=46,higher_order_groups=11,pending_central_pair_reconciliation=549,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            new_source_files_obtained=0,new_remote_source_read=False,implementation=False,native_test=False,new_performance_result=False,actual_lowest_MFU_module_established=False,all_actual_physical_numeric_time_ceilings_proved=False,
            remote_job_status='Alllocalverificationcomplete;no remote/server/model/Torch/NPU/capture/service/processwork;version81isverificationrecoveryonly.'),
        changes=['Correctedlocalverifiernamespacewithouteditingcompletedv80analysis/verifier/manifest;externalmain(80)nowpassesall85entriesandstatechecks.',
            'Retainbothresolvedverifierfailuresasprovenance;version81carriesallv80operatoranalyseswithoutscope/proofpromotion.',
            '35model7framework/342records24semantic/46relations11groups/549centralpending/profile2073/285598+9/1352/global84916/3136/0closurespreserved.'],
        candidate_updates={},framework_updates={},module_updates={},semantic_branches=None,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=['A3-datapath-validation-history-v81.json','A3-external-verifier-success-v81.json','A3-datapath-primary-evidence-v80.json','A3-typed-ingress-regime-models-v80.json','shared-CV-interface-revaluation-v80.json','next-operator-proof-questions-v80.json'],
        publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'verify_A3_datapath_ceiling_v81.py'],executor_prefix='A3-verification-recovery',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
