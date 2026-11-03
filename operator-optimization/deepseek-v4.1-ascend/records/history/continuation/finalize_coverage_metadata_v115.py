"""Correct current relation-count narrative without changing operator evidence or closure."""
from pathlib import Path
from copy import deepcopy
import hashlib,json
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v115.json').exists()
    manifest=read(O/'checkpoint-manifest-v114.json')
    for a in manifest['artifacts']:
        p=Path(a['path']);b=p.read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256']
    rel=read(O/'candidate-relation-clauses-v114.json');cov=read(O/'candidate-relation-coverage-v114.json');nav=read(O/'candidate-pair-navigation-v114.json')
    clauses=rel['clauses'];groups=rel['higher_order_groups']
    pairset={tuple(sorted((r['a'],r['b'])))for r in clauses}
    reviewed={tuple(sorted((r['a'],r['b'])))for r in cov['pairs']if r['clauses']}
    navreviewed={tuple(sorted((r['a'],r['b'])))for r in nav['pairs']if r['semantic_pair_reconciliation_done_in_stated_scope']}
    assert len(clauses)==len(pairset)==len(reviewed)==len(navreviewed)==cov['scoped_conditional_pairs']==134
    assert pairset==reviewed==navreviewed and len(groups)==cov['higher_order_groups']==26
    assert len(cov['pairs'])==595 and len(cov['pairs'])-len(pairset)==cov['remaining_pair_relations']==461
    ids={r['id']for r in cov['nodes']};assert len(ids)==35
    assert len({g['id']for g in groups})==26 and all(set(g['members'])<=ids and len(g['members'])==len(set(g['members']))for g in groups)
    obligations=deepcopy(read(O/'coverage-current-obligations-v114.json'));g=next(g for g in obligations['obligations']if g['id']=='G10');before=g['remaining']
    assert 'Central134 conditional relations/25 groups leave461 pairs'in before
    counts=dict(conditional_pair_relations=len(pairset),higher_order_groups=len(groups),central_pairs_not_reconciled=595-len(pairset),pair_universe=595)
    g['remaining']=f"Central{counts['conditional_pair_relations']} conditional relations/{counts['higher_order_groups']} groups leave{counts['central_pairs_not_reconciled']} pairs to reconcile;"+before.split(';',1)[1]
    g['current_relation_counts_v115']=dict(**counts,authority='candidate-relation-coverage-v115.json',all_variants_closed=False)
    obligations.update(parent='coverage-current-obligations-v114.json',metadata_only_correction=True,new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v115.json',obligations)
    reconciliation=dict(previous_narrative=before,corrected_narrative=g['remaining'],current_counts=counts,
        checked_unique_clauses_coverage_pairs_navigation_pairs_and_group_members=True,
        prior_structured_counts_were_correct=True,new_operator_opportunities=0,new_native_promotions=0,new_whole_closures=0,
        previous_checkpoint_manifest_verified_files=len(manifest['artifacts']),overall_complete=False)
    write(O/'coverage-count-reconciliation-v115.json',reconciliation)
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-pair-navigation','candidate-operator-footprints','operator-analysis-coverage-ledger']:
        d=deepcopy(read(O/(base+'-v114.json')));d['coverage_metadata_correction_v115']=dict(evidence=['coverage-count-reconciliation-v115.json'],operator_evidence_or_counts_changed=False)
        write(O/(base+'-v115.json'),d)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v114.json'));audit.update(previous_goal_turn='PROGRESS:v114 normalINT64range andsample/hash composition;postcheck PASS166',input_checkpoint=114,followup_checkpoint=115)
    r=next(r for r in audit['requirements']if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION')
    assert all(r[k]==v for k,v in counts.items()if k in r)
    r.update(conditional_pair_relations=134,higher_order_groups=26,central_pairs_not_reconciled=461)
    audit['coverage_metadata_correction_v115']=reconciliation
    write(O/'whole-scope-requirement-audit-v115.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v115.json',dict(status='CURRENT_COVERAGE_METADATA_RECONCILIATION_ONLY',new_source_read=False,new_device_work=False,
        correction='CurrentG10narrative25→26groups;structured134/26/461andalloperatorconclusionsunchanged',overall_complete=False))
    remaining=[g['id']+': '+g['title']+'. '+g['remaining']for g in obligations['obligations']]
    remaining.append('Currentv115onlycorrectsG10narrativegroupcountto26andchecks134reviewed/461pending/595universe;no newoptimization opportunity orclosure. Substantive operatoranswerisv114sample/hash work. Padprobe stillawaits explicitreply.')
    report='''本轮修正覆盖记录中的一处滞后文字：v114的结构化关系台账、联合组列表和需求审计均为134条条件关系、26个联合组、461对待核对；G10的文字摘要却仍写25组。现已由关系与组的实际列表重新生成当前义务说明，保持旧v114产物作为历史记录。

核对了关系中的唯一候选对、coverage的已核对集合、navigation的已核对集合三者相等，均为134；总候选对595，剩余461；26个组的ID唯一、成员不重复且都属于35项候选。修正不改变任何算子证据、条件、原生归属、理论上限或完成状态，也不计为新增优化机会。

最新实质算子分析仍是v114：默认Engram整数范围证明、48B历史摘要与选定token的分离、跨迭代状态/lookup/192B发布的条件合同。详见PROFILE-FINDINGS-v114.md及对应源证据。

35项候选、353条语义记录/26暂定模块、9类/1352次待补证原生执行，以及849缺PMU、16未绑定phase图、3136含糊归属均不变，0个整模块关闭。全模块MFU、物理容量上界和整体覆盖仍未完成。原生Pad小探针继续等待单独明确回复；本轮没有远程、模型、NPU、捕获、replay或目标实现操作。
'''
    action=('继续原Goal，当前v115是coverage metadata correction，实质算子分析仍v114sample/hash。'
        '先读coverage-count-reconciliation-v115/current-obligations-v115、sample-hash-summary-v114及v114report/rangeproof/H/work/state/composition。'
        'v114postcheckPASS166，v115precheck验证66predecessorSHAs及relation/nav/coverage集合134、groups26、pending461、universe595。G10旧文字25groups已按实际list修正26，禁止把此勘误当新增优化机会。'
        '后续更新关系/组时应由当前结构化counts重建G10文字，不只对旧固定字符串replace。所有35/353/26模块、134/26联合组/461、813/1297/46、2073/285598conditional、9/1352pending849PMU16phase3136ambiguity0whole/noallMFU保持。'
        '继续有证据的未分析operator关系/条件分支，或补actual参数/metadata/reader关键证据；不要重复已答normalINT64range、Hfactor、slotmin(r,q)、typedclock/PMU公式。'
        'v111新Pad44BBOOL20→24/valueTrue async仍待明确reply，禁止run；v109最小probe已完成授权消耗。若新获准用v116独立记录按v111idle/container/SHAguards一capture/noReplay/noModel/noopaqueGPUderef。'
        'TaskACTIVE Loop004EVALUATING GoalACTIVE。别改完成v114/v115或重跑finalizer；postverify latestmanifest及当前G10与structured134/26/461一致。')
    persist_refinement(version=115,previous=114,title='Current coverage-count narrative reconciliation',
        summary=dict(filename='coverage-metadata-summary-v115.json',previous_turn_classification='PROGRESS:v114 normalINT64range andsample/hash composition;postcheck PASS166',
            metadata_only_correction=True,new_operator_opportunities=0,branch_records=353,semantic_modules=26,new_pairs=0,new_groups=0,
            relation_clauses=134,higher_order_groups=26,pending_central_pair_reconciliation=461,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
            new_device_work=False,implementation=False,native_test=False,remote_job_status='Localmetadatareconciliation finished;no remotejob/model/NPU/capture/replay. Padprobe unexecuted.'),
        changes=['ReconcileG10remaining-obligation narrative25→26groups againstactual134clause-pair/26group/461pending lists;operatorconclusionsunchanged.'],
        candidate_updates={},module_updates={},remaining=remaining,action=action,
        jobs=[dict(kind='design-check',status='pass',description='Check134uniqueclause/coverage/navigationpairs,595total/461pending,26uniquevalidgroups;correctcurrentG10narrativeonly.',
            artifacts=[O/'coverage-count-reconciliation-v115.json',O/'coverage-current-obligations-v115.json',O/'whole-scope-requirement-audit-v115.json'])],
        report=report,read_first=['coverage-count-reconciliation-v115.json','coverage-current-obligations-v115.json','sample-hash-summary-v114.json','Engram-history-factorization-v114.json','sample-hash-state-and-publication-v114.json'],
        publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'persist_profile_refinement_v90.py'],executor_prefix='coverage-metadata',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
