"""Correct two prose denominators without changing any measured count or model."""
from copy import deepcopy
from pathlib import Path
from collections import Counter
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
    assert not (O/'REVIEW-CHECKPOINT-v59.json').exists()
    p=read(O/'HC-operator-profile-and-edges-v58.json')
    count=Counter(r['operator'] for r in p['regimes'])
    assert count=={'HcPre':8,'HcPost':8}
    pre=[r for r in p['regimes'] if r['operator']=='HcPre']
    assert sum(r['executions'] for r in pre)==8944 and all(r['all_observed_pre_mix_present'] for r in pre)
    c=deepcopy(read(O/'HC-operator-numerical-and-dependency-contract-v58.json'))
    old=c['facts'][0]
    assert old.startswith('All16profile regimes include external pre_mix.')
    c['facts'][0]=old.replace('All16profile regimes include external pre_mix.','All8capturedHcPre regimes include external pre_mix; the16totalHCregimes consistof8HcPre+8HcPost.')
    write(O/'HC-operator-numerical-and-dependency-contract-v59.json',c)
    v=deepcopy(read(O/'HC-fusion-split-reorganization-variants-v58.json'))
    condition=v['variants'][0]['conditions'][0]
    assert 'allcaptured16regimes' in condition
    v['variants'][0]['conditions'][0]=condition.replace('allcaptured16regimes','all8capturedHcPre regimes;16totalHCregimes include8HcPost')
    write(O/'HC-fusion-split-reorganization-variants-v59.json',v)
    write(O/'HC-regime-denominator-clarification-v59.json',dict(total_HC_regimes=16,HcPre_regimes=8,HcPost_regimes=8,
      HcPre_executions=8944,HcPost_executions=8944,all_HcPre_regimes_have_external_pre_mix=True,
      correction='v58 two prose strings said all16regimes have externalpre_mix; thecorrectdenominator is8HcPre regimes. HcPost doesnot have thatinput. Rawcounts,models,profilepairs andfusiondecisions unchanged.',
      supersedes=['HC-operator-numerical-and-dependency-contract-v58.json#facts[0]','HC-fusion-split-reorganization-variants-v58.json#variants[0].conditions[0]'],overall_complete=False))
    write(O/'PROFILE-CODE-QUESTION-GATE-v59.json',dict(status='HC_REGIME_DENOMINATORS_RECONCILED',
      previous_turn_classification='PROGRESS:v58HCoperator_fusion_split_work_and_bounds',source_read=False,new_device_work=False,
      question='Clarify16totalregimes versus8HcPre externalpre_mix regimes;no further source/frameworkresearch.',overall_complete=False))
    cp=read(O/'REVIEW-CHECKPOINT-v58.json')
    action=cp['next_action'].replace('最新v58','最新v59（v58算子分析加分母澄清）').replace('所有16profile signature有externalpre_mix','所有8个HcPre regime有externalpre_mix；另8个regime是HcPost')
    action=action.replace('HC-operator-numerical-and-dependency-contract-v58','HC-operator-numerical-and-dependency-contract-v59').replace('HC-fusion-split-reorganization-variants-v58','HC-fusion-split-reorganization-variants-v59')
    e=['HC-regime-denominator-clarification-v59.json','HC-operator-numerical-and-dependency-contract-v59.json',
      'HC-fusion-split-reorganization-variants-v59.json','HC-operator-profile-and-edges-v58.json','HC-work-traffic-and-reference-bounds-v58.json','operator-regime-refinement-v58.json']
    report=(O/'PROFILE-FINDINGS-v58.md').read_text().replace('覆盖16个捕获形状/角色/上下文组合，','覆盖16个捕获形状/角色/上下文组合（8个HcPre、8个HcPost），')
    persist_refinement(version=59,previous=58,title='HC captured-regime denominator clarification',
      summary=dict(filename='HC-regime-clarification-summary-v59.json',previous_turn_classification='PROGRESS:v58HCoperator_fusion_split_work_and_bounds',
        total_HC_regimes=16,HcPre_regimes=8,HcPost_regimes=8,HcPre_executions=8944,HcPost_executions=8944,
        raw_profile_counts_or_models_changed=False,new_source_read=False,implementation=False,native_test=False,new_performance_result=False,
        remote_job_status='No new remote work;v58HCoperator sources/profile/models retained,onlytwo prose denominators corrected.'),
      changes=['Correctexternalpre_mix denominator to8HcPre regimes;16totalincludes8HcPost. Rawcounts,models,operatorfusiondecisions unchanged.'],
      candidate_updates={'F02':dict(review_v59=dict(clarification=str(O/e[0]),numerical_contract=str(O/e[1]),conditional_variants=str(O/e[2]),
        relation='8HcPre+8HcPost=16totalregimes;all8HcPre externalpre_mix inputs present. v58operatorwork/fusion/ceilinganalysis andallgaps unchanged.',conditional_operator_analysis_deliverable=True))},
      framework_updates={},module_updates={'M06':dict(finding='Captured16HCregimes split8Pre/8Post;externalpre_mix claim appliesonlytoall8Pre regimes/8944Preexecutions. Originalwork/IO/fusionmodels andcounts unchanged.',
        evidence=e[:3],candidates=['F02'],gap='v58typedphysicalrate/cache/phase/newconsumerconditions andremainingoperatorscopegaps retained.')},
      remaining=deepcopy(read(O/'module-coverage-gaps-v58.json')['remaining']),action=action,
      jobs=[dict(kind='review',status='pass',description='8Pre+8Post denominator verifiedagainstall16capturedregimes;externalpre_mix onlyappliestoPre. Correctedprosewithout changingrawtimings/models/edgecounts orframeworkinventory.',artifacts=[O/x for x in e]+[Path(__file__)])],
      report=report,read_first=e,publication_rows={},gate_updates=dict(overall_complete=False),scripts=[],
      executor_prefix='zcode-HC-denominator',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
