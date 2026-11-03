"""Route existing M22 navigation sites to operator contracts,not new semantics proofs."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import O,W,read,write


def main():
    assert not(O/'M22-adjacent-operator-boundary-crosswalk-v79.json').exists()
    inv=read(O/'M22-semantic-branch-inventory-v46.json');allrows=inv['static_branch_sites']
    selected=[r for r in allrows if r['semantic_family']in ['M22-S06','M22-S07','M22-S08','M22-S09']]
    assert len(selected)==900
    worker=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'
    h=hashlib.sha256(worker.read_bytes()).hexdigest();assert all(r['source_sha256']==h for r in selected)
    matrix={m['id']:m for m in read(O/'module-phase-branch-matrix-v78.json')['modules']}
    routes={
      'M22-S06':dict(modules=['M04','M05','M21'],evidence=['embedding-initialization-operator-contract-v67.json','selected-input-operator-contract-v70.json','planner-source-algorithm-contract-v71.json'],
          obligation='Preserve actual inputIDs/positions/lengths/slots/masks/metadata schemas,rowmaps and producer publication;operatorconditionalfusion requires declared reads/writes and capturedbank epochs.'),
      'M22-S07':dict(modules=['M15','M16','M17','M18','M19','M20','M24'],evidence=['heads-sampling-operator-contract-v65.json','index-context-operator-contract-v64.json','M24-branch-history-before-v65.json'],
          obligation='Preserveaccepted/rejected states,orderedaux/context sources,fullscore/RNG/outputsuffix and draft/target/iterationgeneration. Scalarvalue equalityalone doesnot remove async lifecycle.'),
      'M22-S08':dict(modules=['M00','M02','M08','M09','M10','M18','M24'],evidence=['C2-cache-object-and-allocation-contract-v25.json','state-engram-operator-contract-v62.json','ring-storage-source-contract-v79.json','ring-state-layout-and-lifetime-proof-v79.json'],
          obligation='Preserve exact physicalstorage/parameterlayout/tuple andownerkey,pagestride,fullobserver and allwriters/lastreader protocol. Shape/bytes/attribute names do not prove stateowneridentity.'),
      'M22-S09':dict(modules=['M00','M01','M03','M05','M21','M25'],evidence=['all-module-fixed-case-domain-audit-v78.json','M21-hardware-and-DSACP-dispatch-contract-v47.json','request-guard-saved-profile-v54.json','M25-parent-and-sentinel-contract-v54.json'],
          obligation='Retainoriginalframeworkcontrolflow,model/adapter/MM/backend selection and externalinput/output/error contracts;inspectsourceonlyifonechangesa specificoperatorrewrite,notallrunnerinternals.')}
    rows=[]
    for r in selected:
        rt=routes[r['semantic_family']]
        for name in rt['evidence']:assert(O/name).exists()
        rows.append(dict(site_id=r['id'],source=r['source'],source_sha256=r['source_sha256'],owner=r['owner'],line=r['line'],kind=r['kind'],condition=r['condition'],
            source_family=r['semantic_family'],original_review_status=r['review_status'],responsible_operator_modules=rt['modules'],
            operator_family_records=[dict(module=mid,canonical_families=[b.get('id',f'{mid}-LEGACY-{i:02d}')for i,b in enumerate(matrix[mid]['branches'],1)],
                source_matrix='module-phase-branch-matrix-v78.json',family_list_is_navigation_not_site_semantic_equivalence=True)for mid in rt['modules']],
            operator_contract_evidence=rt['evidence'],required_boundary_obligation=rt['obligation'],
            crosswalk_state='EXISTING_OPERATOR_BOUNDARY_REFERENCE_ROUTED;NOT_GUARD_DATAFLOW_OR_RUNTIME_PROOF',
            new_native_operator_or_conditional_branch_proof=False))
    counts=dict(Counter(r['source_family']for r in rows));assert counts=={'M22-S06':168,'M22-S07':149,'M22-S08':323,'M22-S09':260}
    unreviewed=[r for r in allrows if r['review_status']=='UNREVIEWED']
    remain=[r for r in unreviewed if r['id']not in {x['site_id']for x in rows}]
    assert len(unreviewed)==991 and len(remain)==91 and all(r['semantic_family']=='M22-S05'for r in remain)
    write(O/'M22-adjacent-operator-boundary-crosswalk-v79.json',dict(
        source_inventory='M22-semantic-branch-inventory-v46.json',source_inventory_sha256=hashlib.sha256((O/'M22-semantic-branch-inventory-v46.json').read_bytes()).hexdigest(),
        worker_snapshot_path=str(worker),worker_snapshot_sha256=h,records=rows,record_count=900,source_family_counts=counts,
        exact_prior_local_guard_dataflows_preserved=110,new_guard_dataflows_claimed=0,
        remaining_unrouted_static_navigation_sites=91,remaining_unrouted_family='M22-S05',
        distinction='900source-site associations are now indexed against operator contracts. Their originalUNREVIEWED states are preserved;mapping names/filenames orsitecounts isnot a proof all900conditions were analyzed. No frameworkbranch omitted/silentlyclosed.',
        operator_scope_rule='Use a contextualoperatorcontract:declare all value/shape/dtype/layout/read-write-footprint/alias/error/RNG/state/publication effects. Keep original surrounding controlflow. Allwithin-domain observationsmustmatch;outside thatdomain keepbaselinebranch orretainunknown. Onlycodequestions thataffecttheseeffects neednewsource reads.',
        graph_lifetime_gates=['Keepmain/draft/phase/layer/rank andcapturedbank namespace. Exactnativeparams,weakrefs/workspaceownership andoriginaloptionsremain.',
            'Source/inputcontentgeneration,physicalstorageowner andnativeparametergeneration aredistinct;stablepointer/registrykey isnotallthree.',
            'RetireafterallactualreadersANDwriters;preservecopy/clear/exception/RNG/fulloutput effects whereobservable.',
            'Opaque/full-byteexternalobservers orunknownalias inhibit specificstoragerewrites;they do notrequire unrelatedframeworkresearch.'],
        unresolved_actual_operand_domains=['9existingregimes1352exec','16phase-unboundgraphs','actualnewringresidentS/bankU/tupleowner/halfslotepochs','all decisivecapturedparam/address/lifetime bindings'],
        no_new_target_source_or_remote_work=True,whole_module_closed=False,overall_complete=False))
    print(dict(routed_navigation_sites=900,counts=counts,new_guard_dataflow_proofs=0,prior_guard_proofs=110,unrouted_lifecycle_sites=91,status='OPERATOR_BOUNDARY_CROSSWALK_READY'))


if __name__=='__main__':main()
