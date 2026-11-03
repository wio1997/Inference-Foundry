"""Finite registry/workspace guard cases, namespace invariants and update/sync alternatives."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
import hashlib
import json

from finalize_origin_coverage_v9 import O,read,write
from finalize_Engram_whole_frontier_v30 import graph


def main():
    assert not(O/'REVIEW-CHECKPOINT-v45.json').exists()
    stamp=datetime.now(timezone.utc).isoformat()
    src=read(O/'M22-registry-workspace-point-source-v45.json')
    assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
    sem=deepcopy(read(O/'M22-semantic-branch-inventory-v44.json'))
    sites=[s for s in sem['static_branch_sites']if s['semantic_family']=='M22-S02'];assert len(sites)==28
    notes={
        46:'Check each marker substring against lowercased RuntimeError text; CPU diagnostic predicate only, not device failure/performance evidence.',
        47:'Either explicit resource marker, or error-code AND stream-resource text qualifies; code alone is insufficient. Caller wraps then raises the error.',
        52:'Old-stack predicate case-folds text and checks marker substring; no model/device work or current upgrade requirement follows from this helper.',
        144:'Missing wrapper attribute delegates to current runnable only when runnable has that attribute. Do not cache a delegated owner across runnable/config replacement without proof.',
        146:'Debug flag selects richer AttributeError text; it does not validate graph tensor addresses, mutate graph state or change missing-attribute control outcome.',
        323:'None registry is a no-op for weak-ref conversion; source default None is not an actual runtime registry snapshot.',
        325:'Visit every currently present workspace key, not only initial capture sizes; other registry planes are not modified here.',
        326:'None workspace is skipped; non-None reference is replaced by weak_ref_tensors. Preserve lifetime/alias owners; this is not a tensor copy/clear or evidence that memory is free.',
        340:'Eligible updatable backend returns before context/backend-impl update. Otherwise set_current_vllm_config protects get_impl_cls resolution and forwards current metadata/scalars/spec/draft data.',
        373:'Main registry initializer is one-time in this helper interface and raises if already non-None; graph resizing cannot simply re-call setter.',
        376:'Initialize main events with a distinct empty list per capture-size key; no ExternalEvent objects allocated by this comprehension.',
        377:'Initialize main workspace values None for capture keys; this does not allocate GPU workspace storage.',
        378:'Initialize main handle lists per key; Python container identity does not prove unique native handle generations.',
        379:'Initialize main attention-parameter lists per key; equal empty contents are not shareable mutable list identity.',
        385:'Only a non-None main registry receives workspace assignment. Assignment may add a new num_tokens key without adding events/handles/attn_params keys.',
        398:'Draft registry has its own one-time initialization guard; it is not main registry even if token-key set is equal.',
        401:'Draft events are fresh lists per key/namespace; preserve actual event pairing/order/generation.',
        402:'Draft workspace starts None; no actual draft allocation/activity inferred, especially with current outer eager draft.',
        403:'Draft handle lists are distinct Python lists, not proof of native task identity or different backing memory.',
        404:'Draft attention parameters stay in draft namespace and layer/capture order; same num_tokens does not identify main parameters.',
        410:'Draft workspace update silently does nothing before registry creation, and otherwise may add a workspace-only key.',
        423:'Draft-prefill registry is another independent initializer; source presence does not prove current dspark execution uses it.',
        426:'Draft-prefill events are fresh per key, not automatically interchangeable with draft-decode events.',
        427:'Draft-prefill workspace initial None is a placeholder, not no future native workspace requirement.',
        428:'Draft-prefill native-handle list needs its own captured task/order/generation proof before reuse.',
        429:'Draft-prefill attention-param tuples are independently stored; no cross-phase CSE by token count alone.',
        435:'Draft-prefill workspace assignment requires initialized registry; adding workspace key does not register a complete graph bucket.',
    }
    ev=['M22-registry-workspace-point-source-v45.json','M22-registry-and-update-alternative-contract-v45.json']
    rows=[]
    for s in sites:
        rows.append(dict(id=s['id'],owner=s['owner'],line=s['line'],kind=s['kind'],condition=s['condition'],semantics=notes[s['line']],evidence=ev,
                         state='EXACT_GUARD_LOCAL_DATAFLOW_AND_CONDITIONAL_DOMAIN_RECONCILED',actual_native_handle_or_workspace_generation_proved=False))
    assert len(rows)==28
    prior=read(O/'M22-original-scope-and-guard-reconciliation-v44.json')
    merged=deepcopy(prior['guard_dataflow_reconciled'])+rows
    assert len(merged)==len({r['id']for r in merged})==69
    graph_sites={s['id']for s in sem['static_branch_sites']if 'compilation/acl_graph.py'in s['source']}
    assert len(graph_sites)==55 and graph_sites<={r['id']for r in merged}
    write(O/'M22-registry-guard-crosswalk-v45.json',dict(generated_utc=stamp,new_reconciled_guards=rows,new_guard_sites=28,
        total_M22_core_guard_dataflows=69,graphfile_static_sites=55,graphfile_reconciled_site_ids=sorted(graph_sites),
        remaining_graphfile_helper_sites=0,actual_full_module_or_native_implementation_proved=False,
        whole_module_closed=False,overall_complete=False))

    facts=[
        dict(grade='DOCUMENTED',statement='GraphParams stores four maps keyed by num_tokens: events,workspaces,handles,attn_params. Main,draft,draft-prefill are three separate global namespaces and one-time setters; same integer token key does not identify a complete captured task/parameter/bank generation.'),
        dict(grade='DOCUMENTED',statement='Initializer creates Python maps and per-key empty lists; workspace starts None. It does not allocate a Tensor workspace or native event/handle object here. Distinct registry/list objects do not prove distinct physical backing memory or native-handle epochs.'),
        dict(grade='DOCUMENTED',statement='Each workspace updater is a no-op while its registry is None, else assigns workspaces[num_tokens] with no membership/whole-bucket initialization check. It may add a workspace-only key. Workspace-key presence alone is not graph/event/handle/parameter readiness.'),
        dict(grade='DOCUMENTED',statement='weak_ref_workspaces iterates actual workspace keys, skips None, and replaces only workspace references through weak_ref_tensors. It does not clear/convert events,handles,attention tuples, nor prove underlying allocation dead or removable device-copy work.'),
        dict(grade='DOCUMENTED',statement='Legacy update helper returns immediately for use_updatable_graph backend. Otherwise get_impl_cls is resolved inside current-vllm-config context and receives update_stream,forward_context,num_tokens,vllm_config,speculative_config,draft_attn_metadatas. Hoisting dispatch needs config/backend/owner/purity proof; no current hot-path cost is measured.'),
        dict(grade='DERIVED_WITH_SOURCE_CONDITIONS',statement='For one active ordinary FULL wrapper with same stable backend/predicate as runner helper, updatable graph update and backend-impl update are mutually exclusive. They are alternatives, not two independent updates whose costs can be added.'),
        dict(grade='DERIVED_WITH_SOURCE_CONDITIONS',statement='For one runner/ordinary-wrapper pair with the same ENPU boolean passed during construction, wrapper stream synchronize requires not ENPU; runner generic-helper synchronize requires ENPU. At most one of these explicit barriers is selected. This is not a statement about all nested wrappers,backend/native barriers or actual observed wait time.'),
        dict(grade='DOCUMENTED',statement='Attribute proxy delegates current runnable or raises AttributeError, with DEBUG affecting message only. Error recognizers only classify RuntimeError text; no actual HDK/stream failure or hardware recommendation is inferred.'),
    ]
    write(O/'M22-registry-and-update-alternative-contract-v45.json',dict(generated_utc=stamp,facts=facts,
        sources=['M22-registry-workspace-point-source-v45.json','M22-capture-replay-input-output-and-update-contract-v44.json','M22-runner-update-and-enpu-binding-point-source-v44.json'],
        actual_registry_snapshots_available=False,actual_workspace_pointers_available=False,actual_event_handle_generations_available=False,
        actual_ENPU_or_backend_bound=False,
        not_new_gains=['Python registry/list initialization as GPU memory allocation','weak references as Tensor copies/zeroes',
                       'mutually exclusive update paths as additive native work','alternative wrapper/runner barriers as two removable waits'],
        retained=['Actual bucket+namespace+graph/task/layer/event/handle/workspace binding and every owner/readiness/retirement epoch.',
                  'Native backend update implementation/epoch semantics,live ENPU/backend/mode where decisive.',
                  '16phase-unbound graphs,worker adjacent scope,resource/fullprocess/global26 coverage.'],
        implementation=False,native_validation=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))

    # Pure Python models of the source maps; no Tensor/Event/native objects created.
    def initialize(sizes):return dict(events={n:[]for n in sizes},workspaces={n:None for n in sizes},handles={n:[]for n in sizes},attn_params={n:[]for n in sizes})
    def workspace_update(reg,n,value):
        if reg is not None:reg['workspaces'][n]=value
    def initialize_once(reg,sizes):
        if reg is not None:raise ValueError('already initialized')
        return initialize(sizes)
    sizes=[8,16,24];main=initialize(sizes);draft=initialize(sizes);prefill=initialize(sizes)
    assert len({id(v)for r in [main,draft,prefill]for k in ['events','handles','attn_params']for v in r[k].values()})==27
    workspace_update(None,8,'ignored')
    workspace_update(main,12,'workspace12')
    assert 12 in main['workspaces'] and all(12 not in main[k]for k in ['events','handles','attn_params'])
    shared={'opaque_storage':'S'};workspace_update(main,24,shared);workspace_update(draft,24,shared)
    assert main is not draft and main['workspaces'][24]is draft['workspaces'][24]
    rejected=False
    try:initialize_once(main,sizes)
    except ValueError:rejected=True
    assert rejected
    write(O/'M22-registry-key-completeness-and-alias-proof-v45.json',dict(generated_utc=stamp,grade='DERIVED_SOURCE_MAP_MODEL',
        namespace_example=['main:24','draft:24','draft_prefill:24'],capture_sizes=sizes,
        fresh_list_containers_across_three_namespaces=27,
        workspace_only_key=dict(key=12,workspace_present=True,events_present=False,handles_present=False,attn_params_present=False),
        distinct_namespace_can_reference_same_opaque_storage=True,second_initialization_rejected=True,update_before_initialization_is_noop=True,
        implications=['Bucket readiness needs all required graph/parameter/event/handle planes,not workspace key alone.',
                      'Namespace separation is semantic owner separation,not proof of physical memory duplication or copy elimination.',
                      'New bucket/reconfigured generation cannot be established by calling existing one-time setter again.'],
        model_not_actual_runtime_snapshot=True,no_device_objects_created=True,whole_module_closed=False,overall_complete=False))

    valuations=[]
    for full,replay,capturing,sparse,compress,enpu,updatable,draft_eagle in product([False,True],repeat=8):
        guard=full and not capturing and not sparse and not compress
        wrapper_update=full and replay and updatable
        backend_impl_update=guard and not updatable
        wrapper_sync=full and replay and not draft_eagle and not enpu
        runner_sync=guard and enpu
        assert not(wrapper_update and backend_impl_update)
        assert not(wrapper_sync and runner_sync)
        valuations.append(dict(full=full,replay=replay,capturing=capturing,sparse=sparse,compress=compress,enpu=enpu,
            updatable=updatable,draft_eagle=draft_eagle,wrapper_update=wrapper_update,backend_impl_update=backend_impl_update,
            wrapper_sync=wrapper_sync,runner_sync=runner_sync))
    assert len(valuations)==256
    write(O/'M22-update-and-sync-mutual-exclusion-proof-v45.json',dict(generated_utc=stamp,grade='DERIVED_BOOLEAN_SOURCE_DOMAINS',
        conditions=['One runner call/one active ordinary wrapper;matching FULL semantics,stable common backend/predicate,unchanged common ENPU flag.',
                    'Truth table includes unreachable boolean combinations; identities hold over all and therefore over reachable subset.'],
        valuations=valuations,boolean_valuations=256,both_update_paths_selected=0,both_explicit_syncs_selected=0,
        scope_exclusions=['Not a count of real calls,wait time or nested wrappers.','Does not exclude other backend/native/device/event synchronization.',
                          'No inference actual ENPU/backend value or native update epoch semantics.'],
        implementation=False,native_validation=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))

    old_variants=read(O/'M22-graph-generation-conditional-variants-v44.json')
    conditions=[
        ('NAMESPACE','Main,draft,draft-prefill namespaces and phase/owner/capture generation are part of reuse eligibility; num_tokens alone is insufficient.'),
        ('BUCKET_COMPLETE','Workspace-key presence alone does not initialize events/handles/attention params or graph entry; require all actual consumer planes.'),
        ('INITIALIZE','Respect one-time setters and no-op pre-init updates; no unreviewed runtime bucket/config reset or dropped publication.'),
        ('ALIAS','Distinct Python containers do not prove different Tensor storage/native handles; actual alias and last-reader epochs must be bound.'),
        ('LAYERS','Mutable event/handle/attn-param lists preserve captured layer/task pairing and order,not just length or equal empty contents.'),
        ('WEAKREF','Weak workspace/output ownership protocol must survive all uses; weak-ref conversion is not an eliminated copy/allocation.'),
        ('CONFIG','Backend impl selection retains the correct current vllm config/source context; no method-lookup hoist without stable owner/purity.'),
        ('UPDATE_ALTERNATIVES','Common stable backend makes wrapper-updatable and backend-impl updates alternative; never sum duplicate update savings.'),
        ('SYNC_ALTERNATIVES','Common ENPU flag makes these wrapper/runner explicit syncs alternative; do not extrapolate to all nested/native waits.'),
        ('NATIVE_EPOCH','All native parameter update epochs,CPU host completion and device ready/last-reader fences from v44 remain necessary.'),
        ('MEMORY','Initializer list/dict slots/None values are not GPU workspace allocations; price actual retained Tensor/native memory and aliases.'),
        ('ACTUAL_CASE','No live registry/ENPU/backend/handle snapshot has been obtained; conditional source proof is not actual dynamic branch attestation.'),
        ('PREDICATE','Imported use_updatable_graph local definition remains deployment-unattested this turn; do not infer current eligibility from class/name alone.'),
        ('NO_GAIN','No causal current wait/memory/latency savings are inferred from helper names,boolean valuations or registry key counts.'),
    ]
    write(O/'M22-registry-refined-generation-conditions-v45.json',dict(generated_utc=stamp,
        refined_existing_variant_ids=[v['id']for v in old_variants['variants']],new_variants=[],
        refinements=[dict(id='M22-v45-'+k,requirement=v,actual_current_legality_proved=False)for k,v in conditions],
        implementation=False,native_validation=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))
    g=graph(['namespace_config_capture_generation','all_bucket_planes_initialized','captured_task_layer_handle_pairing','workspace_owner_and_ready','native_update_or_graph_update','graph_replay_and_all_readers','generation_retirement'],[
        ('namespace_config_capture_generation','all_bucket_planes_initialized','ONE_TIME_OR_EXPLICIT_NEW_GENERATION'),
        ('all_bucket_planes_initialized','captured_task_layer_handle_pairing','NOT_WORKSPACE_KEY_ONLY'),
        ('captured_task_layer_handle_pairing','workspace_owner_and_ready','CORRECT_REFERENCE_ALIAS_LIFETIME'),
        ('workspace_owner_and_ready','native_update_or_graph_update','MUTUALLY_EXCLUSIVE_UPDATE_PROTOCOL'),
        ('native_update_or_graph_update','graph_replay_and_all_readers','PRESERVE_NATIVE_EPOCH_AND_CPU_GPU_FENCES'),
        ('graph_replay_and_all_readers','generation_retirement','ALL_LAST_READERS_BEFORE_REUSE')])
    write(O/'M22-registry-generation-obligations-v45.json',dict(generated_utc=stamp,conditional_graph=g,graphs_acyclic=True,
        implementation=False,whole_module_closed=False,overall_complete=False))

    evidence=['M22-registry-guard-crosswalk-v45.json','M22-registry-and-update-alternative-contract-v45.json',
        'M22-registry-key-completeness-and-alias-proof-v45.json','M22-update-and-sync-mutual-exclusion-proof-v45.json',
        'M22-registry-refined-generation-conditions-v45.json']
    for s in sem['static_branch_sites']:
        if s['semantic_family']=='M22-S02':s.update(review_status='EXACT_GUARD_LOCAL_DATAFLOW_RECONCILED',runtime_binding='SOURCE_CONDITIONS_NOT_ACTUAL_BANK_OR_HANDLE_GENERATIONS',semantic_evidence=evidence)
    for b in sem['branches']:
        if b['id']=='M22-S02':b.update(state='ALL_28_REGISTRY_WORKSPACE_PROXY_GUARDS_LOCALLY_ANALYZED',
            gap='Finite graph-file helper source cases are answered. Actual registry contents/aliases/native handles/event order and all consumers remain S12/S13/S16; original worker scope remains elsewhere.',
            evidence=list(dict.fromkeys(b['evidence']+evidence)))
        if b['id']in ['M22-S12','M22-S13','M22-S14','M22-S16']:b['evidence']=list(dict.fromkeys(b['evidence']+evidence))
    sem.update(generated_utc=stamp,parent_inventory='M22-semantic-branch-inventory-v44.json')
    assert len(sem['branches'])==16 and len(sem['static_branch_sites'])==1101
    write(O/'M22-semantic-branch-inventory-v45.json',sem)
    scope=deepcopy(prior);scope.update(generated_utc=stamp,parent_scope='M22-original-scope-and-guard-reconciliation-v44.json',
        guard_dataflow_reconciled=merged,local_reconciled_sites=69,graphfile_reconciled_static_sites=55,
        remaining_graphfile_helper_sites=0,other_static_sites_not_automatically_proved=1032)
    write(O/'M22-original-scope-and-guard-reconciliation-v45.json',scope)
    audit=deepcopy(read(O/'M22-original-requirement-completion-audit-v44.json'))
    audit.update(generated_utc=stamp,parent_audit='M22-original-requirement-completion-audit-v44.json',locally_reconciled_guard_sites=69,
        graphfile_static_guard_sites_reconciled=55,remaining_graphfile_helper_sites=0,result='NOT_COMPLETE')
    for r in audit['requirements']:
        r['evidence']=list(dict.fromkeys(r['evidence']+evidence+['M22-semantic-branch-inventory-v45.json']))
        r['completion_proved']=False
        r['note']='All55graph-file static guard source cases plus14runner core guards locally analyzed; actual native bank/event/handle epochs,worker relevant-case scope,phase/resource/process coverage remain. Conditional implementation is not a completion gate.'
    write(O/'M22-original-requirement-completion-audit-v45.json',audit)
    print(json.dumps(dict(new_helper_guards=28,total_core_guard_dataflows=69,all_graphfile_static_guards=55,remaining_graphfile_helper_sites=0,
        boolean_valuations=256,new_variants=0,refined_existing_variants=3,condition_refinements=14,
        original_static_sites=1101,semantic_families=16,original_gates_open=8,whole_module_closed=False,overall_complete=False)))


if __name__=='__main__':main()
