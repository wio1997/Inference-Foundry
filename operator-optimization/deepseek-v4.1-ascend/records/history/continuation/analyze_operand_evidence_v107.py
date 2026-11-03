"""Read-only evidence acquisition limits for unresolved operator operands; CPU only."""
from pathlib import Path
import hashlib,json,subprocess,sys
from finalize_origin_coverage_v9 import N,O,read,write

def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def segment(path,first,last,question):
    return dict(**pin(path),first_line=first,last_line=last,question=question,
                source=Path(path).read_text().splitlines()[first-1:last])

def fill_extent(provider,capacity,count,offset=0):
    """Declared API-unit reference, not a decoder of opaque captured task args."""
    if provider not in ['aclrtMemsetAsync','aclrtMemsetD32Async']:
        return dict(state='UNKNOWN_PROVIDER',written_bytes=None)
    if any(type(x)is not int or x<0 for x in [capacity,count,offset]):
        return dict(state='INVALID_REFERENCE_DOMAIN',written_bytes=None)
    n=count*(4 if provider=='aclrtMemsetD32Async'else 1)
    if n>capacity or offset+capacity>2**64-1:
        return dict(state='INVALID_REFERENCE_DOMAIN',written_bytes=None)
    return dict(state='DECLARED_API_REFERENCE_ONLY',written_bytes=n,interval=[offset,offset+n],
                capacity_is_not_write_extent=True,native_task_provider_not_inferred=True)

def main():
    assert not(O/'REVIEW-CHECKPOINT-v107.json').exists()
    idx=read(O/'sdk-operand-source-index-v107.json')
    for r in idx['files']:
        assert r['status']=='READ'and pin(r['local_path'])['sha256']==r['sha256']
    header=Path(next(r['local_path']for r in idx['files']if r['path'].endswith('/acl_rt.h')))
    wrapper=Path(next(r['local_path']for r in idx['files']if r['path'].endswith('/graphs.py')))
    probe=N/'diagnostic-operand-probe-v107/inspect_graph_params.py'
    pivot=read(O/'pivot-filter-source-contract-v92.json')
    assert pin(pivot['path'])['sha256']==pivot['sha256']=='0b5fc337058e22bc28a8530ccec904d6ccd89b9548b02955c657ac429b1669d2'
    inputs=[O/'sdk-operand-source-index-v107.json',N/'sdk-operand-source166-v107.json',
        O/'pivot-filter-source-contract-v92.json',O/'sampling-filter-race-source-contract-v91.json',
        O/'minimal-operator-evidence-plan-v98.json',O/'coverage-current-obligations-v106.json',
        O/'module-phase-branch-matrix-v106.json',probe]+[Path(r['local_path'])for r in idx['files']]
    write(O/'operand-evidence-inputs-v107.json',dict(inputs=[pin(p)for p in inputs],new_device_work=False))
    parts=[segment(header,a,b,q)for a,b,q in [
        (595,613,'Does the task enum/union expose a typed MemSet range?'),
        (644,661,'Which opaque sizes/addresses are actually declared?'),
        (2343,2374,'What is the D32 count unit?'),
        (2540,2564,'What is the byte-fill count versus capacity unit?'),
        (3790,3800,'What does GetParams promise?'),
        (4765,4783,'Can an own capture handle be queried without object layout guessing?'),
        (4800,4815,'Which graph dump interface is declared?'),
        (5030,5075,'Which finite task/type/sequence enumeration getters exist?')]]
    parts+=[segment(wrapper,666,710,'How does the Python wrapper reject a stale graph dump?')]
    write(O/'operand-evidence-source-contract-v107.json',dict(parts=parts,
        header_matches_saved_v12=True,loaded_library_ABI_or_exported_symbols_attested=False,
        task_enum_has_dedicated_MemSet=False,kernel_argsSize_means_write_extent=False,
        opInfoSize_means_write_extent=False,opaque_args_dereferenced=False,
        LP64_reference_only=dict(pointer=8,size_t=8,enum=4,kernel_params=88,task_params=200),
        absence_of_Python_raw_getter_does_not_prove_Cpp_base_absence=True,
        source_file_identity_does_not_attest_loaded_process_identity=True,
        no_framework_wide_research=True,overall_complete=False))
    methods=[
        dict(method='Saved profile own attrs/shapes',result='INSUFFICIENT_FOR_NINE_PENDING_REGIMES',
             can_establish='v98 has empty/N-A MemSet own attrs and static BOOL geometry; enclosing ConstantPadNd owner already bound.',
             cannot_establish='MemSet dst/write extent/overlap/role;BOOL field and all readers.'),
        dict(method='record_op_args / profile_memory',result='NO_DOCUMENTED_GUARANTEE_OF_REQUIRED_FIELDS',
             can_establish='Configured operator-argument statistics or allocation/lifecycle evidence,depending on actual output.',
             cannot_establish='Enabling a switch alone does not prove per-task native destination/count or exact bytes written.'),
        dict(method='Installed aclmdlRITaskGetParams',result='DECLARED_GENERIC_GETTER;RUNTIME_CAPABILITY_UNTESTED',
             can_establish='If supported:task type,opaque packet address/size and kernel metadata for a selected owned graph.',
             cannot_establish='No dedicated typed MemSet union;argsSize/opInfoSize are not the fill length;no assumed opaque ABI decode.'),
        dict(method='Own tiny capture metadata probe',result='PREPARED_NOT_EXECUTED;APPROVAL_PENDING',
             can_establish='A bounded test of getter/dump metadata on controlled own calls,if explicitly approved and an idle device is available.',
             cannot_establish='It cannot identify the saved model\'s operands,field/readers,loaded kernel ABI or same-call receive counts.'),
        dict(method='Same-call model operand/identity diagnostic',result='SEPARATE_REQUIRED_EVIDENCE;NOT_AUTHORIZED_OR_EXECUTED_HERE',
             can_establish='Only with exact call/native-task/graph/replay/update/storage-epoch joins can typed ranges or field identity answer the original pending question.',
             cannot_establish='A new run or coincident task name/address alone does not prove identity of a historical saved call.')]
    write(O/'operand-evidence-method-gates-v107.json',dict(methods=methods,
        acceptance_contract=dict(identity=['process/rank/device','loaded implementation/provider ABI','capture graph/modelRI and parameter/update epoch',
            'native stream/task/sequence plus replay or call correlation','storage object/version/lifetime'],
            MemSet=['verified provider/count unit','destination interval','actual write bytes,not capacity or argument-packet bytes',
                'overlap/old-reader/new-writer order','workspace versus demanded payload role'],
            BOOL24=['semantic field','storage epoch','all read/mutation observers and fallback flags'],
            MC2=['same-call expert_token_nums or complete all-rank bound','capacity C for that call','loaded ABI/write range/lifetimes'],
            same_call_counts_do_not_need_tensor_dump=True),
        consequences=dict(no_delete_merge_or_shrink_promoted=True,MC2_C_not_processed_rows=True,
            capture_not_replay=True,task_id_without_graph_epoch_insufficient=True),
        official_supplemental_docs=[
            dict(version='9.1.0',url='https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0033.html'),
            dict(version='9.1.0',url='https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_2180.html')],
        public_document_version_not_installed_runtime_attestation=True,overall_complete=False))
    refs=[]
    for provider,capacity,count,offset in [('aclrtMemsetAsync',8192,64,0),('aclrtMemsetD32Async',7936,16,256),
        ('aclrtMemsetAsync',8192,8192,0),('aclrtMemsetD32Async',8192,2048,0),
        ('aclrtMemsetAsync',8192,0,0),('aclrtMemsetD32Async',8192,0,0),
        ('aclrtMemsetAsync',63,64,0),('aclrtMemsetD32Async',63,16,0),
        ('opaque-kernel',8192,16,0),('aclrtMemsetAsync',8192,-1,0),
        ('aclrtMemsetAsync',8192,64,2**64-1),('aclrtMemsetD32Async',8192,16,256)]:
        refs.append(dict(provider=provider,capacity_bytes=capacity,count=count,offset=offset,
                         reference=fill_extent(provider,capacity,count,offset)))
    description=json.loads(subprocess.check_output([sys.executable,str(probe)],text=True))
    write(O/'operand-evidence-plan-v107.json',dict(status='PREPARED_NOT_EXECUTED',probe=pin(probe),description=description,
        approval_state='PENDING_ASYNC_USER_REPLY;ELAPSED_TIME_IS_NOT_APPROVAL',
        runtime_guards=['Explicit affirmative new-device diagnostic authorization',
            'Read occupancy first;skip occupied devices;do not kill jobs/start a stopped model container',
            'Match exact header/library/environment and stop on unsupported getters',
            'Independent new output directory;bounded own graph/tasks;no replay/model load/opaque dereference'],
        application_payload_not_total_context_memory=True,API_unit_reference_cases=refs,
        unit_references_not_actual_native_extent_or_correlation_proof=True,overall_complete=False))
    write(O/'sampling-prior-answer-routing-v107.json',dict(
        corrected_rows=['M17-OP-04','M17-OP-07'],prior_claim='UpstreamTritonfilterbody remains unread',
        correction='Exact frozen upstream monolithic NPU source and local contract were reviewed in v92;do not reread as a new open body.',
        original_evidence='pivot-filter-source-contract-v92.json',source=pin(pivot['path']),
        function_ranges=[dict(name=f['name'],first_line=f['first_line'],last_line=f['last_line'])for f in pivot['functions']],
        NPU_route=pivot['NPU_route'],actual_runtime_flags_or_compilation_attested=False,
        still_open=['Genuine remaining processors/readers with a specific operator-changing question',
            'Actual flags/loaded native precision/RNG/parameter banks','Full demanded output and observer closure'],
        no_new_fusion_relation_or_numeric_ceiling=True,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED_OFFLINE',SDK_files=4,methods=len(methods),unit_reference_cases=len(refs),
        stale_branch_rows=2,device_probe_executed=False,new_native_promotions=0)))

if __name__=='__main__':main()
