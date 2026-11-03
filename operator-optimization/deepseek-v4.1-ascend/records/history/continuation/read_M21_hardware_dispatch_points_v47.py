"""Recover bounded missing dispatch predicates from local frozen server-source artifacts."""
import ast,hashlib
from datetime import datetime,timezone
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,R,W,read,write


def main():
    assert not(O/'REVIEW-CHECKPOINT-v47.json').exists()
    stamp=datetime.now(timezone.utc).isoformat()
    write(O/'PROFILE-CODE-QUESTION-GATE-v47.json',dict(status='PROFILE_FIRST_MISSING_HARDWARE_DISPATCH_PREDICATES',
        previous_turn_classification='PROGRESS:v46modeDP41guards',
        profile_facts=['Current snapshot has529named legacy SMLA/QLI metadata DIRECT calls;packed/legacy selection must not be guessed from shapes.',
            'v42 exact remote predicate read failed authentication;v46 established historical166stage utils5f88source SHA.',
            'Complete saved graph data remains on166;current existing-login/socket checks found no usable connection without new authentication.'],
        question='What does normal DeviceOperator selection return for the documented A3 build versus A5, and which former missing predicate definitions can be answered locally?',
        allowed_source=['Known-stage-pinned device_op base/A5 getter,get_device_adaptor andDeviceOperator binding only.',
            'Frozen hardware_profile only adapter-family enum/profile mapping entries/current binding/getters.',
            'Frozen device_config only DeviceConfig/build-info selector/getter;old saved build-info metadata explicitly historical.',
            'Historically166stage-pinned utils.enable_dsa_cp exact small function only.'],
        no_model_torch_NPU_service_health_work=True,no_broad_repository_or_SDK_audit=True,
        no_new_SSH_login=True,whole_module_closed=False,overall_complete=False))
    base=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';out=[]
    def capture(rel,functions,classes=(),assigns=(),pinned=None):
        p=base/rel;b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
        if pinned:assert sha==pinned
        t=ast.parse(b);lines=b.decode().splitlines();items=[]
        def item(node,kind,name):
            assert node.end_lineno-node.lineno+1<=45
            return dict(kind=kind,name=name,start=node.lineno,end=node.end_lineno,
                decorators=[ast.unparse(d)for d in getattr(node,'decorator_list',[])],
                source=[f'{i+1}: {lines[i]}'for i in range(node.lineno-1,node.end_lineno)])
        for owner,name in functions:
            scope=t.body if not owner else next(c for c in t.body if isinstance(c,ast.ClassDef)and c.name==owner).body
            f=next(x for x in scope if isinstance(x,ast.FunctionDef)and x.name==name);items.append(item(f,'function',owner+'.'+name if owner else name))
        for name in classes:
            c=next(c for c in t.body if isinstance(c,ast.ClassDef)and c.name==name);items.append(item(c,'class_fields',name))
        for x in t.body:
            if isinstance(x,(ast.Assign,ast.AnnAssign)):
                ts=x.targets if isinstance(x,ast.Assign)else[x.target]
                if any(isinstance(a,ast.Name)and a.id in assigns for a in ts):items.append(item(x,'binding',','.join(a.id for a in ts if isinstance(a,ast.Name))))
        result=dict(path=str(p),sha256=sha,items=items,source_grade='HISTORICALLY_STAGE_HASH_MATCH'if pinned else'DOCUMENTED_FROZEN_SOURCE;NO_FRESH_166_DEPENDENCY_HASH_ATTESTATION')
        out.append(result);return t,lines,result
    tree,lines,device=capture('device/device_op.py',[
        ('BaseDeviceAdaptor','get_dsv41_packed_cache_ops'),('A5DeviceAdaptor','get_dsv41_packed_cache_ops'),('','get_device_adaptor')],
        assigns=['DeviceOperator'],pinned='01ce13ea60e46f901ad69ad4c71e4be0e31cd69fbc56ebb7d898f957d5e54f67')
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='Ascend310PDeviceAdaptor')
    device['compatibility_inheritance']=dict(class_name=cls.name,bases=[ast.unparse(x)for x in cls.bases],
        overrides_packed_getter=any(isinstance(x,ast.FunctionDef)and x.name=='get_dsv41_packed_cache_ops'for x in cls.body))
    tree,lines,hardware=capture('device/hardware_profile.py',[
        ('','get_hardware_profile'),('','get_current_hardware_profile')],classes=['DeviceAdaptorFamily'],assigns=['_CURRENT_HARDWARE_PROFILE'])
    assignment=next(x for x in tree.body if isinstance(x,(ast.Assign,ast.AnnAssign))and
        any(isinstance(a,ast.Name)and a.id=='_HARDWARE_PROFILES'for a in(x.targets if isinstance(x,ast.Assign)else[x.target])))
    entries=[]
    for d in ast.walk(assignment.value):
        if isinstance(d,ast.Dict):
            for k,v in zip(d.keys,d.values):
                if isinstance(v,ast.Call):
                    field=next((z for z in v.keywords if z.arg=='device_adaptor_family'),None)
                    if field:
                        entries.append(dict(device_key=ast.unparse(k),adaptor_family=ast.unparse(field.value),
                            start=field.lineno,end=field.end_lineno,source=[f'{i+1}: {lines[i]}'for i in range(field.lineno-1,field.end_lineno)]))
    assert len(entries)==4;hardware['selected_profile_entries']=entries
    capture('device/device_config.py',[('','_device_type_from_build_info'),('','get_device_config')],classes=['DeviceConfig'])
    capture('utils.py',[('','enable_dsa_cp')],pinned='5f88fd83bcba539c3d24324ed104469914254989996342b27bb36a326e3f8458')
    old_build=R/'agents/fusion/frozen-source/vllm_ascend/_build_info.py'
    assert old_build.exists();b=old_build.read_bytes();tree=ast.parse(b);oldfields=[];lines=b.decode().splitlines()
    for x in tree.body:
        if isinstance(x,ast.Assign)and any(isinstance(a,ast.Name)and a.id in ['__device_type__','__soc_version__']for a in x.targets):
            oldfields.append(dict(start=x.lineno,end=x.end_lineno,source=[f'{i+1}: {lines[i]}'for i in range(x.lineno-1,x.end_lineno)],literal=ast.literal_eval(x.value)))
    assert any(f['literal']=='A3'for f in oldfields)
    write(O/'M21-hardware-adaptor-and-DSACP-source-v47.json',dict(generated_utc=stamp,files=out,
        historical_build_info=dict(path=str(old_build),sha256=hashlib.sha256(b).hexdigest(),fields=oldfields,
            provenance='Historical source169 frozen artifact;not fresh166generated build-info snapshot.'),
        current166_build_info_attested=False,no_model_or_torch_import=True,whole_module_closed=False,overall_complete=False))
    write(O/'M22-profile-access-revalidation-v47.json',dict(generated_utc=stamp,
        prior_success_receipt='M21-config166-v42.json',prior_auth_failure_receipt='M21-predicates166-v42.json',
        success_observed_utc=read(N/'M21-config166-v42.json')['observed_utc'],failure_observed_utc=read(N/'M21-predicates166-v42.json')['observed_utc'],
        helper_control_persist_seconds=600,keychain_metadata_last_modified='20261026065538Z',
        current166_control_socket_present=False,current166_matching_SSH_connection_found=False,
        fresh_password_retry_performed=False,credentials_value_read_or_logged=False,other_sessions_changed=False,
        interpretation='Cached connection expiry is consistent with roughly10minute success/failure gap;not proof of server-side cause. Current valid login information requested;independent local evidence work continues.',
        authentication_question_pending=True,remote_profile_read_executed=False,
        no_remote_model_device_service_health_work=True,whole_module_closed=False,overall_complete=False))
    print('Bounded adapter/profile/config/DSACP source recorded;historical A3build explicitlynotfresh166attestation;no new login.')
    for f in out:
        for x in f['items']:
            if x['name']=='enable_dsa_cp':print('\n'.join(x['source']))


if __name__=='__main__':main()
