#!/usr/bin/env python3
"""CPU-only synthetic tests; does not import torch or execute NPU preflight."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from loop078_count_markers_analyze import record, analyze
from loop078_count_markers_patch import SOURCES, patch


def fixture():
    source=dict(ptr=100,storage_ptr=100,storage_bytes=48,offset=0,shape=[12],stride=[1],
                dtype='torch.int32',device='npu:0',nbytes=48)
    labels=['R_BEGIN_64','R_DONE_64','W_PRE_65','W_POST_65']
    markers={k:dict(event=i+1,stream=2 if i<2 else 3,device='npu:0',
        entry=64 if i<2 else 65,generation=64,storage=source) for i,k in enumerate(labels)}
    ordered=['commit_entry','sync_pre','sync_post','count_add_pre','count_add_post',
             'seq_add_pre','seq_add_post','mirrors_done','launch65','downstream_pre','downstream_post']
    host={k:dict(ns=(i+2)*100,generation=65 if k=='launch65' else 64,entry=65,ptr=None)
          for i,k in enumerate(ordered)}
    host['count_add_pre']['ptr']=200;host['seq_add_pre']['ptr']=300
    host['downstream_pre']['ptr']=300
    host['clone_pre']=dict(ns=None,ptr=None)
    host['clone_post']=dict(ns=None,ptr=None)
    host['launch64']=dict(ns=100,generation=64,entry=64)
    host['progress64']=dict(ns=150,generation=63,entry=64)
    host['progress65']=dict(ns=1400,generation=64,entry=65)
    return dict(torch_version='synthetic',torch_npu_version='synthetic',commit_pending=True,schema=2,rank=0,cohort=1,phase='warmup',run_id='B',generation_error=None,
        parking=False,reason='normal_proposer',draft_graph=False,cycles=66,
        accepted_counts=[[2]*12 for _ in range(66)],remaining=[1024]*12,initial_output_counts=[0]*12,
        device_records=markers,production_event=99,copy_stream=2,source=source,
        destination=dict(device='cpu',ptr=200),host=host,seq_mirrors=[dict(ptr=300)],
        first_clone=None,downstream=dict(lineage='direct_mirror',site='dsa_v1.build_prefill_metadata.max_item',pointer=300),before_anchor_ms=dict(zip(labels,[10,9,8,7])),
        direct=dict(margin=dict(ms=1)))


def main():
    t=dict(direct_uncertainty_ms=.01,anchor_uncertainty_ms=.02,local_marker_allowance_ms=.03,
        same_device_cross_stream_validated=True,d2h_completion_host_visibility_validated=True,
        lazy_event_preflight_passed=True)
    row=fixture(); assert record(row,t)['observed_ordering']
    checks=['normal_zero_initial_output_passes']
    from types import SimpleNamespace
    import loop078_count_markers as hooks
    class Tensor:
        device=SimpleNamespace(type='cpu')
        def __init__(self,ptr):self.ptr=ptr
        def data_ptr(self):return self.ptr
    hooks.D=dict(entry=65,commit_generation=64,mirror_ptrs=[300],derived_ptr=None,
        first_clone=None,downstream=None,host={k:dict(ns=None,ptr=None,generation=None,entry=None) for k in hooks.HOST})
    selected=hooks.clone_before(Tensor(300),'utils.clone')
    hooks.clone_after(selected,Tensor(400))
    assert hooks.D['downstream'] is None and hooks.D['host']['downstream_pre']['ns'] is None
    assert hooks.D['first_clone']['source_pointer']==300 and hooks.D['first_clone']['output_pointer']==400
    selected=hooks.downstream_before(Tensor(400),'utils.add_')
    hooks.downstream_after(selected)
    assert selected and hooks.D['downstream']['lineage']=='tracked_clone'
    assert hooks.D['host']['clone_post']['ns'] <= hooks.D['host']['downstream_pre']['ns']
    assert not hooks.downstream_before(Tensor(400),'dsa.max_item')
    hooks.D=None
    checks.append('clone_does_not_suppress_first_numeric_consumer')
    clone_only=copy.deepcopy(row)
    clone_only['first_clone']=dict(source_pointer=300,output_pointer=400,lineage='direct_mirror',site='utils.clone')
    clone_only['host']['clone_pre']=dict(ns=950,ptr=300)
    clone_only['host']['clone_post']=dict(ns=970,ptr=400)
    clone_only['downstream']=None
    clone_only['host']['downstream_pre']['ns']=None
    clone_only['host']['downstream_post']['ns']=None
    assert not record(clone_only,t)['downstream_observed']
    checks.append('clone_only_is_not_downstream_observed')
    negative=copy.deepcopy(row);negative['direct']['margin']['ms']=-1
    negative['before_anchor_ms'].update(R_BEGIN_64=10,R_DONE_64=8,W_PRE_65=9,W_POST_65=7)
    assert record(negative,t)['observed_ordering'] is False
    checks.append('negative_margin_is_inconclusive_not_race')
    def reject(name,change):
        r=copy.deepcopy(row);change(r)
        try:record(r,t)
        except (ValueError,KeyError,TypeError):checks.append(name)
        else:raise AssertionError(name)
    reject('wrong_copy_stream',lambda r:r.__setitem__('copy_stream',4))
    reject('missing_consumer',lambda r:r['host']['count_add_pre'].__setitem__('ns',None))
    reject('bad_generation',lambda r:r.__setitem__('generation_error','mismatch'))
    reject('event_reuse',lambda r:r['device_records']['W_PRE_65'].__setitem__('event',1))
    reject('anchor_disagreement',lambda r:r['direct']['margin'].__setitem__('ms',5))
    reject('clone_mislabeled_numeric',lambda r:r['downstream'].__setitem__('site','utils.clone'))
    reject('parking',lambda r:r.__setitem__('parking',True))
    with tempfile.TemporaryDirectory() as tmp:
        proof=Path(tmp)/'timing.json'
        script=Path(__file__).with_name('loop078_count_markers_preflight.py')
        proof.write_text(json.dumps(dict(torch_version='synthetic',torch_npu_version='synthetic',passed=True,devices=list(range(8)),timing=t,
            producer_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())))
        ref=dict(path=str(proof),sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
        controls={name:dict(http_posts=60,max_concurrency=12,outputs_per_request=1024,cohorts=5,
            errors=0,full_target=True,post_handoff_oracle_calls=0,host_mirror_exact=True,
            source_restored=True,lifecycle_closed=True,runtime_wall_s=10+i,
            request_ledger_sha256='synthetic',acceptance_sha256=name,run_id=name)
            for i,name in enumerate(('A0','B','A1'))}
        manifest=dict(timing_report=ref,evidence_files=[ref],controls=controls)
        rows=[]
        for c in range(1,6):
            for rank in range(8):
                r=copy.deepcopy(row);r.update(cohort=c,rank=rank,phase='warmup' if c<=4 else 'diagnostic');rows.append(r)
        result=analyze(rows,manifest)
        assert result['status']=='ACCEPTED' and not result['control_gate_passed']
        assert not result['original_schedule_extrapolation_supported'] and len(result['records'])==40
        checks.append('different_acceptance_preserves_local_ordering_only')
    for key,path in SOURCES.items():
        original=path.read_text();edited=patch(key,original);compile(edited,str(path),'exec')
        for token in ('synchronize(', 'wait_stream(', 'wait_event(', 'non_blocking=True'):
            assert original.count(token)==edited.count(token),(key,token)
    checks.append('generated_sources_preserve_existing_waits_and_copy')
    print(json.dumps(dict(passed=True,checks=checks)))


if __name__=='__main__':main()
