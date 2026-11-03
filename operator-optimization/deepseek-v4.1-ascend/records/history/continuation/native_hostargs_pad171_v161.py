"""Opt-in 69B Pad pair and CPU-profiler field check;default is CPU-only description."""
import argparse,json,hashlib,os,time
from pathlib import Path

HEADER='/usr/local/Ascend/cann-9.2.0/aarch64-linux/include/acl/acl_rt.h'
HEADER_SHA='62d127fbb899b1476e054b1815ca6e0c71a362868c655f0e3da4616fb7598fb7'
LIB='/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libascendcl.so'
LIB_SHA='1c06b9c4a8910cfb585eab4ed0066bc09e28a0cf273dfcb15458f7cc49a78cea'
ADAPTER_SHA='4bd42a98eb1a87662a48314bac94cfb0ee09577244e7e944d3462447c1c5cdb8'

def describe():
    return {'status':'PREPARED_NOT_EXECUTED','purpose':'Joint builtin-field and eager HostArgs capability check on one own Pad pair,not the original model or all runtime error paths','device':0,'input_owner_shape':[21],'input_view_shape':[20],'input_view_storage_offset':1,'output_shapes':[[24],[24]],'dtype':'BOOL','application_NPU_tensor_bytes':69,'driver_context_allocator_and_workspace_bytes_not_bounded_by_69':True,'host_test_output_bytes':48,'model_loads':0,'graph_captures':0,'graph_replays':0,'CPU_profiler_sessions':2,'profiler_NPU_activity_or_hardware_counters':False,'control_profile_memory':False,'observed_profile_memory':True,'profile_shapes_and_stack':True,'offline_profile_analysis_launched':False,'native_records_max':2,'host_packet_bytes_max':4096,'placeholder_count_max':32,'native_trace_bytes_max':65536,'all_diagnostic_output_guard_bytes':8388608,'process_timeout_seconds':45,'hard_kill_grace_seconds':5,'capture_scope':'OWN_EAGER_PAD_PAIR_ONLY','full_model_collector_ready':False,'fresh_specific_authorization_required':True,'internal_CANN_error_state_equivalence_not_proved_by_success_case':True}

def metadata(t):
    s=t.untyped_storage()
    return {'TensorImpl':int(t._cdata),'storage_impl':int(s._cdata),'storage_base':int(s.data_ptr()),'tensor_data_ptr':int(t.data_ptr()),'shape':list(t.shape),'stride':list(t.stride()),'storage_offset':t.storage_offset(),'storage_nbytes':s.nbytes(),'storage_extent_is_not_physical_HBM_allocation':True,'dtype':str(t.dtype),'device':str(t.device)}

def run(args):
    bundle=Path(args.bundle).resolve()
    assert bundle.parent==Path('/tmp') and bundle.name.startswith('wio-hostargs-pad-v161-')
    approval=json.loads((bundle/'APPROVED.json').read_text())
    assert approval['status']=='AUTHORIZED_ONCE' and approval['bundle']==str(bundle)
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==approval['probe_source_sha256']
    assert os.environ.get('WIO_NATIVE_PROBE_ACK')==approval['approval_id']
    for name in ['ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_DEVICE_ID']:assert os.environ.get(name)=='0'
    expected={'WIO_HOST_TRACE_ENABLE':'1','WIO_HOST_TRACE_MAX_RECORDS':'2','WIO_HOST_TRACE_MAX_BYTES':'65536','WIO_HOST_TRACE_MAX_SECONDS':'60','WIO_HOST_TRACE_QUERY_IDS':'0'}
    for k,v in expected.items():assert os.environ.get(k)==v,(k,'configuration differs')
    assert hashlib.sha256(Path(HEADER).read_bytes()).hexdigest()==HEADER_SHA
    assert hashlib.sha256(Path(LIB).read_bytes()).hexdigest()==LIB_SHA
    adapter=bundle/'libhost_adapter.so';assert hashlib.sha256(adapter.read_bytes()).hexdigest()==ADAPTER_SHA
    arm=bundle/'armed';assert os.environ['WIO_HOST_TRACE_ARM_FILE']==str(arm) and not arm.exists()
    result={'plan':describe(),'status':'STARTED','pid':os.getpid(),'started_unix':time.time(),'graph_captures':0,'graph_replays':0,'model_loads':0,'original_model_native_operands_recovered':0}
    try:
        import ctypes
        # Both library loading and all accelerator imports occur only after the explicit run/approval guards.
        backend=ctypes.CDLL(LIB)
        observer=ctypes.CDLL(str(adapter));handshake=observer.wio_host_trace_probe;handshake.argtypes=[];handshake.restype=ctypes.c_uint
        flags=handshake();result['resolver_flags']=int(flags);assert flags&7==7,'No real already-loaded backend/name getter;stop before tensor allocation'
        import torch
        import torch_npu
        torch.npu.set_device(0)
        pattern=[i%2==0 for i in range(20)];wanted=pattern+[True]*4
        owner=torch.tensor([False]+pattern,dtype=torch.bool,device='npu:0');x=owner[1:]
        assert x.storage_offset()==1 and x.untyped_storage().data_ptr()==owner.untyped_storage().data_ptr()
        def profile(label,memory):
            handler=torch_npu.profiler.tensorboard_trace_handler(str(bundle/'profiles'/label),worker_name='pad_'+label,analyse_flag=False)
            return torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU],record_shapes=True,with_stack=True,with_modules=False,profile_memory=memory,on_trace_ready=handler)
        with profile('control',False):
            y0=torch.ops.aten.constant_pad_nd(x,[0,4],True)
            control=y0.cpu().tolist()
        assert control==wanted
        result['control_unarmed_complete']=True
        with profile('observed',True):
            arm.touch(exist_ok=False)
            try:
                y1=torch.ops.aten.constant_pad_nd(x,[0,4],True)
                observed=y1.cpu().tolist()
            finally:arm.unlink(missing_ok=True)
        assert observed==wanted==control
        result.update(status='EAGER_CONTROL_OBSERVED_VALUES_MATCH',input_owner=metadata(owner),input=metadata(x),control_output=metadata(y0),observed_output=metadata(y1),expected_values=wanted,control_values=control,observed_values=observed,source_argv=list(__import__('sys').argv),finished_unix=time.time(),CPU_profile_directories=[str(bundle/'profiles'/'control'),str(bundle/'profiles'/'observed')],only_success_path_forwarding_exercised=True)
    except BaseException as e:
        result.update(status='ERROR',error_type=type(e).__name__,error=str(e),finished_unix=time.time())
        raise
    finally:
        arm.unlink(missing_ok=True)
        raw=json.dumps(result,ensure_ascii=False,indent=2).encode();assert len(raw)<=32768
        (bundle/'probe-result.json').write_bytes(raw)
        print(json.dumps({'status':result['status'],'result':str(bundle/'probe-result.json')}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--bundle');a=p.parse_args()
    if not a.run:print(json.dumps(describe(),ensure_ascii=False));return
    assert a.bundle;run(a)
if __name__=='__main__':main()
