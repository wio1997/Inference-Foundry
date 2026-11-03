"""Opt-in native ConstantPadNd capture metadata probe;default mode is CPU-only. Default mode imports no accelerator packages.

Run mode requires a separately authorized idle device/window. It creates only its own
small graph/buffers,never replays the graph,never reads tensor contents and never sets
or disables graph tasks. This is a capability probe,not evidence about the saved model.
"""
from pathlib import Path
import argparse,ctypes,hashlib,json,platform,sys

HEADER='/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h'
HEADER_SHA='62d127fbb899b1476e054b1815ca6e0c71a362868c655f0e3da4616fb7598fb7'
LIB='/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so'
LIB_SHA='1c06b9c4a8910cfb585eab4ed0066bc09e28a0cf273dfcb15458f7cc49a78cea'
PROVIDER_CONTROLS=[{'metadata_path': '/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_legacy/mem_set/MemSet_1a6864193b99ef93ef38616f04a712ab_high_performance.json', 'metadata_sha256': '63edcf30b018827e1f3a58020bbed39bf6175dc55ebf24d78bd8cbffb75b765e', 'binary_path': '/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_legacy/mem_set/MemSet_1a6864193b99ef93ef38616f04a712ab_high_performance.o', 'binary_sha256': '523221b70c2e46ce249123a2d1aae7c94c1980d2b021143c7ea5aca2244a5adb', 'metadata_embedded_binary_sha256': '523221b70c2e46ce249123a2d1aae7c94c1980d2b021143c7ea5aca2244a5adb'}, {'metadata_path': '/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_legacy/pad_v3/PadV3_04c3cea8f4757124da4cd0789ca1ae79_high_performance.json', 'metadata_sha256': 'cdd90c2accfb22403e17021de7d6dc95fb5289f0526e2c2df58e46295a446309', 'binary_path': '/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_legacy/pad_v3/PadV3_04c3cea8f4757124da4cd0789ca1ae79_high_performance.o', 'binary_sha256': '08f452c94ca734e80da9cb669a2e8f64d15b109df6ebc14c3655d058a3da885a', 'metadata_embedded_binary_sha256': '08f452c94ca734e80da9cb669a2e8f64d15b109df6ebc14c3655d058a3da885a'}, {'metadata_path': '/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_legacy/pad_v3/PadV3_f4660b375c589d51c1f0e17818182091_high_performance.json', 'metadata_sha256': '80f946f9d7e076c773454e654b1a67c5486e92cd9227f1cbf9e7f8430d21d110', 'binary_path': '/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_legacy/pad_v3/PadV3_f4660b375c589d51c1f0e17818182091_high_performance.o', 'binary_sha256': '2ad441b088759e2bf40f15c673e23977da2942678762190d5419157dcbe84f7d', 'metadata_embedded_binary_sha256': '2ad441b088759e2bf40f15c673e23977da2942678762190d5419157dcbe84f7d'}]
GETTERS=['aclmdlRICaptureGetInfo','aclmdlRIGetStreams','aclmdlRIGetTasksByStream',
         'aclmdlRITaskGetType','aclmdlRITaskGetSeqId','aclmdlRITaskGetParams']

class KernelParams(ctypes.Structure):
    _fields_=[('funcHandle',ctypes.c_void_p),('cfg',ctypes.c_void_p),('args',ctypes.c_void_p),
        ('isHostArgs',ctypes.c_uint32),('argsSize',ctypes.c_size_t),('numBlocks',ctypes.c_uint32),('rsv',ctypes.c_uint32*10)]
class Payload(ctypes.Union):
    _fields_=[('reserved',ctypes.c_uint8*128),('kernel',KernelParams)]
class TaskParams(ctypes.Structure):
    _fields_=[('type',ctypes.c_int),('rsv0',ctypes.c_uint32*3),('taskGrp',ctypes.c_void_p),
        ('opInfoPtr',ctypes.c_void_p),('opInfoSize',ctypes.c_size_t),('rsv1',ctypes.c_uint8*32),('payload',Payload)]

def describe():
    return dict(status='PREPARED_NOT_EXECUTED',purpose='Reproduce the exact BOOL20→24 ConstantPadNd operator family seen in saved SP_MASK calls,not the different byte/D32 MEMCPY route.',
        SDK_header=HEADER,SDK_header_sha256=HEADER_SHA,provider_controls=PROVIDER_CONTROLS,
        own_application_input_output_bytes=44,driver_context_allocator_internal_casts_and_workspaces_not_bounded_by_44=True,
        controlled_operation=dict(operator='torch.ops.aten.constant_pad_nd',input_shape=[20],input_dtype='BOOL',padding=[0,4],value=True,output_shape=[24]),
        graph_replay_calls=0,model_loads=0,tensor_payload_copied_to_host=0,opaque_device_args_dereferenced=False,
        graph_task_mutator_calls=0,query_stream_limit=8,query_task_limit_per_stream=64,
        next_authorization='New single native-Pad diagnostic approval andfresh idle-device guard required before --run;previous minimal probe already finished.',
        output_does_not_automatically_close_historical_model_operands=True)

def bind(lib,name,args):
    f=getattr(lib,name);f.argtypes=args;f.restype=ctypes.c_int;return f
def require(rc,where):
    if rc!=0:raise RuntimeError(f'{where}:returncode={rc}')
def enumerate_handles(function,owner,maximum):
    n=ctypes.c_uint32(0);require(function(owner,None,ctypes.byref(n)),'enumerate count')
    if n.value>maximum:raise RuntimeError('enumeration exceeds fixed budget')
    if not n.value:return []
    capacity=n.value;values=(ctypes.c_void_p*capacity)();require(function(owner,values,ctypes.byref(n)),'enumerate values')
    if n.value>capacity:raise RuntimeError('enumeration count changed')
    return [values[i]for i in range(n.value)]

def run(args):
    if sys.platform!='linux' or platform.machine()not in ['aarch64','arm64']:
        raise RuntimeError('Run mode is restricted to the pinned Linux AArch64 diagnostic environment')
    if args.device is None or not 0<=args.device<16 or args.output is None:
        raise RuntimeError('--run requires explicit --device 0..15 and a new --output directory')
    output=Path(args.output).resolve()
    if output.parent != Path('/tmp') or not output.name.startswith('wio-native-pad-v111-') or output.exists():
        raise RuntimeError('Output must be a new independent /tmp/wio-native-pad-v111-* directory')
    if hashlib.sha256(Path(HEADER).read_bytes()).hexdigest()!=HEADER_SHA:
        raise RuntimeError('Installed header differs;do not guess ABI')
    if ctypes.sizeof(ctypes.c_void_p)!=8 or ctypes.sizeof(ctypes.c_size_t)!=8 or ctypes.sizeof(KernelParams)!=88 or ctypes.sizeof(TaskParams)!=200:
        raise RuntimeError('LP64 reference layout mismatch')
    if hashlib.sha256(Path(LIB).read_bytes()).hexdigest()!=LIB_SHA:
        raise RuntimeError('Pinned CANN library differs;stop before library/device')
    for control in PROVIDER_CONTROLS:
        if hashlib.sha256(Path(control['metadata_path']).read_bytes()).hexdigest()!=control['metadata_sha256']:
            raise RuntimeError('Pinned native provider metadata differs;stop before library/device')
        if hashlib.sha256(Path(control['binary_path']).read_bytes()).hexdigest()!=control['binary_sha256']:
            raise RuntimeError('Pinned native provider binary differs;stop before library/device')
    output.mkdir(parents=True,exist_ok=False);result=dict(plan=describe(),status='STOPPED_BEFORE_CAPTURE',device=args.device,library_disk_sha256=LIB_SHA)
    try:
        # This library loading and every accelerator operation occur only in opt-in run mode.
        lib=ctypes.CDLL(LIB)
        needed=GETTERS
        missing=[n for n in needed if not hasattr(lib,n)]
        if missing:
            result.update(status='UNSUPPORTED_EXPORTED_SYMBOLS',missing=missing);return
        ptr=ctypes.c_void_p;u32=ctypes.c_uint32
        info=bind(lib,'aclmdlRICaptureGetInfo',[ptr,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ptr)])
        streams=bind(lib,'aclmdlRIGetStreams',[ptr,ctypes.POINTER(ptr),ctypes.POINTER(u32)])
        tasks=bind(lib,'aclmdlRIGetTasksByStream',[ptr,ctypes.POINTER(ptr),ctypes.POINTER(u32)])
        kind=bind(lib,'aclmdlRITaskGetType',[ptr,ctypes.POINTER(ctypes.c_int)])
        seq=bind(lib,'aclmdlRITaskGetSeqId',[ptr,ctypes.POINTER(u32)])
        params=bind(lib,'aclmdlRITaskGetParams',[ptr,ctypes.POINTER(TaskParams)])
        import torch
        import torch_npu
        torch.npu.set_device(args.device)
        bool_input=torch.empty(20,dtype=torch.bool,device=f'npu:{args.device}')
        graph=torch_npu.npu.NPUGraph();model=ptr();capture_status=ctypes.c_int(-1)
        # No replay is required to ask whether captured metadata is available.
        with torch_npu.npu.graph(graph):
            stream=ptr(torch.npu.current_stream().npu_stream)
            bool_output=torch.ops.aten.constant_pad_nd(bool_input,[0,4],True)
            require(info(stream,ctypes.byref(capture_status),ctypes.byref(model)),'capture info')
        if not model.value:raise RuntimeError('Capture getter did not return a modelRI handle')
        rows=[];stream_handles=enumerate_handles(streams,model,8)
        for stream_handle in stream_handles:
            for task in enumerate_handles(tasks,ptr(stream_handle),64):
                t=ctypes.c_int(-1);s=u32(0);require(kind(ptr(task),ctypes.byref(t)),'task type');require(seq(ptr(task),ctypes.byref(s)),'task seq')
                p=TaskParams();rc=params(ptr(task),ctypes.byref(p))
                row=dict(stream_handle=stream_handle,task_handle=task,type=t.value,seq_id=s.value,get_params_returncode=rc)
                if rc==0:
                    row.update(params_type=p.type,opInfo_address=p.opInfoPtr,opInfoSize=p.opInfoSize)
                    if p.type!=t.value:raise RuntimeError('Getter task type mismatch')
                    if p.type==1:
                        k=p.payload.kernel
                        row.update(kernel_function_address=k.funcHandle,args_address=k.args,argsSize=k.argsSize,
                            isHostArgs=k.isHostArgs,numBlocks=k.numBlocks,args_bytes_dereferenced=False)
                rows.append(row)
        result.update(status='CAPTURED_METADATA_QUERIED_NO_REPLAY',capture_status=capture_status.value,modelRI_handle=model.value,
            bool_input_address=bool_input.data_ptr(),bool_output_address=bool_output.data_ptr(),tasks=rows,
            typed_MemSet_destination_or_length_decoded=False,saved_model_correlation=False)
        if args.dump_own_graph:
            path=output/'own-native-pad-graph.json';assert not path.exists();graph.debug_dump(str(path))
            result['debug_dump_generated']=path.is_file()
            if path.is_file():
                if path.stat().st_size>4*1024*1024:raise RuntimeError('Own graph dump exceeds fixed output budget')
                result['debug_dump_bytes']=path.stat().st_size
                result['debug_dump_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception as e:
        result.update(status='STOPPED_UNSUPPORTED_OR_ERROR',error_type=type(e).__name__,error=str(e))
    finally:
        # Raw args/device contents are deliberately never dereferenced or copied here.
        (output/'probe-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(status=result['status'],output=str(output/'probe-result.json'),graph_replayed=False,model_loaded=False)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--device',type=int)
    parser.add_argument('--output');parser.add_argument('--dump-own-graph',action='store_true');args=parser.parse_args()
    if not args.run:print(json.dumps(describe(),ensure_ascii=False));return
    run(args)

if __name__=='__main__':main()
