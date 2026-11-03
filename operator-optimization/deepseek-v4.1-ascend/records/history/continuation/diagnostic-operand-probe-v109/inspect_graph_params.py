"""Opt-in,model-free capture metadata probe. Default mode imports no accelerator packages.

Run mode requires a separately authorized idle device/window. It creates only its own
small graph/buffers,never replays the graph,never reads tensor contents and never sets
or disables graph tasks. This is a capability probe,not evidence about the saved model.
"""
from pathlib import Path
import argparse,ctypes,hashlib,json,platform,sys

HEADER='/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h'
HEADER_SHA='62d127fbb899b1476e054b1815ca6e0c71a362868c655f0e3da4616fb7598fb7'
LIB='/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so'
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
    return dict(status='PREPARED_NOT_EXECUTED',SDK_header=HEADER,SDK_header_sha256=HEADER_SHA,
        LP64_reference=dict(pointer_bytes=8,size_t_bytes=8,enum_bytes=4,kernel_params_bytes=88,task_params_bytes=200),
        own_application_payload_bytes=8192+24+24,driver_context_and_allocator_overhead_not_bounded_by_payload=True,
        graph_replay_calls=0,model_loads=0,tensor_contents_copied_to_host=0,
        graph_task_mutator_calls=0,query_stream_limit=32,query_task_limit_per_stream=256,
        controlled_submissions=[dict(provider='aclrtMemsetAsync',capacity_bytes=8192,count_bytes=64,value=0),
            dict(provider='aclrtMemsetD32Async',offset_bytes=256,capacity_bytes=8192-256,N_uint32=16,value=0),
            dict(provider='aten::zeros_like',shape=[24],dtype='BOOL')],
        pending_authorization='Explicit device diagnostic authorization and idle-device guard required before --run.',
        output_does_not_resolve_saved_model_operands=True)

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
    if output.parent != Path('/tmp') or not output.name.startswith('wio-operand-probe-v109-') or output.exists():
        raise RuntimeError('Output must be a new independent /tmp/wio-operand-probe-v109-* directory')
    if hashlib.sha256(Path(HEADER).read_bytes()).hexdigest()!=HEADER_SHA:
        raise RuntimeError('Installed header differs;do not guess ABI')
    if ctypes.sizeof(ctypes.c_void_p)!=8 or ctypes.sizeof(ctypes.c_size_t)!=8 or ctypes.sizeof(KernelParams)!=88 or ctypes.sizeof(TaskParams)!=200:
        raise RuntimeError('LP64 reference layout mismatch')
    output.mkdir(parents=True,exist_ok=False);result=dict(plan=describe(),status='STOPPED_BEFORE_CAPTURE',device=args.device)
    try:
        # This library loading and every accelerator operation occur only in opt-in run mode.
        lib=ctypes.CDLL(LIB)
        needed=GETTERS+['aclrtMemsetAsync','aclrtMemsetD32Async']
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
        byte_fill=bind(lib,'aclrtMemsetAsync',[ptr,ctypes.c_size_t,ctypes.c_int32,ctypes.c_size_t,ptr])
        word_fill=bind(lib,'aclrtMemsetD32Async',[ptr,ctypes.c_size_t,u32,ctypes.c_size_t,ptr])
        import torch
        import torch_npu
        torch.npu.set_device(args.device)
        buffer=torch.empty(8192,dtype=torch.uint8,device=f'npu:{args.device}')
        bool_input=torch.empty(24,dtype=torch.bool,device=f'npu:{args.device}')
        graph=torch_npu.npu.NPUGraph();model=ptr();capture_status=ctypes.c_int(-1)
        # No replay is required to ask whether captured metadata is available.
        with torch_npu.npu.graph(graph):
            stream=ptr(torch.npu.current_stream().npu_stream)
            require(byte_fill(ptr(buffer.data_ptr()),8192,0,64,stream),'controlled byte fill submission')
            require(word_fill(ptr(buffer.data_ptr()+256),8192-256,0,16,stream),'controlled D32 fill submission')
            bool_output=torch.zeros_like(bool_input)
            require(info(stream,ctypes.byref(capture_status),ctypes.byref(model)),'capture info')
        if not model.value:raise RuntimeError('Capture getter did not return a modelRI handle')
        rows=[];stream_handles=enumerate_handles(streams,model,32)
        for stream_handle in stream_handles:
            for task in enumerate_handles(tasks,ptr(stream_handle),256):
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
            application_buffer_address=buffer.data_ptr(),bool_input_address=bool_input.data_ptr(),bool_output_address=bool_output.data_ptr(),tasks=rows,
            typed_MemSet_destination_or_length_decoded=False,saved_model_correlation=False)
        if args.dump_own_graph:
            path=output/'own-tiny-graph.json';assert not path.exists();graph.debug_dump(str(path))
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
