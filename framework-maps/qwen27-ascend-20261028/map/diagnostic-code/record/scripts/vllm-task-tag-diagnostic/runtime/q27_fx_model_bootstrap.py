"""Defer NPU/compiler imports until the actual installed target module loads."""
import importlib.abc,importlib.machinery,os,re,sys

TARGETS={'npugraph_ex._acl_concrete_graph.fx2acl_converter','torch_npu.dynamo.npugraph_ex._acl_concrete_graph.fx2acl_converter','vllm_ascend.compilation.acl_graph'}
_REGISTRATION=None
_SETTING_UP=False

def _drain_callbacks(rows):
    # Called only after the exact outer capture context returned. No callback
    # fields are inspected until its saved producer streams have drained.
    streams={id(r['stream_ref']):r['stream_ref'] for r in rows if r.get('stream_ref')is not None}
    if len(rows)!=128 or not streams:raise RuntimeError('expected128 callbacks and saved streams')
    for stream in streams.values():stream.synchronize()
    decoded=[]
    for row in rows:
        callback=row['callback'];marker_id=row['marker_id'];match=re.fullmatch(r'.*:layer(\d+):(pre|post)',marker_id)
        if not match:raise RuntimeError('malformed explicit layer marker')
        fields={k:getattr(callback,k,None)for k in ('task_id','acl_stream_id','task_rc','stream_rc','capture_rc','capture_status','model_ri_nonnull','callback_rc')}
        valid=all(fields[k]==0 for k in ('task_rc','stream_rc','capture_rc','callback_rc'))
        decoded.append({**row,**fields,'capture_invocation_id':row['capture_id'],'layer_idx':int(match[1]),'marker_side':match[2],
            'producer_pid':os.getpid(),'callback_nonnull':fields['model_ri_nonnull'],'rc':0 if valid else -1,
            'callback_capture_status':fields['capture_status'],'callback_return_codes':{k:fields[k]for k in ('callback_rc','capture_rc','stream_rc','task_rc')},
            'classification':'capture_endpoint_candidate' if valid and fields['capture_status']==1 and fields['model_ri_nonnull']is True else 'unknown_capture_endpoint',
            'graph_identity_status':'unique_backend_saved_graph_object'})
    return decoded

def _setup():
    global _REGISTRATION,_SETTING_UP
    if _REGISTRATION is not None or _SETTING_UP:return
    _SETTING_UP=True
    try:
        import torch,torch_npu
        import q27_vllm_fx_events as fx
        import q27_vllm_probe as probe
        import vllm_model_outer
        from vllm.config import CUDAGraphMode
        extension=fx.load_anchor_extension('/task/full-map-20261025/vllm-task-tag-build-gen78-root/build-output/build/q27_vllm_task_anchor_tag_root78.so',
            '15f85c34088657aa0c054bf2149b990d6352c474e5971243cee91e5a6c2cff46')
        _REGISTRATION=vllm_model_outer.register_runtime_glue(torch,extension,target_mode=CUDAGraphMode.FULL,
            drain_callbacks=_drain_callbacks,emit=probe._emit,device_provider=lambda fc:int(torch.npu.current_device()))
    finally:_SETTING_UP=False

class Loader(importlib.abc.Loader):
    def __init__(self,wrapped):self.wrapped=wrapped
    def create_module(self,spec):
        method=getattr(self.wrapped,'create_module',None)
        return method(spec)if method else None
    def exec_module(self,module):
        self.wrapped.exec_module(module);_setup()

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in TARGETS:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec and spec.loader and not isinstance(spec.loader,Loader):spec.loader=Loader(spec.loader)
        return spec

def register():
    if os.environ.get('Q27_FX_MODEL_EVENTS')!='1':return {'enabled':False}
    if not any(isinstance(x,Finder)for x in sys.meta_path):sys.meta_path.insert(0,Finder())
    if any(name in sys.modules for name in TARGETS):_setup()
    return {'enabled':True,'runtime_configured':_REGISTRATION is not None}
