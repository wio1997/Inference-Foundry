"""Exact outer graph replay fences and explicit post-EOF cleanup for a canary."""
import json,os,sys,threading,time
from pathlib import Path

_STARTED=False

def install_after_acl_module(module,fx):
    global _STARTED
    if _STARTED:return
    _STARTED=True
    import torch
    import q27_vllm_probe as probe
    import vllm_replay_outer_events as outer
    cls=module.UpdatableGraph
    def held(graph):
        items=[x for x in fx.held_graphs() if x['graph'] is graph]
        if len(items)>1:raise RuntimeError('multiple owners for one actual graph')
        return items[0]if items else None
    def key(self,*unused):
        item=held(self)
        return item['graph_key'] if item else None
    def work(*unused):
        value=getattr(probe._TLS,'step_id',None)
        return json.dumps({'worker_step_id':value},sort_keys=True) if value is not None else None
    outer.install_replay_outer_anchors(cls,event_factory=lambda:torch.npu.Event(enable_timing=False),
        stream_provider=lambda:torch.npu.current_stream(),extension=fx.anchor_extension(),emit=probe._emit,
        graph_key_provider=key,work_key_provider=work,graph_allowed=lambda owner,g:held(g)is not None,
        graph_provider=lambda owner:owner,record_scope_factory=torch.profiler.record_function)
    request_path=os.environ.get('Q27_FX_CLEANUP_REQUEST')
    if not request_path:raise RuntimeError('explicit unique EOF cleanup request path required')
    def cleanup_after_EOF():
        try:
            path=Path(request_path)
            while True:
                if path.is_file():
                    request=json.loads(path.read_text())
                    if request.get('profiler_closed_after_EOF')is True and os.getpid() in request.get('request_pids',[]):break
                time.sleep(.25)
            entries=fx.held_graphs()
            if len(entries)!=1:raise RuntimeError('tiny model canary must own exactly one complete graph')
            item=entries[0];graph=item['graph'];device=item['device']
            if not isinstance(device,int):raise RuntimeError('saved producer device index missing')
            torch.npu.set_device(device);torch.npu.synchronize()
            result=outer.drain_outer_replay_anchors(emit=probe._emit)
            if result['status']!='drained' or outer.pending_counts()['failed_or_unresolved']:
                raise RuntimeError('outer events unresolved; retain refs')
            if not callable(getattr(graph,'reset',None)):raise RuntimeError('exact UpdatableGraph.reset missing')
            graph.reset();torch.npu.synchronize()
            receipt={'graph_python_id':id(graph),'reset_or_discarded':True,'device_drained_before_and_after_reset':True}
            # Stop future dispatch through this exact entry. The controller
            # submits no more requests once it asks for cleanup.
            entry=item['entry'];wrapper=item['wrapper']
            if entry.aclgraph is not graph:raise RuntimeError('saved entry graph ownership changed')
            entry.aclgraph=None;entry.output=None
            removed=[k for k,v in wrapper.concrete_aclgraph_entries.items()if v is entry]
            if len(removed)!=1:raise RuntimeError('saved entry no longer unique')
            del wrapper.concrete_aclgraph_entries[removed[0]]
            release=fx.release_graph_after_reset(graph,receipt)
            if release.get('released')is not True:raise RuntimeError('FX event release not proven after graph reset')
            outer_release=outer.release_drained_outer_events(graph,reset_receipt=receipt)
            torch.npu.synchronize()
            probe._emit('vllm_fx_graph_cleanup',**receipt,release=release,outer_events_released=outer_release,
                        serving_after_cleanup_allowed=False,status='reset_drained_and_events_released')
        except BaseException as error:
            probe._emit('vllm_fx_graph_cleanup_failed',error=repr(error),references_must_remain_retained=True)
    threading.Thread(target=cleanup_after_EOF,name='q27-owned-EOF-cleanup',daemon=True).start()

def register_runtime_glue(torch,extension,*,target_mode,drain_callbacks,emit,device_provider):
    import q27_vllm_fx_events as fx
    old=fx.install_module
    def install(module):
        result=old(module)
        if module.__name__=='vllm_ascend.compilation.acl_graph':install_after_acl_module(module,fx)
        return result
    fx.install_module=install
    result=fx.register_import_finder(torch,extension,target_mode=target_mode,drain_callbacks=drain_callbacks,emit=emit,device_provider=device_provider)
    module=sys.modules.get('vllm_ascend.compilation.acl_graph')
    if module is not None:install_after_acl_module(module,fx)
    return result
