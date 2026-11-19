"""Native G17 active witnesses -> protected G15 persistent bucket live state.

Only Python metadata is bridged out of graph. Captured donor/NoPE core is unchanged.
"""
import copy

def resolve_live_rows(fb,width,*,in_capture=False):
    physical=fb.batch_size
    if type(physical) is not int or physical<=0 or width not in (1,4):raise RuntimeError('Invalid graph physical geometry')
    actual_mode=getattr(fb,'actual_forward_mode',fb.forward_mode)
    if in_capture or bool(getattr(actual_mode,'is_idle',lambda:False)()):return 0
    graph_padding=getattr(fb,'num_padding',0)
    if type(graph_padding) is not int or not 0<=graph_padding<=physical:raise RuntimeError('Invalid native graph bucket padding')
    native_rows=physical-graph_padding
    original=getattr(fb,'_original_batch_size',None)
    original_tokens=getattr(fb,'_original_num_tokens',None)
    if original is None:
        if original_tokens is not None and original_tokens!=native_rows*width:raise RuntimeError('Original token witness has no consistent active request owner')
        return native_rows
    if type(original) is not int or not 0<original<=native_rows:raise RuntimeError('Original active request witness exceeds native physical rows')
    if type(original_tokens) is not int or original_tokens!=original*width:raise RuntimeError('Original request/token witnesses disagree with graph W')
    return original

def stage_backend_g17(backend,fb,in_capture=False):
    from .mtp_graph_bucket import stage_backend
    mode=fb.forward_mode
    width=4 if mode.is_target_verify() or mode.is_draft_extend_v2() else 1
    raw=resolve_live_rows(fb,width,in_capture=in_capture)
    view=copy.copy(fb)
    # Collapse native TP and graph-capacity padding ONLY for bucket live ownership.
    # Never mutate runner fb/positions/seq_lens/out_cache_loc or the real batch size.
    view.num_padding=fb.batch_size-raw
    backend._glm53_graph_live_witness={'physical_requests':fb.batch_size,'active_requests':raw,'width':width,'original_batch_size':getattr(fb,'_original_batch_size',None),'original_num_tokens':getattr(fb,'_original_num_tokens',None),'native_graph_num_padding':getattr(fb,'num_padding',0)}
    return stage_backend(backend,view,in_capture=in_capture)
