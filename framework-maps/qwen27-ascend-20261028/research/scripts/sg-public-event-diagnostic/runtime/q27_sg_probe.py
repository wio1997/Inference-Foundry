"""Opt-in, CPU-only SGLang request/step observation. No tensor reads or device sync.
Install by calling install(namespace) after a target module is imported. Set Q27_SG_PROBE=1.
"""
import functools, importlib.abc, importlib.machinery, json, os, sys, threading, time, types
from pathlib import Path
import hashlib, importlib.util
_LOCK=threading.Lock(); _STEP=threading.local(); _OUT=Path(os.environ.get("Q27_SG_PROBE_DIR", "PRIVATE_PATH_fe5446ca0f07"))
def _plain(v):
    if v is None or isinstance(v,(bool,int,float,str)): return v
    if isinstance(v,dict): return {str(k):_plain(x) for k,x in list(v.items())[:128]}
    if isinstance(v,(list,tuple)): return [_plain(x) for x in v[:128]]
    if hasattr(v,"is_cpu"):
        if not v.is_cpu: return {"device_tensor_unread":True,"shape":list(v.shape),"dtype":str(v.dtype)}
        return v.tolist()
    # Only known host containers may be materialized. Unknown objects, including
    # device tensors without an is_cpu property, are represented opaquely.
    t=type(v); mod=getattr(t,"__module__","")
    if mod.startswith("numpy") and getattr(v,"ndim",None) is not None: return v.tolist()
    if mod=="array" and hasattr(v,"tolist"): return v.tolist()
    if hasattr(v,"device"):
        return {"unread_non_cpu_object":True,"type":mod+"."+getattr(t,"__qualname__",t.__name__),"device":str(getattr(v,"device",None))}
    return str(v)
_Q27_EXPLICIT_RANK = None
_Q27_LAYER_ANCHOR_EXT = None
_Q27_LAYER_ANCHOR_HELPER = None
_Q27_LAYER_ANCHOR_LAYERS_READY = False
_Q27_LAYER_ANCHOR_EXPECTED_SHA = "f1594bd83168014ba96cb0c2703fadfc19d947b0faaa76b75d17b4e6ded7a1bf"

def _layer_anchor_extension():
    global _Q27_LAYER_ANCHOR_EXT, _Q27_LAYER_ANCHOR_HELPER
    if os.environ.get("Q27_SG_LAYER_TASK_ANCHOR") != "1": return None
    if _Q27_LAYER_ANCHOR_EXT is not None: return _Q27_LAYER_ANCHOR_EXT
    so=Path(os.environ.get("Q27_SG_TASK_ANCHOR_SO", ""))
    if not so.is_file(): raise RuntimeError("Q27_SG_TASK_ANCHOR_SO must name the reviewed gen22 extension")
    h=hashlib.sha256(so.read_bytes()).hexdigest()
    if h != _Q27_LAYER_ANCHOR_EXPECTED_SHA: raise RuntimeError("task-anchor extension SHA mismatch")
    spec=importlib.util.spec_from_file_location("q27_sg_task_anchor_gen20",so)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load reviewed task-anchor extension")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    event_mode=os.environ.get("Q27_SG_LAYER_PUBLIC_EVENTS") == "1"
    helper_name="sg_layer_event_anchor.py" if event_mode else "sg_layer_task_anchor.py"
    helper_module="q27_sg_layer_event_anchor_gen44" if event_mode else "q27_sg_layer_task_anchor_gen34"
    helper_path=Path(__file__).with_name(helper_name)
    hs=importlib.util.spec_from_file_location(helper_module,helper_path)
    if hs is None or hs.loader is None: raise RuntimeError("cannot load gen34 anchor helper")
    helper=importlib.util.module_from_spec(hs); hs.loader.exec_module(helper)
    _Q27_LAYER_ANCHOR_EXT=module; _Q27_LAYER_ANCHOR_HELPER=helper
    return module

def _explicit_rank():
    global _Q27_EXPLICIT_RANK
    if _Q27_EXPLICIT_RANK is None:
        try:
            import sys
            dist = sys.modules.get("torch.distributed")
            if dist is not None and dist.is_initialized():
                _Q27_EXPLICIT_RANK = int(dist.get_rank())
        except Exception:
            pass
    return _Q27_EXPLICIT_RANK

def _emit(event, **fields):
    if os.environ.get("Q27_SG_PROBE") != "1": return
    row={"rank_id":_explicit_rank(),"rank_source":"torch.distributed.get_rank when initialized; no PID inference","event":event,"pid":os.getpid(),"wall_ns":time.time_ns(),"mono_ns":time.monotonic_ns(),**fields}
    data=(json.dumps(_plain(row),separators=(",",":"),ensure_ascii=False)+"\n").encode()
    outdir=Path(os.environ.get("Q27_SG_PROBE_DIR", str(_OUT))); outdir.mkdir(parents=True,exist_ok=True)
    with _LOCK:
        fd=os.open(outdir/("sg-"+str(os.getpid())+".jsonl"),os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        try: os.write(fd,data)
        finally: os.close(fd)
def _span(name):
    try:
        import torch
        return torch.profiler.record_function(name)
    except Exception:
        from contextlib import nullcontext
        return nullcontext()
def _reqs(batch):
    out=[]
    for r in (getattr(batch,"reqs",None) or []):
        out.append({k:_plain(getattr(r,k,None)) for k in ("rid","request_id","req_id")})
    return out
def _wrap(cls, name, before=None, after=None, span_name=None):
    original=getattr(cls,name,None)
    if original is None or getattr(original,"_q27_sg_wrapped",False): return False
    @functools.wraps(original)
    def wrapped(self,*args,**kwargs):
        ctx={}; t=time.monotonic_ns()
        if before:
            try: ctx=before(self,args,kwargs) or {}
            except Exception as e: _emit("probe_metadata_error",hook=name,error=repr(e))
        try:
            if span_name:
                with _span(span_name): out=original(self,*args,**kwargs)
            else: out=original(self,*args,**kwargs)
        except BaseException as e:
            _emit("probe_hook_exception",hook=name,error=repr(e),host_wall_ns=time.monotonic_ns()-t)
            raise
        if after:
            try: after(self,args,kwargs,out,ctx,time.monotonic_ns()-t)
            except Exception as e: _emit("probe_metadata_error",hook=name,error=repr(e))
        return out
    wrapped._q27_sg_wrapped=True; setattr(cls,name,wrapped); return True
def install(namespace=None):
    """Patch whichever imported SGLang module namespace is passed; repeat-safe."""
    if os.environ.get("Q27_SG_PROBE") != "1": return []
    namespace=namespace or {}; mod=namespace.get("__name__",""); done=[]
    if mod.endswith("managers.scheduler"):
        cls=namespace.get("Scheduler")
        def rb(self,args,kw):
            b=args[0] if args else kw.get("batch"); return {"forward_mode":str(getattr(b,"forward_mode",None)),"requests":_reqs(b),"forward_iter":getattr(b,"forward_iter",None)}
        def rb_after(self,args,kw,out,c,dt): _emit("scheduler_run_batch",**c,host_wall_ns=dt)
        if cls and _wrap(cls,"run_batch",rb,rb_after,"Q27 sglang scheduler run_batch"): done.append("scheduler.run_batch")
        def result_before(self,args,kw):
            b=args[0] if args else kw.get("batch"); reqs=getattr(b,"reqs",None) or []
            return {"forward_mode":str(getattr(b,"forward_mode",None)),"forward_iter":getattr(b,"forward_iter",None),"requests":_reqs(b),"pre_output_counts":[len(getattr(r,"output_ids",[]) or []) for r in reqs]}
        def commit_after(self,args,kw,out,c,dt):
            b=args[0] if args else kw.get("batch"); reqs=getattr(b,"reqs",None) or []; prev=c.get("pre_output_counts",[]); ledger=[]
            for i,r in enumerate(reqs):
                ids=getattr(r,"output_ids",[]) or []; count=len(ids); old=prev[i] if i<len(prev) else None
                ledger.append({"rid":getattr(r,"rid",None),"committed_output_ids_total":count,"model_sample_token_delta":(count-old if old is not None else None),"last_token_id":ids[-1] if count else None})
            _emit("scheduler_commit",**{k:v for k,v in c.items() if k!="pre_output_counts"},host_wall_ns=dt,token_commit=ledger)
        if cls and _wrap(cls,"process_batch_result",result_before,commit_after,"Q27 sglang scheduler process_batch_result"): done.append("scheduler.process_batch_result")
        if cls and not getattr(cls.process_batch_result,"_q27_ipc_step_scope",False):
            process_result=cls.process_batch_result
            @functools.wraps(process_result)
            def result_with_ipc_scope(self,batch,*args,_original=process_result,**kwargs):
                old=getattr(_STEP,"current",None)
                _STEP.current={"step":getattr(batch,"forward_iter",None),"forward_mode":str(getattr(batch,"forward_mode",None)),"requests":_reqs(batch)}
                try:return _original(self,batch,*args,**kwargs)
                finally:_STEP.current=old
            result_with_ipc_scope._q27_ipc_step_scope=True;cls.process_batch_result=result_with_ipc_scope


    elif mod.endswith("managers.tp_worker"):
        cls=namespace.get("TpModelWorker") or namespace.get("TPWorker")
        if cls is None:
            cls=next((v for v in namespace.values() if isinstance(v,type) and hasattr(v,"forward_batch_generation")),None)
        original=getattr(cls,"forward_batch_generation",None) if cls else None
        if original is not None and not getattr(original,"_q27_sg_wrapped",False):
            @functools.wraps(original)
            def worker_forward(self,*args,**kwargs):
                b=args[0] if args else kwargs.get("batch")
                ctx={"step":getattr(b,"forward_iter",None),"forward_mode":str(getattr(b,"forward_mode",None)),"requests":_reqs(b)}
                old=getattr(_STEP,"current",None);_STEP.current=ctx;start=time.monotonic_ns()
                _emit("worker_forward_enter",**ctx)
                try:
                    with _span(f"Q27 sglang worker forward_batch_generation pid={os.getpid()} step={ctx.get('step')}"): out=original(self,*args,**kwargs)
                    return out
                finally:
                    _emit("worker_forward_exit",**ctx,host_wall_ns=time.monotonic_ns()-start)
                    _STEP.current=old
            worker_forward._q27_sg_wrapped=True;setattr(cls,"forward_batch_generation",worker_forward);done.append("TPWorker.forward_batch_generation")
    elif mod.endswith("model_executor.model_runner"):
        cls=namespace.get("ModelRunner")
        def on_load(self,args,kwargs,out,ctx,dt):
            layers,path,reason=_locate_expected_qwen_layers(getattr(self,"model",None))
            if layers is None:
                _emit("layer_directory_install_failed",path=path,reason=reason,expected_layers=64,expected_counts={"linear_attention":48,"full_attention":16})
                return
            _emit("layer_directory_resolved",path=path,layer_count=len(layers))
            layer_types=["linear_attention" if type(x).__name__=="Qwen3_5LinearDecoderLayer" else "full_attention" for x in layers]
            write_layer_directory(types.SimpleNamespace(layers=layers),layer_types,instrument=True)
            if os.environ.get("Q27_SG_LAYER_TASK_ANCHOR") == "1":
                global _Q27_LAYER_ANCHOR_LAYERS_READY
                try:
                    ext=_layer_anchor_extension()
                    import torch
                    if os.environ.get("Q27_SG_LAYER_PUBLIC_EVENTS") == "1":
                        _Q27_LAYER_ANCHOR_HELPER.install_layer_event_anchor_set(
                            layers,layer_types,ext,_emit,stream_provider=lambda:torch.npu.current_stream())
                    else:
                        _Q27_LAYER_ANCHOR_HELPER.install_layer_anchor_set(
                            layers,layer_types,ext,_emit,stream_provider=lambda:torch.npu.current_stream())
                    _Q27_LAYER_ANCHOR_LAYERS_READY=True
                except Exception as e:
                    _emit("layer_task_anchor_install_failed",error=repr(e),expected_layers=64)
                    raise
        if cls and _wrap(cls,"load_model",None,on_load,"Q27 SGLang ModelRunner.load_model"): done.append("ModelRunner.load_model→layer directory")
        def batch(self,args,kw):
            b=args[0] if args else kw.get("forward_batch"); return {"forward_mode":str(getattr(b,"forward_mode",None)),"batch_size":getattr(b,"batch_size",None),"step":getattr(b,"forward_iter",None) or (getattr(_STEP,"current",None) or {}).get("step"),"requests":(getattr(_STEP,"current",None) or {}).get("requests"),"req_pool_indices":_plain(getattr(b,"req_pool_indices",None)),"seq_lens_cpu":_plain(getattr(b,"seq_lens_cpu",None)),"extend_num_tokens":_plain(getattr(b,"extend_num_tokens",None)),"extend_seq_lens_cpu":_plain(getattr(b,"extend_seq_lens_cpu",None)),"extend_prefix_lens_cpu":_plain(getattr(b,"extend_prefix_lens_cpu",None)),"rids":_plain(getattr(b,"rids",None))}
        if cls and _wrap(cls,"forward",batch,lambda s,a,k,o,c,d:_emit("model_forward",**c,host_wall_ns=d),"Q27 sglang model forward"): done.append("ModelRunner.forward")
        if cls and _wrap(cls,"sample",batch,lambda s,a,k,o,c,d:_emit("model_sample",**c,host_wall_ns=d,model_sample_rows=getattr(o,"shape",[None])[0] if hasattr(o,"shape") else None),"Q27 sglang model sample"): done.append("ModelRunner.sample")
    elif mod.endswith("hardware_backend.npu.graph_runner.npu_cudagraph_backend"):
        cls=namespace.get("NPUCudaGraphBackend")
        if cls and os.environ.get("Q27_SG_LAYER_PUBLIC_EVENTS")=="1":
            import torch, sg_replay_outer_events
            extension=_layer_anchor_extension()
            def graph_key_provider(backend, method, args, kwargs):
                # Frozen signatures have exact shape_key as first argument.
                if args and "shape_key"in kwargs:raise RuntimeError("duplicate shape key")
                return args[0] if args else kwargs["shape_key"]
            def graph_allowed(backend, graph):
                helper=_Q27_LAYER_ANCHOR_HELPER
                with helper._GRAPH_EVENT_HOLDS_LOCK:
                    held=helper._GRAPH_EVENT_HOLDS.get((id(backend),id(graph)))
                    return held is not None and held.get("graph")is graph
            def work_key_provider(*args):
                value=getattr(_STEP,"current",None)
                return json.dumps(value,sort_keys=True) if value is not None else None
            sg_replay_outer_events.install_replay_outer_anchors(cls,
                event_factory=lambda:torch.npu.Event(enable_timing=False),
                stream_provider=lambda:torch.npu.current_stream(),extension=extension,emit=_emit,
                graph_key_provider=graph_key_provider,graph_allowed=graph_allowed,
                work_key_provider=work_key_provider,record_scope_factory=torch.profiler.record_function)

        if cls and os.environ.get("Q27_SG_LAYER_TASK_ANCHOR") == "1":
            original_capture=getattr(cls,"capture_one",None)
            if original_capture is not None and not getattr(original_capture,"_q27_backend_layer_anchor_wrapped",False):
                @functools.wraps(original_capture)
                def capture_one_with_layer_anchors(self,*args,_original=original_capture,**kwargs):
                    if _Q27_LAYER_ANCHOR_HELPER is None or not _Q27_LAYER_ANCHOR_LAYERS_READY:
                        out=_original(self,*args,**kwargs)
                        _emit("layer_task_anchor_unresolved",reason="backend module imported before exact layer hooks were ready",
                              graph_key=str(args[0] if args else kwargs.get("shape_key")))
                        return out
                    if os.environ.get("Q27_SG_LAYER_PUBLIC_EVENTS") == "1":
                        return _Q27_LAYER_ANCHOR_HELPER.capture_backend_one_with_public_events(
                            _original,self,args,kwargs,emit=_emit,
                            expected_shape_key=os.environ.get("Q27_SG_LAYER_EVENT_SHAPE_KEY"))
                    return _Q27_LAYER_ANCHOR_HELPER.capture_backend_one_scope(
                        _original,self,args,kwargs,enabled=True,emit=_emit)
                capture_one_with_layer_anchors._q27_backend_layer_anchor_wrapped=True
                setattr(cls,"capture_one",capture_one_with_layer_anchors)
                done.append("NPUCudaGraphBackend.capture_one→warmup/capture forward endpoints")
    elif mod.endswith("hardware_backend.npu.graph_runner.npu_graph_runner"):
        cls=namespace.get("NPUGraphRunner")
        def gb(self,args,kw):
            b=args[0] if args else kw.get("forward_batch"); return {"step":(getattr(_STEP,"current",None)or{}).get("step"),"requests":(getattr(_STEP,"current",None)or{}).get("requests"),"graph_key":str(getattr(self,"_q27_last_graph_key",None)),"graph_bs":getattr(self,"bs",None),"raw_bs":getattr(self,"raw_bs",None),"captured_req_width":getattr(self,"captured_req_width",None),"actual_batch":getattr(b,"batch_size",None),"forward_mode":str(getattr(b,"forward_mode",None))}
        if cls and _wrap(cls,"_make_graph_key",lambda s,a,k:{"bs":a[0] if a else k.get("bs")},lambda s,a,k,o,c,d:(setattr(s,"_q27_last_graph_key",o),_emit("npu_graph_key",bs=c.get("bs"),graph_key=str(o)))): done.append("NPUGraphRunner._make_graph_key")
        if cls and _wrap(cls,"execute",gb,lambda s,a,k,o,c,d:_emit("npu_graph_execute_returned",**gb(s,a,k),graph_metadata_stage="post_execute",host_wall_ns=d,meaning="host return after graph dispatch/replay; graph metadata read post-execute; no device elapsed time"),"Q27 sglang NPU graph execute") : done.append("NPUGraphRunner.execute")
        if cls and os.environ.get("Q27_SG_LAYER_TASK_ANCHOR") == "1" and os.environ.get("Q27_SG_LAYER_PUBLIC_EVENTS") != "1":
            original_capture=getattr(cls,"_capture_graph",None)
            if original_capture is not None and not getattr(original_capture,"_q27_layer_task_anchor_wrapped",False):
                @functools.wraps(original_capture)
                def capture_with_layer_anchors(self,*args,_original=original_capture,**kwargs):
                    if _Q27_LAYER_ANCHOR_HELPER is None or not _Q27_LAYER_ANCHOR_LAYERS_READY:
                        out=_original(self,*args,**kwargs)
                        _emit("layer_task_anchor_unresolved",reason="graph module imported before exact layer hooks were ready",
                              graph_key=str(getattr(self,"_q27_last_graph_key",None)))
                        return out
                    return _Q27_LAYER_ANCHOR_HELPER.capture_graph_scope(
                        _original,self,args,kwargs,enabled=True,emit=_emit)
                capture_with_layer_anchors._q27_layer_task_anchor_wrapped=True
                setattr(cls,"_capture_graph",capture_with_layer_anchors)
                done.append("NPUGraphRunner._capture_graph→64 endpoint anchors; post-capture drain")
        if cls and _wrap(cls,"_capture_graph",lambda s,a,k:{"graph_bs":getattr(s,"bs",None),"graph_key":str(getattr(s,"_q27_last_graph_key",None))},lambda s,a,k,o,c,d:_emit("npu_graph_capture_returned",**c,host_wall_ns=d),"Q27 sglang NPU graph capture"):
            done.append("NPUGraphRunner._capture_graph")
    elif mod.endswith("managers.detokenizer_manager"):
        cls=namespace.get("DetokenizerManager")
        def tok_before(self,args,kw):
            o=args[0] if args else kw.get("recv_obj"); rids=list(getattr(o,"rids",[]) or [])
            decode_ids=getattr(o,"decode_ids",[]) or []; out_ids=getattr(o,"output_ids",[]) or []
            return {"rids":rids,"decode_id_counts":[len(x) for x in decode_ids],"model_output_id_counts":[len(x) for x in out_ids],"recv_type":type(o).__name__}
        def decode_after(self,args,kw,out,c,dt):
            _emit("detokenizer_cpu_token_to_text",**c,host_wall_ns=dt,decoded_text_char_counts=[len(x) for x in (out or [])],text_values_omitted=True)
        if cls and _wrap(cls,"_decode_batch_token_id_output",tok_before,decode_after,"Q27 detokenizer CPU token-to-text"): done.append("DetokenizerManager._decode_batch_token_id_output")
        if cls and _wrap(cls,"handle_batch_token_id_out",tok_before,lambda s,a,k,o,c,d:_emit("detokenizer_batch_output",**c,host_wall_ns=d,output_type=type(o).__name__,output_text_char_counts=[len(x) for x in (getattr(o,"output_strs",[]) or [])]),"Q27 detokenizer batch output"): done.append("DetokenizerManager.handle_batch_token_id_out")
    elif mod.endswith("managers.schedule_batch"):
        cls=namespace.get("Req")
        for method in ("tail_str","_locate_str_stop_finished_len"):
            if cls and _wrap(cls,method,lambda s,a,k:{"rid":getattr(s,"rid",None),"purpose":"stop string/regex matching CPU token decode"},lambda s,a,k,o,c,d:_emit("scheduler_cpu_token_decode",**c,host_wall_ns=d,result_type=type(o).__name__),"Q27 SGLang stop matcher CPU decode"):
                done.append("Req."+method)
    elif mod.endswith("managers.tokenizer_manager"):
        cls=namespace.get("TokenizerManager")
        def out_before(self,args,kw):
            o=args[0] if args else kw.get("recv_obj"); return {"rids":_plain(getattr(o,"rids",None)),"output_ids":_plain(getattr(o,"output_ids",None)),"output_ids_len":_plain(getattr(o,"output_ids_len",None)),"type":type(o).__name__}
        if cls and _wrap(cls,"_handle_batch_output",out_before,lambda s,a,k,o,c,d:_emit("tokenizer_detokenizer_batch_output",**c,host_wall_ns=d),"Q27 tokenizer/detokenizer output"): done.append("TokenizerManager._handle_batch_output")
    elif mod.endswith("entrypoints.http_server"):
        # ASGI body observer runs at the server send boundary. SSE frames carry
        # request id in their JSON payload when present; chunks remain frames.
        app=namespace.get("app")
        if app is not None and not getattr(app,"_q27_sg_probe_middleware",False):
            class _SSEObserver:
                def __init__(self,app): self.app=app
                async def __call__(self,scope,receive,send):
                    state={"sse":scope.get("path")=="/generate","frame":0,"rid":None}
                    async def observed(message):
                        typ=message.get("type")
                        if typ=="http.response.start":
                            headers={k.lower():v.lower() for k,v in message.get("headers",[])}
                            state["sse"] = state["sse"] or b"text/event-stream" in headers.get(b"content-type",b"")
                        if scope.get("type")=="http" and state["sse"] and typ=="http.response.body":
                            body=message.get("body",b""); rid=None
                            try:
                                txt=body.decode("utf-8",errors="ignore")
                                data=next((x[6:] for x in txt.splitlines() if x.startswith("data: ") and x[6:]!="[DONE]"),None)
                                if data:
                                    payload=json.loads(data); rid=payload.get("id") or (payload.get("meta_info") or {}).get("id")
                            except Exception: pass
                            if rid is not None: state["rid"]=rid
                            more=bool(message.get("more_body",False)); idx=state["frame"];state["frame"]+=1
                            part="first_last" if idx==0 and not more else ("first" if idx==0 else ("more" if more else "last"))
                            _emit("api_sse_body",rid=state["rid"],path=scope.get("path"),body_index=idx,body_part=part,more_body=more,body_bytes=len(body),meaning="ASGI SSE body chunk/frame; not token count")
                        await send(message)
                    try:
                        await self.app(scope,receive,observed)
                    except BaseException as e:
                        _emit("api_sse_exception",path=scope.get("path"),rid=state.get("rid"),error=repr(e),body_count=state["frame"])
                        raise
            app.add_middleware(_SSEObserver); app._q27_sg_probe_middleware=True; done.append("http_server ASGI SSE body")
    if os.environ.get("Q27_HYBRID_CACHE_PROBE")=="1":
        import q27_hybrid_bootstrap
        def hybrid_emit(row):
            _emit(row["event"],**{k:v for k,v in row.items()if k!="event"})
        status=q27_hybrid_bootstrap.install_from_modules(hybrid_emit,_span)
        if status.get("installed"):_emit("hybrid_cache_probe_installed",status=status)
    import frontend_stage_probe_patch as front
    done.extend(front.install_sgl(namespace))
    if done: _emit("probe_installed",module=mod,hooks=done)
    return done

def install_ipc_observer():
    """Opt-in observer for the natural CPU token IPC send boundary.

    Call explicitly after pyzmq is importable. Only an object whose exact type
    name is BatchTokenIDOutput is observed. Metadata comes only from existing
    CPU lists; no tensors are inspected or copied. The wrapped send's return or
    exception is preserved. Does not require the regular module install hooks.
    """
    if os.environ.get("Q27_SG_PROBE") != "1": return False
    try:
        import zmq
        cls=zmq.Socket
        original=getattr(cls,"send_pyobj",None)
        if original is None or getattr(original,"_q27_sg_ipc_wrapped",False): return False
    except Exception:
        return False

    def lengths(value):
        if value is None: return None
        try: return [len(x) for x in value]
        except Exception: return None

    @functools.wraps(original)
    def observed(self,obj,*args,**kwargs):
        if type(obj).__name__ != "BatchTokenIDOutput":
            return original(self,obj,*args,**kwargs)
        try:
            rids=getattr(obj,"rids",None)
            fields={
                "rids":list(rids) if rids is not None else None,
                "step":(getattr(_STEP,"current",None) or {}).get("step"),
                "forward_mode":(getattr(_STEP,"current",None) or {}).get("forward_mode"),
                "decode_id_counts":lengths(getattr(obj,"decode_ids",None)),
                "output_id_counts":lengths(getattr(obj,"output_ids",None)),
                "completion_tokens":list(getattr(obj,"completion_tokens",None) or []),
                "decoded_text_char_counts":lengths(getattr(obj,"decoded_texts",None)),
                "socket_type":type(self).__module__+"."+type(self).__qualname__,
            }
        except Exception as e:
            fields={"metadata_error":repr(e),"socket_type":type(self).__module__+"."+type(self).__qualname__}
        begin_wall=time.time_ns(); begin_mono=time.monotonic_ns()
        result=None; error=None
        try:
            result=original(self,obj,*args,**kwargs)
            return result
        except BaseException as e:
            error=repr(e)
            raise
        finally:
            end_mono=time.monotonic_ns(); end_wall=time.time_ns()
            try:
                _emit("ipc_batch_token_id_send",**fields,
                      send_begin_wall_ns=begin_wall,send_begin_mono_ns=begin_mono,
                      send_end_wall_ns=end_wall,send_end_mono_ns=end_mono,
                      send_exception=error)
            except Exception:
                # Observation failures must never alter IPC success or failure.
                pass
    observed._q27_sg_ipc_wrapped=True
    cls.send_pyobj=observed
    return True

def _locate_expected_qwen_layers(root):
    """Search only known model wrapper attributes and require one exact 64-layer Qwen list."""
    if root is None: return None,None,"self.model is None after load_model"
    queue=[("self.model",root)]; seen_nodes=set(); matches={};
    while queue:
        path,node=queue.pop(0)
        if node is None or id(node) in seen_nodes: continue
        seen_nodes.add(id(node))
        layers=getattr(node,"layers",None)
        if layers is not None:
            try: seq=list(layers)
            except Exception: seq=[]
            if len(seq)==64:
                key=tuple(id(x) for x in seq); matches.setdefault(key,(path,seq))
        for attr in ("model","language_model","text_model"):
            try: child=getattr(node,attr,None)
            except Exception: child=None
            if child is not None: queue.append((path+"."+attr,child))
    if len(matches)!=1:
        return None,None,("no 64-layer candidate found" if not matches else "multiple distinct 64-layer candidates found: "+", ".join(x[0] for x in matches.values()))
    path,layers=next(iter(matches.values()))
    counts={"Qwen3_5LinearDecoderLayer":0,"Qwen3_5AttentionDecoderLayer":0}
    for layer in layers:
        name=type(layer).__name__
        if name not in counts: return None,path,"64-layer candidate contains unexpected module type "+name
        counts[name]+=1
    if counts!={"Qwen3_5LinearDecoderLayer":48,"Qwen3_5AttentionDecoderLayer":16}:
        return None,path,"layer class counts mismatch: "+repr(counts)
    return layers,path,None

def write_layer_directory(model, layer_types=None, instrument=True):
    """Record per-layer static identity; optionally wrap Python forward spans.

    Python forward spans execute during eager execution and graph capture. Graph
    replay does not call these Python methods, so this never labels replay time.
    Call after model construction and before graph capture.
    """
    rows=[]
    layers=getattr(model,"layers",None)
    if layers is not None:
        for i,layer in enumerate(layers):
            typ=(layer_types[i] if layer_types and i<len(layer_types) else type(layer).__name__)
            name=type(layer).__module__+"."+type(layer).__qualname__
            rows.append({"layer_idx":i,"layer_type":str(typ),"module":name})
            if instrument and not getattr(layer,"_q27_sg_layer_wrapped",False):
                original=layer.forward
                @functools.wraps(original)
                def layer_forward(*args,_original=original,_idx=i,_typ=str(typ),**kwargs):
                    begin=time.monotonic_ns(); ctx=getattr(_STEP,"current",None)or{}; event=f"Q27 sglang layer_forward idx={_idx} layer_type={_typ} step={ctx.get('step','unknown')} phase={ctx.get('forward_mode','unknown')}"
                    try:
                        with _span(event): out=_original(*args,**kwargs)
                    except BaseException as e:
                        _emit("layer_forward_exception",layer_idx=_idx,layer_type=_typ,error=repr(e),host_wall_ns=time.monotonic_ns()-begin,python_call=True)
                        raise
                    _emit("layer_forward_python_call",layer_idx=_idx,layer_type=_typ,step=(getattr(_STEP,"current",None) or {}).get("step"),host_wall_ns=time.monotonic_ns()-begin,python_call=True,meaning="eager or graph-capture Python forward; graph replay does not invoke layer.forward")
                    return out
                layer.forward=layer_forward
                layer._q27_sg_layer_wrapped=True
    _emit("static_layer_directory",layers=rows,source="module/config metadata; no timings")
    return rows

class _ProbeLoader(importlib.abc.Loader):
    def __init__(self,loader): self._loader=loader
    def create_module(self,spec):
        create=getattr(self._loader,"create_module",None)
        return create(spec) if create else None
    def exec_module(self,module):
        self._loader.exec_module(module)
        install(module.__dict__)
    def __getattr__(self,name): return getattr(self._loader,name)
class _ProbeFinder(importlib.abc.MetaPathFinder):
    def __init__(self,targets,delegates): self.targets=set(targets); self.delegates=delegates
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in self.targets: return None
        for finder in self.delegates:
            find=getattr(finder,"find_spec",None)
            if find is None: continue
            spec=find(fullname,path,target)
            if spec is not None:
                if spec.loader is not None and not isinstance(spec.loader,_ProbeLoader): spec.loader=_ProbeLoader(spec.loader)
                return spec
        return None

def install_finder(target_modules=None):
    """Install an opt-in finder that delegates to, then wraps, existing loaders.

    It keeps old profile-overlay finders in the delegation chain, so their
    manifest modules still load through their original verified loader. Other
    exact target modules fall through to the normal finders and are wrapped.
    Existing loaded target modules are installed immediately.
    """
    if os.environ.get("Q27_SG_PROBE") != "1": return None
    targets=set(target_modules or (
        "sglang.srt.mem_cache.unified_cache.components.mamba","sglang.srt.mem_cache.memory_pool","sglang.srt.mem_cache.mamba_checkpoint_pool",
        "sglang.srt.managers.scheduler","sglang.srt.managers.tp_worker",
        "sglang.srt.model_executor.model_runner",
        "sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner",
        "sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend",
        "sglang.srt.managers.tokenizer_manager","sglang.srt.managers.schedule_batch",
        "sglang.srt.managers.detokenizer_manager","sglang.srt.entrypoints.http_server"))
    existing=getattr(sys,"_q27_sg_probe_finder",None)
    if existing is not None:
        existing.targets.update(targets); return existing
    delegates=list(sys.meta_path); finder=_ProbeFinder(targets,delegates)
    sys.meta_path.insert(0,finder);sys._q27_sg_probe_finder=finder
    for name in targets:
        module=sys.modules.get(name)
        if module is not None: install(module.__dict__)
    return finder
