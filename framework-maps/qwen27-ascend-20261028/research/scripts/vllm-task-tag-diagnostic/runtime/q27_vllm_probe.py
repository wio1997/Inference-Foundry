"""Opt-in host-side vLLM markers; no tensor value reads or device synchronization."""
from __future__ import annotations
import functools, importlib.abc, importlib.machinery, inspect, json, os, threading, time
from pathlib import Path

_OUT = Path(os.environ.get("Q27_VLLM_PROBE_OUT", "PRIVATE_PATH_ce57f68b29b5"))
_TLS = threading.local()
_LOCK = threading.Lock()
_INSTALLED = set()


def _span(label):
    try:
        import torch
        return torch.profiler.record_function(label)
    except Exception:
        from contextlib import nullcontext
        return nullcontext()


_Q27_EXPLICIT_RANK = None

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
    row = {"rank_id":_explicit_rank(),"rank_source":"torch.distributed.get_rank when initialized; no PID inference","framework":"vllm", "event":event, "pid":os.getpid(),
           "wall_ns":time.time_ns(), "mono_ns":time.monotonic_ns(),
           "thread_id":threading.get_ident(), "step_id":getattr(_TLS,"step_id",None), **fields}
    _OUT.mkdir(parents=True, exist_ok=True)
    raw=(json.dumps(row, default=str, separators=(",",":"))+"\n").encode()
    with _LOCK:
        fd=os.open(_OUT/"events.jsonl", os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o600)
        try: os.write(fd, raw)
        finally: os.close(fd)


def _wrap_method(cls, name, event, fields_fn=None):
    key=(cls,name)
    if key in _INSTALLED or not hasattr(cls,name): return False
    original=getattr(cls,name)
    def before(self,args,kwargs):
        prev=getattr(_TLS,"step_id",None)
        if event == "worker_step":
            _TLS.step_id=f"{os.getpid()}-{time.monotonic_ns()}"
        elif event == "api_completion":
            _TLS.step_id=f"api-{os.getpid()}-{time.monotonic_ns()}"
        elif event == "sampler":
            # sample_tokens can run after execute_model has returned, on the
            # same runner but outside its thread-local step scope.
            last_step=getattr(self,"_q27_last_step_id",None)
            if last_step is not None:
                _TLS.step_id=last_step
        sid=getattr(_TLS,"step_id",None)
        fields=fields_fn(self,args,kwargs) if fields_fn else {}
        started=time.monotonic_ns()
        _emit(event+"_enter", step_id=sid, **fields)
        return prev,sid,fields,started
    def after(self,prev,sid,started,fields=None):
        if event == "worker_step" and sid is not None:
            # Persistent Python-side anchor for the subsequent sampler call.
            self._q27_last_step_id=sid
        _emit(event+"_exit", step_id=sid, host_wall_ns=time.monotonic_ns()-started, **(fields or {}))
        if event in ("worker_step","api_completion","sampler"): _TLS.step_id=prev
    def span_name(sid,fields):
        label=f"Q27 vLLM {event} pid={os.getpid()} step={sid}"
        if event == "model_layer":
            label += f" layer_idx={fields.get('layer_idx')} layer_type={fields.get('layer_type')} phase={getattr(_TLS,'step_phase','unknown')}"
        return label
    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapped(self,*args,**kwargs):
            prev,sid,fields,started=before(self,args,kwargs)
            try:
                with _span(span_name(sid,fields)):
                    result=await original(self,*args,**kwargs)
                if event == "api_completion" and inspect.isasyncgen(result):
                    result=_observe_async_iterator(result,sid)
                elif event == "api_completion" and hasattr(result,"body_iterator"):
                    result.body_iterator=_observe_async_iterator(result.body_iterator,sid)
                return result
            finally: after(self,prev,sid,started,fields)
    else:
      @functools.wraps(original)
      def wrapped(self,*args,**kwargs):
        # Qwen3_5DecoderLayer is decorated with support_torch_compile. During
        # Dynamo tracing, skip every host probe operation (clock/PID/JSON/span)
        # and invoke the original Python forward directly. This avoids introducing
        # probe graph breaks; layer spans therefore describe eager/capture-side
        # host execution only and are absent for graph replay.
        if event == "model_layer":
            try:
                import torch
                if torch.compiler.is_compiling():
                    return original(self,*args,**kwargs)
            except (AttributeError, ImportError):
                pass
        prev,sid,fields,started=before(self,args,kwargs)
        try:
            with _span(span_name(sid,fields)):
                result=original(self,*args,**kwargs)
            # EngineCore.step returns output containers; IDs are Python keys and
            # no device tensors are dereferenced here.
            if event == "engine_step" and isinstance(result,tuple) and result and isinstance(result[0],dict):
                _emit("engine_outputs", step_id=sid, request_ids=list(result[0].keys()), done=result[1] if len(result)>1 else None)
            if event == "graph_dispatch":
                # Same CPU arrays used by native uniform-decode eligibility.
                # No device tensor reads, and no assumption that q_len==1 means decode.
                request_position_state=None
                try:
                    ib=self.input_batch
                    n=kwargs.get('num_reqs',args[1] if len(args)>1 else len(ib.req_ids))
                    comp=ib.num_computed_tokens_cpu[:int(n)];prom=ib.num_prompt_tokens[:int(n)]
                    assert not hasattr(comp,'is_cpu') or comp.is_cpu
                    assert not hasattr(prom,'is_cpu') or prom.is_cpu
                    done=[int(a)>=int(b) for a,b in zip(comp,prom)]
                    ids=list(ib.req_ids[:int(n)])
                    if len(ids)==len(comp)==len(prom):
                        request_position_state=[{"request_id":rid,"num_computed_tokens":int(a),"num_prompt_tokens":int(b)}for rid,a,b in zip(ids,comp,prom)]
                    _TLS.step_phase='decode' if done and all(done) else 'mixed' if any(done) else 'prefill' if done else 'unknown'
                except Exception:_TLS.step_phase='unknown'
                mode, desc = result[:2] if isinstance(result,tuple) and len(result)>=2 else (None,None)
                _emit("graph_dispatch_result", step_id=sid, runner_instance=f"{os.getpid()}-{id(self)}",
                      actual_tokens=kwargs.get("num_tokens",args[0] if args else None), actual_requests=kwargs.get("num_reqs",args[1] if len(args)>1 else None), step_phase=getattr(_TLS,"step_phase","unknown"), dispatch_mode=str(mode), descriptor_class=type(desc).__name__ if desc is not None else None,
                      padded_tokens=getattr(desc,"num_tokens",None), padded_requests=getattr(desc,"num_reqs",None),
                      request_position_state=request_position_state,
                      capture_sizes=list(getattr(self,"cudagraph_batch_sizes",()) or ()))
            if event == "token_ready" and result is not None:
                _emit("token_ready", external_request_id=getattr(result,"request_id",None),
                      finished=getattr(result,"finished",None),
                      outputs=len(getattr(result,"outputs",()) or ()),
                      token_ids=sum(len(getattr(o,"token_ids",()) or ()) for o in (getattr(result,"outputs",()) or ())))
            return result
        finally: after(self,prev,sid,started,fields)
    setattr(cls,name,wrapped); _INSTALLED.add(key); return True

async def _observe_async_iterator(iterator, api_call_id):
    index=0
    outcome="complete"
    try:
        async for chunk in iterator:
            payload = None
            try:
                text = chunk.decode(errors="replace") if isinstance(chunk,bytes) else chunk
                if isinstance(text,str) and text.startswith("data:"):
                    data = text[5:].strip()
                    if data != "[DONE]": payload=json.loads(data)
            except (ValueError, TypeError):
                payload = None
            _emit("api_sse_chunk", api_call_id=api_call_id, external_request_id=(payload or {}).get("id") if isinstance(payload,dict) else None, chunk_index=index,
                  chunk_bytes=(len(chunk) if isinstance(chunk,(bytes,bytearray)) else
                               len(chunk.encode()) if isinstance(chunk,str) else None))
            index+=1
            yield chunk
    except BaseException:
        outcome="error_or_cancel"
        raise
    finally:
        _emit("api_sse_done", api_call_id=api_call_id, chunks=index, outcome=outcome)


def _api_request_fields(self,args,kwargs):
    request=args[0] if args else kwargs.get('request')
    raw=args[1] if len(args)>1 else kwargs.get('raw_request')
    headers=getattr(raw,'headers',None)
    return {'request_object_id':id(request) if request is not None else None,'external_request_id':headers.get('x-request-id') if headers is not None else None}

def _safe_schedule(self,args,kwargs):
    return {"running_before":len(getattr(self,"running",())), "waiting_before":len(getattr(self,"waiting",()))}

def _runner_fields(self,args,kwargs):
    so=args[0] if args else kwargs.get("scheduler_output")
    sched=getattr(so,"num_scheduled_tokens",None)
    return {"scheduled_tokens":dict(sched) if isinstance(sched,dict) else None,
            "total_scheduled_tokens":getattr(so,"total_num_scheduled_tokens",None),"new_request_prefill_state":[{"request_id":getattr(r,"req_id",None),"prompt_tokens":len(getattr(r,"prompt_token_ids",[])or[]),"num_computed_tokens":getattr(r,"num_computed_tokens",None)}for r in(getattr(so,"scheduled_new_reqs",[])or[])]}

def _layer_fields(self,args,kwargs):
    return {"layer_class":type(self).__module__+"."+type(self).__qualname__,
            "layer_idx":getattr(self,"layer_idx",getattr(self,"layer_number",None)),
            "layer_type":getattr(self,"layer_type",None), "component":"decoder_layer"}

def _output_fields(self,args,kwargs):
    outputs=args[0] if args else kwargs.get("engine_core_outputs",())
    return {"request_ids":[getattr(x,"request_id",None) for x in outputs],"output_count":len(outputs),
            "cpu_output_rows":[{"request_id":getattr(x,"request_id",None),"new_token_count":len(getattr(x,"new_token_ids",()) or ()),"finished":getattr(x,"finished",None)} for x in outputs],
            "engine_core_timestamp":args[1] if len(args)>1 else kwargs.get("engine_core_timestamp")}

def _detokenizer_fields(self,args,kwargs):
    token_ids=args[0] if args else kwargs.get("new_token_ids",())
    req=getattr(self,"request",None)
    return {"request_id":getattr(self,"_q27_request_id",getattr(req,"request_id",None)),"new_token_count":len(token_ids or ()),
            "stop_terminated":bool(args[1] if len(args)>1 else kwargs.get("stop_terminated",False))}

def _token_ready_fields(self,args,kwargs):
    return {"internal_request_id":getattr(self,"request_id",None),
            "external_request_id":getattr(self,"external_req_id",None),
            "output_kind":str(getattr(self,"output_kind",None))}


def install():
    """Patch classes after import. Returns a manifest of hooks that installed."""
    installed=[]
    def hook(module, clsname, method, event, ff=None):
        cls=getattr(module,clsname,None)
        if cls is not None and _wrap_method(cls,method,event,ff): installed.append(f"{module.__name__}.{clsname}.{method}")
    import sys
    for m in tuple(sys.modules.values()):
        if not m: continue
        name=getattr(m,"__name__","")
        if name in {'vllm.renderers.online_renderer','vllm.renderers.base','vllm.renderers.hf'}:
            import q27_render_stage_probe
            q27_render_stage_probe.install(m)
        if name.endswith("v1.core.sched.scheduler"): hook(m,"Scheduler","schedule","scheduler",_safe_schedule)
        elif name.endswith("vllm_ascend.worker.model_runner_v1"):
            hook(m,"NPUModelRunner","execute_model","worker_step",_runner_fields)
            hook(m,"NPUModelRunner","sample_tokens","sampler")
            hook(m,"NPUModelRunner","_determine_batch_execution_and_padding","graph_dispatch")
        elif name.endswith("v1.engine.core"): hook(m,"EngineCore","step","engine_step")
        elif name.endswith("model_executor.models.qwen3_5"):
            hook(m,"Qwen3_5DecoderLayer","forward","model_layer",_layer_fields)
        elif name.endswith("entrypoints.openai.completion.serving"):
            hook(m,"OpenAIServingCompletion","create_completion","api_completion",_api_request_fields)
        elif name.endswith("v1.engine.output_processor"):
            hook(m,"OutputProcessor","process_outputs","output_processing",_output_fields)
            hook(m,"RequestState","make_request_output","token_ready",_token_ready_fields)
        elif name.endswith("v1.engine.detokenizer"):
            cls=getattr(m,"BaseIncrementalDetokenizer",None)
            if cls is not None and not getattr(cls.__init__,"_q27_identity",False):
                init=cls.__init__
                @functools.wraps(init)
                def identified(self,request,*args,_init=init,**kwargs):
                    result=_init(self,request,*args,**kwargs)
                    self._q27_request_id=getattr(request,"request_id",None)
                    return result
                identified._q27_identity=True;cls.__init__=identified
            hook(m,"BaseIncrementalDetokenizer","update","detokenizer",_detokenizer_fields)
    import frontend_stage_probe_patch as front
    front.install_vllm(getattr(sys.modules.get("vllm.v1.engine.async_llm"),"__dict__",{}),getattr(sys.modules.get("vllm.v1.engine.input_processor"),"__dict__",{}),getattr(sys.modules.get("vllm.v1.engine.core_client"),"__dict__",{}))
    if os.environ.get("Q27_HYBRID_CACHE_PROBE")=="1":
        import q27_hybrid_vllm_probe
        q27_hybrid_vllm_probe.install(_emit,_span)
    _emit("install", hooks=installed, already_installed=len(_INSTALLED),
          model_directory={"decoder_layers":64,"gdn_layers":48,"full_attention_layers":16})
    return installed


class _Finder(importlib.abc.MetaPathFinder):
    """Install after exact target modules load, without editing their source."""
    TARGETS={"vllm.v1.worker.mamba_utils","vllm_ascend.patch.worker.patch_mamba_utils","vllm.v1.core.sched.scheduler","vllm_ascend.worker.model_runner_v1",
             "vllm.v1.engine.core","vllm.model_executor.models.qwen3_5",
             "vllm.entrypoints.openai.completion.serving","vllm.v1.engine.output_processor",
             "vllm.v1.engine.detokenizer","vllm.v1.engine.async_llm",
             "vllm.v1.engine.input_processor","vllm.v1.engine.core_client","vllm.renderers.online_renderer","vllm.renderers.base","vllm.renderers.hf"}
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.TARGETS: return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec and spec.loader and not isinstance(spec.loader,_Loader): spec.loader=_Loader(spec.loader)
        return spec

class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped): self.wrapped=wrapped
    def create_module(self,spec):
        fn=getattr(self.wrapped,"create_module",None); return fn(spec) if fn else None
    def exec_module(self,module):
        self.wrapped.exec_module(module); install()
        if os.environ.get("Q27_CAPTURE_CATALOG"):
            import q27_capture_catalog
            q27_capture_catalog.install(module)


def register_import_finder():
    if not any(isinstance(f,_Finder) for f in __import__("sys").meta_path):
        __import__("sys").meta_path.insert(0,_Finder())
    install()
    return True
