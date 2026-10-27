"""Opt-in CPU-only vLLM renderer/tokenizer stage observer."""
from __future__ import annotations
import asyncio, contextvars, functools, hashlib, json, os, threading, time
from pathlib import Path

_ENV="Q27_RENDER_PROBE_JSONL"
_SHA_ENV="Q27_RENDER_TEXT_SHA"
_LOCK=threading.RLock()
_ACTIVE={}
_CURRENT=contextvars.ContextVar("q27_render_probe_request",default=None)
_INSTALLED=set()
_OUT=None
_HASH_TEXT=False
_MAX_HASH_CHARS=8192

def _now():return time.monotonic_ns()
def _kind_len(x):
    if isinstance(x,dict):
        if "prompt_token_ids" in x:
            v=x.get("prompt_token_ids");return "pretokenized_ids",len(v) if isinstance(v,(list,tuple)) else None
        if "prompt_embeds" in x:return "embeds",None
        if isinstance(x.get("prompt"),str):return "text",len(x["prompt"])
        x=x.get("prompt")
    if isinstance(x,str):return "text",len(x)
    if isinstance(x,(bytes,bytearray)):return "bytes",len(x)
    if isinstance(x,(list,tuple)):
        if not x:return "empty_list",0
        if all(isinstance(i,int) and not isinstance(i,bool) for i in x):return "pretokenized_ids",len(x)
        if all(isinstance(i,str) for i in x):return "text_batch",sum(len(i) for i in x)
        if all(isinstance(i,(list,tuple)) and all(isinstance(j,int) and not isinstance(j,bool) for j in i) for i in x):return "pretokenized_batch",sum(len(i) for i in x)
        return "list",len(x)
    return type(x).__name__ if x is not None else "none",None
def _texts(x):
    if isinstance(x,dict):
        if isinstance(x.get("prompt"),str):yield x["prompt"]
        return
    if isinstance(x,str):yield x
    elif isinstance(x,(list,tuple)):
        for v in x:
            if isinstance(v,str):yield v
def _text_sha(s):
    if not _HASH_TEXT or len(s)>_MAX_HASH_CHARS:return None
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
def _describe(x):
    kind,n=_kind_len(x); texts=list(_texts(x))
    return {"path_type":kind,"input_length":n,"prompt_object_id":id(x) if x is not None else None,
            "text_object_ids":[id(s) for s in texts],
            "text_sha16":[h for s in texts if (h:=_text_sha(s)) is not None]}
def _write_unchecked(row):
    global _OUT
    if _OUT is None:return
    row.setdefault("pid",os.getpid());row.setdefault("thread_id",threading.get_ident());row.setdefault("ts_ns",_now())
    row.setdefault("mono_ns",row["ts_ns"]);row.setdefault("wall_ns",time.time_ns())
    with _LOCK:
        _OUT.parent.mkdir(parents=True,exist_ok=True)
        with _OUT.open("a",encoding="utf-8") as f:f.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
def _write(row):
    try:return _write_unchecked(row)
    except Exception:return None  # sink must never change an inference result
def _ctx_for_prompt(prompt):
    rid=_CURRENT.get()
    if rid is not None:return rid,"contextvar"
    d=_describe(prompt);oids=set(d["text_object_ids"]);hashes=set(d["text_sha16"])
    candidates=[]
    with _LOCK:
        for k,v in _ACTIVE.items():
            if oids.intersection(v["text_object_ids"]):candidates.append(k)
        if not candidates and hashes:
            for k,v in _ACTIVE.items():
                if hashes.intersection(v["text_sha16"]):candidates.append(k)
    if len(candidates)==1:return candidates[0],"unique_active_text_identity"
    if len(candidates)>1:return None,"ambiguous_active_text_identity"
    return None,"no_unique_request_context"
def _token_length(x):
    if isinstance(x,dict):
        ids=x.get("prompt_token_ids")
        if isinstance(ids,(list,tuple)):return len(ids)
        if isinstance(ids,int):return 1
    return None
def _invoke_sync(name,original,self,prompt,*args,**kwargs):
    rid,basis=_ctx_for_prompt(prompt);d=_describe(prompt);start=_now()
    _write({"event":name+"_begin","request_object_id":rid,"request_binding":basis,**d})
    try:
        result=original(self,prompt,*args,**kwargs)
    except BaseException as exc:
        _write({"event":name+"_end","request_object_id":rid,"request_binding":basis,"duration_ns":_now()-start,"status":"exception","exception_type":type(exc).__name__})
        raise
    _write({"event":name+"_end","request_object_id":rid,"request_binding":basis,"duration_ns":_now()-start,"status":"ok","output_token_length":_token_length(result)})
    return result
def _invoke_async(name,original,self,prompt,*args,**kwargs):
    async def call():
        rid,basis=_ctx_for_prompt(prompt);d=_describe(prompt);start=_now()
        _write({"event":name+"_begin","request_object_id":rid,"request_binding":basis,**d})
        try:result=await original(self,prompt,*args,**kwargs)
        except BaseException as exc:
            _write({"event":name+"_end","request_object_id":rid,"request_binding":basis,"duration_ns":_now()-start,"status":"exception","exception_type":type(exc).__name__})
            raise
        _write({"event":name+"_end","request_object_id":rid,"request_binding":basis,"duration_ns":_now()-start,"status":"ok","output_token_length":_token_length(result)})
        return result
    return call()
def _wrap_sync(cls,attr,event):
    key=(cls,attr)
    if key in _INSTALLED:return
    original=getattr(cls,attr,None)
    if original is None:return
    @functools.wraps(original)
    def wrapped(self,prompt,*args,**kwargs):return _invoke_sync(event,original,self,prompt,*args,**kwargs)
    setattr(cls,attr,wrapped);_INSTALLED.add(key)
def _wrap_async(cls,attr,event):
    key=(cls,attr)
    if key in _INSTALLED:return
    original=getattr(cls,attr,None)
    if original is None:return
    @functools.wraps(original)
    async def wrapped(self,prompt,*args,**kwargs):return await _invoke_async(event,original,self,prompt,*args,**kwargs)
    setattr(cls,attr,wrapped);_INSTALLED.add(key)
def _wrap_render(cls):
    attr="render_completion";key=(cls,attr)
    if key in _INSTALLED:return
    original=getattr(cls,attr,None)
    if original is None:return
    @functools.wraps(original)
    async def wrapped(self,request,*args,**kwargs):
        rid=id(request);raw=getattr(request,"prompt",None);desc=_describe(raw);start=_now()
        active={"text_object_ids":set(desc["text_object_ids"]),"text_sha16":set(desc["text_sha16"])}
        with _LOCK:_ACTIVE[rid]=active
        token=_CURRENT.set(rid)
        _write({"event":"render_completion_begin","request_object_id":rid,**desc})
        try:result=await original(self,request,*args,**kwargs)
        except BaseException as exc:
            _write({"event":"render_completion_end","request_object_id":rid,"duration_ns":_now()-start,"status":"exception","exception_type":type(exc).__name__})
            raise
        else:
            _write({"event":"render_completion_end","request_object_id":rid,"duration_ns":_now()-start,"status":"ok","result_type":type(result).__name__})
            return result
        finally:
            _CURRENT.reset(token)
            with _LOCK:_ACTIVE.pop(rid,None)
    setattr(cls,attr,wrapped);_INSTALLED.add(key)
def install(namespace,output_path=None,hash_text=None):
    """Patch renderer classes in one imported module namespace, before renderer instances exist.

    Call for online_renderer, base, and hf module namespaces as their import-finder
    hooks fire. Repeated calls are idempotent. No service state or tensors are read.
    """
    global _OUT,_HASH_TEXT
    path=output_path or os.environ.get(_ENV)
    if path:
        _OUT=Path(path)
    if hash_text is None:_HASH_TEXT=os.environ.get(_SHA_ENV,"0").lower() in {"1","true","yes"}
    else:_HASH_TEXT=bool(hash_text)
    ns=vars(namespace) if not isinstance(namespace,dict) else namespace
    patched=[]
    online=ns.get("OnlineRenderer")
    if isinstance(online,type):_wrap_render(online);patched.append("OnlineRenderer.render_completion")
    base=ns.get("BaseRenderer")
    if isinstance(base,type):
        _wrap_sync(base,"_tokenize_prompt","base_tokenize_prompt")
        _wrap_sync(base,"tokenize_prompt","base_tokenize_prompt_sync_path")
        _wrap_async(base,"tokenize_prompt_async","base_tokenize_prompt_async_path")
        patched.append("BaseRenderer._tokenize_prompt/tokenize_prompt[_async]")
    hf=ns.get("HfRenderer")
    if isinstance(hf,type) and "_tokenize_prompt" in hf.__dict__:
        _wrap_sync(hf,"_tokenize_prompt","hf_tokenize_prompt_override")
        patched.append("HfRenderer._tokenize_prompt")
    return {"installed":bool(patched),"patched":patched,"output":str(_OUT) if _OUT else None,"text_sha_enabled":_HASH_TEXT,
            "hf_tokenize_override_present":bool(isinstance(hf,type) and "_tokenize_prompt" in hf.__dict__)}
