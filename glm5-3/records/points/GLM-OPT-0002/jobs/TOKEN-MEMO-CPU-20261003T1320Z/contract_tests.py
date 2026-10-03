"""CPU mock contract tests; not native inference or performance evidence."""
import asyncio, json, tempfile
from pathlib import Path
import httpx
from native_chat_token_memo import NativeChatTokenMemoTransport as Memo

ORIGIN="http://native.invalid:9081"
def body(text="hello",**fields):
    return json.dumps(dict(model="glm-52",messages=[dict(role="user",content=text)],
                           stream=True,cache_salt="a",**fields)).encode()
def request(data=None,url=None,headers=None):
    return httpx.Request("POST",url or ORIGIN+"/v1/chat/completions",
                         content=data if data is not None else body(),
                         headers=headers or {"content-type":"application/json"})
async def run():
    checks=[];calls=[];gate=asyncio.Event();gate.set();fail=[False]
    wire=b'data: {"native":true}\n\ndata: [DONE]\n\n'
    async def backend(req):
        calls.append((req.url.path,await req.aread(),list(req.headers.multi_items())))
        if req.url.path=="/tokenize":
            await gate.wait()
            if fail[0]: return httpx.Response(400,content=b'{"error":"native template error"}')
            return httpx.Response(200,json=dict(tokens=[1,2,3],count=3,max_model_len=144384))
        return httpx.Response(418,content=wire,headers=[("x-native","a"),("x-native","b")])
    with tempfile.TemporaryDirectory() as td:
        memo=Memo(httpx.MockTransport(backend),{ORIGIN:"epoch1"},audit_dir=td,
                  trace_path=str(Path(td)/"trace"),stats_path=str(Path(td)/"stats"))
        a=await memo.handle_async_request(request()); assert a.status_code==418 and await a.aread()==wire
        original=json.loads(body());forward=json.loads(calls[-1][1])
        assert forward.pop("kv_transfer_params")=={"prompt_token_ids":[1,2,3]} and forward==original
        assert int(dict(calls[-1][2])["content-length"])==len(calls[-1][1])
        assert a.headers.get_list("x-native")==["a","b"]; checks.append("native status/wire/duplicate headers and sampling preserved")
        await memo.handle_async_request(request(body(cache_salt="different") if False else
             json.dumps(dict(original,cache_salt="different",temperature=0.1)).encode()))
        assert memo.stats["hits"]==1 and memo.stats["tokenizer_CPU_calls"]==1
        checks.append("same message with changed salt/sampling exact token cache hit")
        await memo.handle_async_request(request(body(chat_template_kwargs={"enable_thinking":False})))
        assert memo.stats["tokenizer_CPU_calls"]==2;checks.append("template kwargs bound to cache")
        memo.epochs[ORIGIN]="epoch2";await memo.handle_async_request(request())
        assert memo.stats["tokenizer_CPU_calls"]==3;checks.append("epoch changes cannot hit old cache key")
        ineligible=[
            json.dumps(dict(original,tools=[])).encode(),
            json.dumps(dict(original,kv_transfer_params={"prompt_token_ids":[9]})).encode(),
            json.dumps(dict(original,messages=[{"role":"system","content":"x"},{"role":"user","content":"y"}])).encode(),
            json.dumps(dict(original,messages=[{"role":"user","content":[{"type":"text","text":"hello"}]}])).encode(),
            json.dumps(dict(original,chat_template_kwargs={"unknown":True})).encode(),
            json.dumps(dict(original,truncate_prompt_tokens=1)).encode(),b'invalid json']
        for data in ineligible:
            before=memo.stats["tokenizer_CPU_calls"];await memo.handle_async_request(request(data))
            assert calls[-1][1]==data and memo.stats["tokenizer_CPU_calls"]==before
        for req in [request(url=ORIGIN+"/v1/chat/completions?x=1"),
                    request(url="http://foreign.invalid/v1/chat/completions"),
                    request(headers={"content-type":"application/json","digest":"unchanged"})]:
            before=memo.stats["tokenizer_CPU_calls"];data=await req.aread()
            await memo.handle_async_request(req);assert calls[-1][1]==data and memo.stats["tokenizer_CPU_calls"]==before
        checks.append("unknown/state/tools/multimodal/truncation/query/signed/foreign bypass byte exact")
        fail[0]=True;before=memo.stats["lookup_errors"]
        data=body("error input");await memo.handle_async_request(request(data))
        assert calls[-1][1]==data and memo.stats["lookup_errors"]==before+1
        fail[0]=False;checks.append("tokenize error returns original native chat path without retry")
        gate.clear();before=memo.stats["tokenizer_CPU_calls"]
        t1=asyncio.create_task(memo.handle_async_request(request(body("concurrent"))))
        await asyncio.sleep(0.01)
        t2=asyncio.create_task(memo.handle_async_request(request(body("concurrent"))))
        await asyncio.sleep(0.01);t1.cancel()
        try: await t1
        except asyncio.CancelledError: pass
        gate.set();await t2;await asyncio.sleep(0)
        assert memo.stats["tokenizer_CPU_calls"]==before+1 and not memo.pending
        checks.append("singleflight and cancelled caller retain bounded CPU lookup; no extra generation")
        limit=max(v[3]for v in memo.cache.values());memo.max_bytes=limit
        while memo.cache:
            _,v=memo.cache.popitem(last=False);memo.used-=v[3]
        await memo.handle_async_request(request(body("evict1")))
        await memo.handle_async_request(request(body("evict2")))
        assert memo.used<=limit and len(memo.cache)<=1 and memo.stats["evictions"]>0
        memo.max_bytes=1;await memo.handle_async_request(request(body("oversize")))
        assert memo.stats["oversize_entries"]>0
        checks.append("LRU accounting with source metadata, eviction and oversize bound")
        memo.max_bytes=33554432;memo.max_pending=1;gate.clear()
        pending=asyncio.create_task(memo.handle_async_request(request(body("pending1"))))
        await asyncio.sleep(0.01);before=memo.stats["tokenizer_CPU_calls"]
        overflowbody=body("pending2");await memo.handle_async_request(request(overflowbody))
        assert calls[-1][1]==overflowbody and memo.stats["tokenizer_CPU_calls"]==before and memo.stats["pending_overflow"]==1
        await memo.aclose()
        try:await pending
        except asyncio.CancelledError:pass
        assert not memo.pending;checks.append("pending overflow original-path fallback and shutdown cancels CPU work")
        rows=[json.loads(l)for l in (Path(td)/"trace").read_text().splitlines()]
        assert rows[-1]["event"]=="chat_token_cache_closed"
        assert json.loads((Path(td)/"stats").read_text())["pending"]==0
        checks.append("parseable trace/stats with newline delimiter")
        broken=Memo(httpx.MockTransport(backend),{ORIGIN:"epoch1"})
        gate.set();broken.audit_dir=Path(td)/"stats";broken.trace_path=str(Path(td)/"missing"/"trace")
        res=await broken.handle_async_request(request());assert await res.aread()==wire and broken.stats["audit_errors"]>0
        await broken.aclose();checks.append("audit I/O failure does not corrupt native response")
    return dict(simulation=True,native_model_calls=0,passed=True,checks=checks)
