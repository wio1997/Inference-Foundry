"""GLM control boundary on the installed full PD proxy.

Default behaviour is the installed compatibility implementation. Optional tracing
observes request-stage boundaries without parsing/replacing forwarded SSE data.
Use a task-owned port; this module does not replace the live proxy by itself.
"""
import contextvars, functools, hashlib, importlib.util, json, os, sys, time, uuid
from pathlib import Path

SOURCE=os.environ.get('GLM_STOCK_PROXY','/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py')
EXPECTED_SHA=os.environ.get('GLM_STOCK_PROXY_SHA256','38278b82dbaa1bb6d854e622b95844c3b1381276486d596859a2dc35602a26d5')
trace_context=contextvars.ContextVar('glm_trace_context',default=None)

class TraceSink:
    def __init__(self,path):
        self.path=path;self.fd=None
        if path:
            self.fd=os.open(path,os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
    def emit(self,event,**fields):
        context=trace_context.get()
        if self.fd is None or context is None:return
        payload={'trace_id':context['trace_id'],'event':event,'monotonic_ns':time.monotonic_ns(),'pid':os.getpid(),**fields}
        try:os.write(self.fd,(json.dumps(payload,separators=(',',':'))+'\n').encode())
        except OSError:context['trace_write_failed']=True

class StageTraceMiddleware:
    def __init__(self,app,sink):self.app=app;self.sink=sink
    async def __call__(self,scope,receive,send):
        if scope.get('type')!='http' or self.sink.fd is None:
            return await self.app(scope,receive,send)
        context={'trace_id':uuid.uuid4().hex,'trace_write_failed':False}
        token=trace_context.set(context);first_body=True
        self.sink.emit('request_received',path=scope.get('path'),method=scope.get('method'))
        async def receive_observed():
            message=await receive()
            if message['type']=='http.disconnect':self.sink.emit('client_disconnect')
            elif message['type']=='http.request':
                self.sink.emit('request_body_chunk',bytes=len(message.get('body',b'')),more=message.get('more_body',False))
            return message
        async def send_observed(message):
            nonlocal first_body
            if message['type']=='http.response.start':self.sink.emit('response_start',status=message['status'])
            elif message['type']=='http.response.body':
                body=message.get('body',b'')
                if first_body and body:self.sink.emit('response_first_bytes',bytes=len(body));first_body=False
                if not message.get('more_body',False):self.sink.emit('response_last_bytes',bytes=len(body))
            await send(message)
        try:return await self.app(scope,receive_observed,send_observed)
        finally:
            self.sink.emit('request_finished',trace_write_failed=context['trace_write_failed'])
            trace_context.reset(token)


def instrument_stock(stock,sink):
    """Keep native call/return, retry, cancellation and payload contracts intact."""
    if sink.fd is None:return
    for name in ('assign_instances','send_request_to_service','_finish_instance'):
        native=getattr(stock,name)
        def decorate(native,name):
            @functools.wraps(native)
            async def observed(*args,**kwargs):
                fields={}
                if name=='send_request_to_service':fields={'endpoint':args[1] if len(args)>1 else kwargs.get('endpoint'),'backend_request_id':args[3] if len(args)>3 else kwargs.get('request_id')}
                sink.emit(name+'_begin',**fields)
                try:
                    result=await native(*args,**kwargs)
                    if name=='assign_instances':
                        sink.emit(name+'_end',backend_request_id=result.request_id,prefiller_key=result.prefiller_key,decoder_key=result.decoder_key,prefiller_cached_tokens=result.prefiller_cached_tokens)
                    else:sink.emit(name+'_end',**fields)
                    return result
                except BaseException as error:
                    sink.emit(name+'_error',error_type=type(error).__name__,**fields);raise
            return observed
        setattr(stock,name,decorate(native,name))
    native_stream=stock.stream_service_response
    @functools.wraps(native_stream)
    async def observed_stream(*args,**kwargs):
        fields={'backend_request_id':kwargs.get('request_id',args[3] if len(args)>3 else None)}
        sink.emit('decode_open_begin',**fields);first=True
        generator=native_stream(*args,**kwargs)
        try:
            async for chunk in generator:
                if first:sink.emit('decode_first_bytes',bytes=len(chunk),**fields);first=False
                yield chunk
        except BaseException as error:
            sink.emit('decode_stream_error',error_type=type(error).__name__,**fields);raise
        finally:
            await generator.aclose();sink.emit('decode_stream_closed',**fields)
    stock.stream_service_response=observed_stream


def load_stock():
    path=Path(SOURCE);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=EXPECTED_SHA:raise RuntimeError('installed proxy identity differs; recover source before loading')
    name='_glm_stock_proxy';spec=importlib.util.spec_from_file_location(name,path);stock=importlib.util.module_from_spec(spec);sys.modules[name]=stock;spec.loader.exec_module(stock);return stock

stock=None
sink=None

def create_app():
    global sink,stock
    if stock is None:stock=load_stock()
    sink=TraceSink(os.environ.get('GLM_TRACE_PATH'));instrument_stock(stock,sink)
    return StageTraceMiddleware(stock.create_app(),sink)

if __name__=='__main__':
    stock=load_stock()
    stock.global_args=stock.parse_args();stock.setup_logging(stock.global_args.log_level);stock.bootstrap_parent_process(stock.global_args)
    import uvicorn
    try:
        uvicorn.run('glm_gateway:create_app',host=stock.global_args.host,port=stock.global_args.port,workers=stock.global_args.workers,factory=True,app_dir=str(Path(__file__).resolve().parent))
    finally:stock.cleanup_manager_config(stock.global_args.port)
