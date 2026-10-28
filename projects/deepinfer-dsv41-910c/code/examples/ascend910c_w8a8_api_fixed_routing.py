"""HTTP completions frontend for the native DeepInfer file-IPC workers.

Runs in a separate CPU container. Only the deployed tokenizer is reused from
vLLM; there is no engine, model runner, scheduler or forwarded inference API.
"""
import argparse
from dataclasses import dataclass,field
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid


@dataclass
class Request:
    token_ids:list
    max_tokens:int
    ignore_eos:bool
    id:str=field(default_factory=lambda:'cmpl-'+uuid.uuid4().hex)
    events:queue.Queue=field(default_factory=queue.Queue)


class Frontend:
    def __init__(self,args):
        from vllm.tokenizers import get_tokenizer
        self.args=args
        self.tokenizer=get_tokenizer(args.model_dir,tokenizer_mode='deepseek_v41')
        config=json.loads((Path(args.model_dir)/'config.json').read_text())
        self.eos=config['eos_token_id']
        from deepinfer.engine.tokenization_cache import TokenizationCache
        self.token_cache=TokenizationCache(self.tokenizer.encode,token_budget=args.tokenizer_cache_tokens)
        self.pending=queue.Queue()
        self.counts={'requests':0,'prompt_tokens':0,'cached_tokens':0,'output_tokens':0,'failed_requests':0,
                     'tokenizer_cache_hits':0,'tokenizer_cache_hit_tokens':0}
        self.lock=threading.Lock()

    def ready(self):
        return len(list(self.args.control_dir.glob('ready-rank*.json')))==16

    def scheduler(self):
        while True:
            first=self.pending.get()
            batch=[first]
            deadline=time.monotonic()+self.args.batch_wait_ms/1000
            while len(batch)<8:
                try:
                    batch.append(self.pending.get(timeout=max(0,deadline-time.monotonic())))
                except queue.Empty:
                    break
            # Experiment-only canonical assignment: paired modes must send
            # the same requests to the same DP ranks and slots.
            if os.environ.get('DEEPINFER_SINGLE_POINT_ABLATION')=='1' and len(batch)==8:
                batch.sort(key=lambda request:tuple(request.token_ids))
            jobid=uuid.uuid4().hex
            directory=self.args.control_dir/'jobs'/jobid
            directory.mkdir(parents=True)
            body={'eos_token_id':self.eos,'requests':[{'id':r.id,'token_ids':r.token_ids,
                'max_tokens':r.max_tokens,'ignore_eos':r.ignore_eos} for r in batch]}
            if os.environ.get('DEEPINFER_SINGLE_POINT_ABLATION')=='1':
                mode=json.loads((self.args.control_dir/'single-point-mode.json').read_text())
                if (not isinstance(mode,dict) or set(mode)!={'varlen','tail_ffn','engram_prefetch','engram_pinned','engram_uva'}
                        or any(type(value) is not bool for value in mode.values())):
                    raise ValueError('invalid frozen single-point mode')
                body['single_point_ablation']=mode
            (directory/'request.json').write_text(json.dumps(body,separators=(',',':')))
            pending=self.args.control_dir/'current.tmp'
            pending.write_text(json.dumps({'id':jobid}))
            pending.replace(self.args.control_dir/'current.json')
            readers={}
            done=set()
            started=time.monotonic()
            try:
                while len(done)<len(batch):
                    errors=list(directory.glob('error-rank*.txt'))
                    if errors:
                        raise RuntimeError(errors[0].read_text()[-5000:])
                    if time.monotonic()-started>self.args.request_timeout:
                        raise TimeoutError('native inference batch timed out')
                    for dp in range(2):
                        path=directory/f'events-dp{dp}.jsonl'
                        if dp not in readers and path.exists():
                            readers[dp]=path.open()
                        if dp not in readers:
                            continue
                        handle=readers[dp]
                        while True:
                            offset=handle.tell()
                            line=handle.readline()
                            if not line:
                                break
                            if not line.endswith('\n'):
                                handle.seek(offset);break
                            event=json.loads(line)
                            batch[event['index']].events.put(event)
                            if event['type']=='done':
                                done.add(event['index'])
                                with self.lock:
                                    self.counts['requests']+=1
                                    for key in ('prompt_tokens','cached_tokens','output_tokens'):
                                        self.counts[key]+=event[key]
                    time.sleep(.001)
                # Workers retain ownership of the global EP schedule until
                # both DP replicas finish; do not overwrite IPC with a new job.
                while len(list(directory.glob('result-dp*.json')))<2:
                    errors=list(directory.glob('error-rank*.txt'))
                    if errors:
                        raise RuntimeError(errors[0].read_text()[-5000:])
                    if time.monotonic()-started>self.args.request_timeout:
                        raise TimeoutError('native worker completion timed out')
                    time.sleep(.005)
            except Exception as exc:
                for i,request in enumerate(batch):
                    if i not in done:
                        request.events.put({'type':'error','message':str(exc)})
                with self.lock:
                    self.counts['failed_requests']+=len(batch)-len(done)
            finally:
                for handle in readers.values():
                    handle.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',required=True)
    p.add_argument('--control-dir',type=Path,required=True)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8172)
    p.add_argument('--batch-wait-ms',type=int,default=500)
    p.add_argument('--tokenizer-cache-tokens',type=int,default=0)
    p.add_argument('--request-timeout',type=int,default=3600)
    args=p.parse_args()
    frontend=Frontend(args)

    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'

        def send_json(self,value,status=200):
            encoded=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(encoded)))
            self.end_headers();self.wfile.write(encoded)

        def do_GET(self):
            if self.path=='/health':
                self.send_json({'status':'ok' if frontend.ready() else 'loading','backend':'DeepInfer-native'},200 if frontend.ready() else 503)
            elif self.path=='/v1/models':
                self.send_json({'object':'list','data':[{'id':'dsv41','object':'model','owned_by':'DeepInfer'}]})
            elif self.path=='/metrics':
                text=''.join(f'deepinfer_{key}_total {value}\n' for key,value in frontend.counts.items()).encode()
                self.send_response(200);self.send_header('Content-Type','text/plain');self.send_header('Content-Length',str(len(text)));self.end_headers();self.wfile.write(text)
            else:
                self.send_json({'error':'unknown endpoint'},404)

        def do_POST(self):
            if self.path!='/v1/completions':
                self.send_json({'error':'use /v1/completions'},404);return
            if not frontend.ready():
                self.send_json({'error':'native workers loading'},503);return
            try:
                body=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
                if body.get('model','dsv41')!='dsv41' or body.get('n',1)!=1:
                    raise ValueError('model=dsv41 and n=1 are required')
                # Greedy is explicit; unsupported stochastic settings fail.
                if body.get('temperature',0) not in (0,0.0):
                    raise ValueError('native backend currently supports temperature=0')
                prompt=body['prompt']
                tokenizer_hit=False
                if isinstance(prompt,str):
                    ids,tokenizer_hit=frontend.token_cache.get(prompt)
                    if tokenizer_hit:
                        with frontend.lock:
                            frontend.counts['tokenizer_cache_hits']+=1
                            frontend.counts['tokenizer_cache_hit_tokens']+=len(ids)
                elif isinstance(prompt,list) and all(type(i) is int for i in prompt):
                    ids=prompt
                else:
                    raise ValueError('prompt must be text or one token-ID list')
                maximum=int(body.get('max_tokens',16))
                if not ids or maximum<1 or len(ids)+maximum>34816:
                    raise ValueError('prompt/output must fit context34816')
                request=Request(ids,maximum,bool(body.get('ignore_eos',False)))
            except (ValueError,KeyError,TypeError) as exc:
                self.send_json({'error':str(exc)},400);return
            frontend.pending.put(request)
            streaming=bool(body.get('stream',False))
            created=int(time.time())
            if streaming:
                self.send_response(200)
                self.send_header('Content-Type','text/event-stream')
                self.send_header('Cache-Control','no-cache')
                self.send_header('Connection','close')
                self.end_headers()
                self.close_connection=True
            generated=[];published='';hit=0;timings=[]
            def send_chunk(text='',finish=None,usage=None):
                message={'id':request.id,'object':'text_completion','created':created,'model':'dsv41',
                         'choices':[{'index':0,'text':text,'logprobs':None,'finish_reason':finish}]}
                if usage is not None:
                    message['usage']=usage
                self.wfile.write(('data: '+json.dumps(message,ensure_ascii=False)+'\n\n').encode());self.wfile.flush()
            try:
                while True:
                    event=request.events.get(timeout=args.request_timeout)
                    if event['type']=='error':
                        if streaming:
                            self.wfile.write(('data: '+json.dumps({'error':event['message']})+'\n\n').encode());self.wfile.flush()
                        else:
                            self.send_json({'error':event['message']},500)
                        return
                    if event['type']=='start':
                        hit=event['cached_tokens']
                    elif event['type']=='token':
                        generated.append(event['token_id']);timings.append(time.time())
                        decoded=frontend.tokenizer.decode(generated,skip_special_tokens=False)
                        # Keep an incomplete UTF8 character buffered until the
                        # following token resolves it; never emit replacement.
                        if not decoded.endswith('\ufffd') and decoded.startswith(published):
                            delta=decoded[len(published):];published=decoded
                            if streaming:
                                send_chunk(delta)
                    elif event['type']=='done':
                        decoded=frontend.tokenizer.decode(generated,skip_special_tokens=False)
                        usage={'prompt_tokens':len(ids),'completion_tokens':len(generated),
                            'total_tokens':len(ids)+len(generated),
                            'prompt_tokens_details':{'cached_tokens':hit}}
                        if streaming:
                            send_chunk(decoded[len(published):] if decoded.startswith(published) else '',event['finish_reason'],usage)
                            self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
                        else:
                            self.send_json({'id':request.id,'object':'text_completion','created':created,'model':'dsv41',
                                'choices':[{'index':0,'text':decoded,'logprobs':None,'finish_reason':event['finish_reason']}],
                                'usage':usage})
                        audit={'id':request.id,'tokenizer_cache_hit':tokenizer_hit,'input_tokens':len(ids),'cached_tokens':hit,
                            'prompt_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                            'generated_token_ids':generated,'text':decoded,'token_times':timings,'finish_reason':event['finish_reason']}
                        with frontend.lock:
                            with (args.control_dir/'api-results.jsonl').open('a') as f:
                                f.write(json.dumps(audit,ensure_ascii=False)+'\n')
                        return
            except (BrokenPipeError,ConnectionResetError):
                # Scheduler/worker still finish the submitted EP batch.
                return

        def log_message(self,fmt,*values):
            print(fmt%values,flush=True)

    threading.Thread(target=frontend.scheduler,daemon=True).start()
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(json.dumps({'stage':'http_listening','host':args.host,'port':args.port,'backend':'DeepInfer native'}),flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
