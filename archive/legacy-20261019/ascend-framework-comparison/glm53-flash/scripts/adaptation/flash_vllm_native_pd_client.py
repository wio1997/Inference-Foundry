# SPDX-License-Identifier: Apache-2.0
"""Strict native vLLM P->D completion caller; no calls on import.

Use only inside a separately frozen ZCode runtime unit. One P POST and one D
POST, no retry, fallback, token truncation or artificial cache hit accounting.
Client monotonic time includes prefill, handoff, decode and network latency.
"""
import json,time,urllib.request,urllib.error,uuid,math

class StreamEvidence:
    def __init__(self,issued_ns):
        self.issued_ns=issued_ns;self.first_text_ns=None;self.done=False
        self.finish_reason=None;self.stop_reason=None;self.usage=None
        self.text_parts=[];self.response_id=None;self.events=0

    def event(self,data,received_ns):
        if self.done:raise ValueError('SSE data after DONE')
        if data=='[DONE]':self.done=True;return
        row=json.loads(data);self.events+=1
        if 'error' in row:raise ValueError(('NativeAPIStreamError',row['error']))
        if row.get('id'):self.response_id=row['id']
        if row.get('usage') is not None:self.usage=row['usage']
        for c in row.get('choices',[]):
            if c.get('index',0)!=0:raise ValueError('Expected exactly one completion')
            text=c.get('text','')
            if not isinstance(text,str):raise ValueError('Malformed completion text')
            if text:
                if self.first_text_ns is None:self.first_text_ns=received_ns
                self.text_parts.append(text)
            if c.get('finish_reason') is not None:
                self.finish_reason=c['finish_reason'];self.stop_reason=c.get('stop_reason')

    def report(self,terminal_ns,expected_input,expected_output):
        usage=self.usage or {};inp=usage.get('prompt_tokens');out=usage.get('completion_tokens')
        valid=type(inp) is int and type(out) is int and inp==expected_input and out==expected_output
        return {'SSE_DONE':self.done,'finish_reason':self.finish_reason,'stop_reason':self.stop_reason,
            'response_id':self.response_id,'usage':self.usage,'native_events':self.events,
            'TTFT_s':None if self.first_text_ns is None else (self.first_text_ns-self.issued_ns)/1e9,
            'E2E_s':(terminal_ns-self.issued_ns)/1e9,
            'TPOT_s':None if self.first_text_ns is None or type(out) is not int or out<2 else (terminal_ns-self.first_text_ns)/1e9/(out-1),
            'text':''.join(self.text_parts),'exact_token_usage':valid,
            'success':bool(self.done and self.finish_reason is not None and self.first_text_ns is not None and valid)}

def _payload_ids(ids):
    if not isinstance(ids,list) or not ids or any(type(v) is not int or not 0<=v<154856 for v in ids):
        raise ValueError('Expected actual Flash tokenizer IDs, vocab154856/model padded154880')

class NativeVllmPDClient:
    def __init__(self,*,prefill_url,decode_url,model,timeout=1800.):
        if timeout<=0 or not model:raise ValueError('Invalid native Flash client')
        allowed_p={f'http://172.16.12.14:{19000+i}/v1/completions' for i in range(2)}
        allowed_d={f'http://172.16.12.15:{19020+i}/v1/completions' for i in range(16)}
        if prefill_url not in allowed_p or decode_url not in allowed_d:raise ValueError('Outside Flash owned API topology')
        self.prefill_url=prefill_url;self.decode_url=decode_url;self.model=model;self.timeout=float(timeout)

    def generate(self,ids,*,output_tokens=600,request_id=None,retain_text=True):
        _payload_ids(ids)
        if type(output_tokens) is not int or output_tokens<=0 or len(ids)+output_tokens>133120:raise ValueError('No context truncation allowed')
        request_id=request_id or 'flash-'+uuid.uuid4().hex
        if not isinstance(request_id,str) or not 1<=len(request_id)<=128 or not all(c.isalnum() or c in '-_.:' for c in request_id):raise ValueError('Invalid request ID')
        issued=time.monotonic_ns();ev=StreamEvidence(issued);result={'project_id':'glm53-flash','request_id':request_id,'logical_input_tokens':len(ids),'requested_output_tokens':output_tokens,'issued_monotonic_ns':issued,'generation_POST_attempted':0,'HTTP_responses_received':0,'success':False,'prefill_url':self.prefill_url,'decode_url':self.decode_url};stage='prefill'
        headers={'Content-Type':'application/json','X-Request-Id':request_id}
        common={'model':self.model,'prompt':ids,'temperature':0.,'seed':1024,'ignore_eos':True}
        try:
            pbody={**common,'max_tokens':1,'stream':False,'kv_transfer_params':{'do_remote_decode':True,'do_remote_prefill':False}}
            req=urllib.request.Request(self.prefill_url,data=json.dumps(pbody,separators=(',',':')).encode(),headers=headers)
            result['generation_POST_attempted']+=1
            with urllib.request.urlopen(req,timeout=self.timeout) as response:
                result['HTTP_responses_received']+=1;result['prefill_HTTP_status']=response.status;prefill=json.load(response)
            result['prefill_response_id']=prefill.get('id');result['prefill_usage']=prefill.get('usage');result['prefill_terminal_monotonic_ns']=time.monotonic_ns()
            if prefill.get('usage',{}).get('prompt_tokens')!=len(ids):raise ValueError('Prefill actual input usage mismatch')
            params=prefill.get('kv_transfer_params')
            if not isinstance(params,dict) or params.get('do_remote_prefill') is not True:raise ValueError('Missing actual native PD metadata; recomputation forbidden')
            result['native_transfer_params']=params
            dbody={**common,'max_tokens':output_tokens,'stream':True,'stream_options':{'include_usage':True},'kv_transfer_params':params}
            req=urllib.request.Request(self.decode_url,data=json.dumps(dbody,separators=(',',':')).encode(),headers=headers);stage='decode';result['generation_POST_attempted']+=1
            with urllib.request.urlopen(req,timeout=self.timeout) as response:
                result['HTTP_responses_received']+=1;result['decode_HTTP_status']=response.status;data=[]
                while True:
                    raw=response.readline();received=time.monotonic_ns()
                    if not raw:break
                    line=raw.decode('utf-8').rstrip('\r\n')
                    if not line:
                        if data:
                            ev.event('\n'.join(data),received);data=[]
                            if ev.done:break
                    elif line.startswith('data:'):data.append(line[5:].lstrip(' '))
                    elif line.startswith(':'):continue
                if data:raise ValueError('SSE ended before final event boundary')
        except urllib.error.HTTPError as exc:
            result['HTTP_responses_received']+=1;result[stage+'_HTTP_status']=exc.code;result['error']=repr(exc)
            try:result['native_HTTP_error_body']=exc.read(65536).decode(errors='replace')
            except Exception as body_error:result['HTTP_error_body_read_error']=repr(body_error)
        except Exception as exc:result['error']=repr(exc)
        terminal=time.monotonic_ns();result.update(ev.report(terminal,len(ids),output_tokens));result['terminal_monotonic_ns']=terminal
        result['success']=result['success'] and not result.get('error') and result.get('prefill_HTTP_status')==result.get('decode_HTTP_status')==200
        if not retain_text:result.pop('text',None)
        return result
