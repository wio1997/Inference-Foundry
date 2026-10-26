"""Project-local vLLM pull relay; streams only D output, all payload uses connector SDMA."""
import asyncio
import hashlib
import json
import re
import struct
import time

from aiohttp import web


def route(prefix):
    assert type(prefix) is int and 0 <= prefix < 64
    p_rank=prefix%4;d_rank=prefix%16
    return {'P':{'node':30 if p_rank<2 else 31,'global_dp':p_rank,'wire_local_dp':p_rank%2,'port':30300},
            'D':{'node':43 if d_rank<8 else 45,'global_dp':d_rank,'wire_local_dp':d_rank%8,'port':30400}}


class VLLMPullGateway:
    def __init__(self, session, allowed_namespaces, rows):
        self.session=session;self.allowed_namespaces=frozenset(allowed_namespaces);self.rows=rows

    def application(self):
        app=web.Application(client_max_size=16*1024**2)
        app.router.add_post('/v1/completions',self.completion)
        return app

    async def completion(self, request):
        row={'gateway_dispatch_monotonic_ns':time.monotonic_ns(),'status':'VALIDATING'}
        stream=None
        try:
            body=await request.json();ids=body.get('prompt');rid=body.get('request_id','')
            assert isinstance(ids,list) and all(type(x) is int and x>=0 for x in ids)
            assert (len(ids),body.get('max_tokens')) in ((74400,1),(80000,600))
            assert body.get('min_tokens')==body['max_tokens'] and body.get('ignore_eos') is True
            assert body.get('temperature')==0 and body.get('seed')==20261017
            assert body.get('stream') is True and body.get('model')=='glm53-w8a8'
            assert body.get('return_token_ids') is False and body.get('skip_special_tokens') is False
            assert body.get('cache_salt') in self.allowed_namespaces and not body.get('kv_transfer_params')
            match=re.search(r'(?:^|-)p(\d+)(?:-|$)',rid);assert match
            prefix=int(match.group(1));selected=route(prefix)
            row.update(logical_request_id=rid,namespace=body['cache_salt'],prefix_id=prefix,route=selected,
                input_tokens=len(ids),output_target=body['max_tokens'],
                input_sha256_le_u32=hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest())
            p=selected['P'];d=selected['D']
            prefill=dict(body);prefill.update(request_id=rid+'.P',max_tokens=1,min_tokens=1,stream=False,
                kv_transfer_params={'do_remote_decode':True,'do_remote_prefill':False})
            prefill.pop('stream_options',None)
            p_headers={'X-Request-Id':rid+'.P','X-data-parallel-rank':str(p['wire_local_dp'])}
            row['status']='PREFILL';row['prefill_dispatch_monotonic_ns']=time.monotonic_ns()
            async with self.session.post(f"http://172.16.12.{p['node']}:{p['port']}/v1/completions",json=prefill,headers=p_headers) as response:
                row['prefill_http_status']=response.status
                raw=await response.text()
                if response.status!=200:
                    row['status']='PREFILL_REJECTED'
                    row['upstream_error_body']=raw
                    return web.Response(status=response.status,text=raw,content_type='application/json')
                result=json.loads(raw)
            row['prefill_terminal_monotonic_ns']=time.monotonic_ns();row['prefill_usage']=result.get('usage')
            assert result['usage']['prompt_tokens']==len(ids) and result['usage']['completion_tokens']==1
            params=result['kv_transfer_params']
            assert params['remote_num_prompt_tokens']==len(ids) and params['remote_block_ids']
            assert params['do_remote_prefill'] is True and params['do_remote_decode'] is False
            assert params['remote_host']==f"172.16.12.{p['node']}" and params['remote_port']==26032+p['global_dp']
            row['prefill_remote_request_id']=params['remote_request_id']
            row['prefill_remote_engine_id']=params['remote_engine_id']
            row['prefill_exported_blocks']=[len(blocks) for blocks in params['remote_block_ids']]
            decode=dict(body);decode.update(request_id=rid+'.D',kv_transfer_params=params)
            assert decode['prompt'] is ids and len(decode['prompt'])==row['input_tokens']
            d_headers={'X-Request-Id':rid+'.D','X-data-parallel-rank':str(d['wire_local_dp'])}
            row['status']='DECODE';row['decode_dispatch_monotonic_ns']=time.monotonic_ns()
            async with self.session.post(f"http://172.16.12.{d['node']}:{d['port']}/v1/completions",json=decode,headers=d_headers) as response:
                row['decode_http_status']=response.status;row['decode_headers_monotonic_ns']=time.monotonic_ns()
                if response.status!=200:
                    row['status']='DECODE_REJECTED'
                    raw=await response.text();row['upstream_error_body']=raw
                    # No speculative source ACK here: D may have admitted a transfer.
                    return web.Response(status=response.status,text=raw,content_type='application/json')
                stream=web.StreamResponse(status=200,headers={'Content-Type':'text/event-stream','Cache-Control':'no-cache'})
                await stream.prepare(request)
                async for chunk in response.content.iter_any():
                    if 'first_forward_monotonic_ns' not in row:row['first_forward_monotonic_ns']=time.monotonic_ns()
                    await stream.write(chunk)
                await stream.write_eof()
            row['status']='D_STREAM_COMPLETE'
            return stream
        except asyncio.CancelledError:
            row['status']='CANCELLED';row['error']='Gateway request cancelled; native D owns transfer drain'
            raise
        except Exception as error:
            row['status']='FAILED';row['error']=repr(error)
            if stream is None:return web.json_response({'error':{'message':str(error),'type':'GLM53GatewayError'}},status=502)
            raise
        finally:
            row['gateway_terminal_monotonic_ns']=time.monotonic_ns();self.rows.append(row)
            print('GLM53_GATEWAY_EVENT='+json.dumps(row,sort_keys=True),flush=True)
