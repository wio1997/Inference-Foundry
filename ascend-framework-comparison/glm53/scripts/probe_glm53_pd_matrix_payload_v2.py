"""Exercise all16 D-DP ranks and all4 P-DP ranks with short real requests."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct
import threading
import time
import urllib.request
import urllib.error

from tokenizers import Tokenizer


# Native completion control ONLY for v1 D calls rejected with HTTP400 before inference.
# The controller froze the exact P request IDs from those four raw responses.
import zmq
import msgspec
REJECTED_TRANSFERS = [{'D_DP': 8, 'P_DP': 0, 'error': '{"error":{"message":"data_parallel_rank 8 is out of range [0, 8).","type":"BadRequestError","param":null,"code":400}}', 'params': {'do_remote_prefill': True, 'do_remote_decode': False, 'remote_block_ids': [[13, 19, 20, 21]], 'remote_num_prompt_tokens': 512, 'remote_engine_id': '8bbe7ef5-dd05-45b8-8661-47fb6ec9c7ba-7a0471823dc74988a9a42c49ea72867b_dp0', 'remote_request_id': 'cmpl-glm53-pd-matrix-20261017-v1-d08-P-0-852e0b53', 'remote_host': '172.16.12.30', 'remote_port': 26032, 'last_token_id': 220}}, {'D_DP': 9, 'P_DP': 1, 'error': '{"error":{"message":"data_parallel_rank 9 is out of range [0, 8).","type":"BadRequestError","param":null,"code":400}}', 'params': {'do_remote_prefill': True, 'do_remote_decode': False, 'remote_block_ids': [[5, 11, 12, 13]], 'remote_num_prompt_tokens': 512, 'remote_engine_id': '8bbe7ef5-dd05-45b8-8661-47fb6ec9c7ba-7a0471823dc74988a9a42c49ea72867b_dp1', 'remote_request_id': 'cmpl-glm53-pd-matrix-20261017-v1-d09-P-0-a6f449d5', 'remote_host': '172.16.12.30', 'remote_port': 26033, 'last_token_id': 220}}, {'D_DP': 12, 'P_DP': 0, 'error': '{"error":{"message":"data_parallel_rank 12 is out of range [0, 8).","type":"BadRequestError","param":null,"code":400}}', 'params': {'do_remote_prefill': True, 'do_remote_decode': False, 'remote_block_ids': [[18, 23, 24, 25]], 'remote_num_prompt_tokens': 512, 'remote_engine_id': '8bbe7ef5-dd05-45b8-8661-47fb6ec9c7ba-7a0471823dc74988a9a42c49ea72867b_dp0', 'remote_request_id': 'cmpl-glm53-pd-matrix-20261017-v1-d12-P-0-b3b9f9aa', 'remote_host': '172.16.12.30', 'remote_port': 26032, 'last_token_id': 220}}, {'D_DP': 13, 'P_DP': 1, 'error': '{"error":{"message":"data_parallel_rank 13 is out of range [0, 8).","type":"BadRequestError","param":null,"code":400}}', 'params': {'do_remote_prefill': True, 'do_remote_decode': False, 'remote_block_ids': [[10, 15, 16, 17]], 'remote_num_prompt_tokens': 512, 'remote_engine_id': '8bbe7ef5-dd05-45b8-8661-47fb6ec9c7ba-7a0471823dc74988a9a42c49ea72867b_dp1', 'remote_request_id': 'cmpl-glm53-pd-matrix-20261017-v1-d13-P-0-96b136e5', 'remote_host': '172.16.12.30', 'remote_port': 26033, 'last_token_id': 220}}]
context=zmq.Context()
try:
    for item in REJECTED_TRANSFERS:
        params=item['params'];assert params['remote_host']=='172.16.12.30'
        assert params['remote_port']==26032+item['P_DP'] and item['P_DP'] in (0,1)
        assert params['remote_request_id'].startswith('cmpl-glm53-pd-matrix-20261017-v1-')
        sock=context.socket(zmq.REQ);sock.setsockopt(zmq.SNDTIMEO,3000);sock.setsockopt(zmq.RCVTIMEO,3000)
        try:
            endpoint=f"tcp://{params['remote_host']}:{params['remote_port']}";sock.connect(endpoint)
            sock.send(msgspec.msgpack.encode((b'done_recving_msg',params['remote_request_id'])))
            ack=sock.recv();assert ack==b'ACK'
            print('GLM53_REJECTED_D_CONTROL_ACK='+json.dumps({'endpoint':endpoint,'request_id':params['remote_request_id'],'ack':'ACK','physical_bytes_freed':None}),flush=True)
        finally:sock.close(linger=0)
finally:context.term()

NAMESPACE='glm53-pd-matrix-20261017-v2'
EXPECTED=[220,99619,11,323,14785,3363,220,102340,13,4457,279,4583,3255,323,10334,1246]
path=Path('/data/wio/models/GLM-5.3-w8a8/tokenizer.json')
assert hashlib.sha256(path.read_bytes()).hexdigest()=='19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'
text=('Inventory verification. The code maple means 17, cedar means 23, and river means 41. '
      'Read the complete record and explain how to check that the numbers add up. '
      'A reliable answer uses each stated number exactly once and shows the result.\n')*40
ids=Tokenizer.from_file(str(path)).encode(text,add_special_tokens=False).ids[:512]
assert len(ids)==512
sha=hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()
assert sha=='d6e4cf16d46a7e29461cd4a7e19fa45725cb1b42ac3f76fda226260074db107e'
lock=threading.Lock();attempted=0


def leg(drank,prefill,params=None):
    global attempted
    prank=drank%4;node=(30 if prank<2 else 31) if prefill else (43 if drank<8 else 45)
    rank=prank if prefill else drank;wire_rank=rank%(2 if prefill else 8);maximum=1 if prefill else 16
    request_id=f'{NAMESPACE}-d{drank:02}-'+('P' if prefill else 'D')
    body={'model':'glm53-w8a8','prompt':ids,'max_tokens':maximum,'min_tokens':maximum,
          'temperature':0,'seed':20261017,'ignore_eos':True,'return_token_ids':True,
          'stream':False,'request_id':request_id,'cache_salt':f'{NAMESPACE}-d{drank:02}',
          'kv_transfer_params':{'do_remote_decode':True,'do_remote_prefill':False} if prefill else params}
    row={'request_id':request_id,'D_DP':drank,'P_DP':prank,'node':node,'rank':rank,
         'wire_local_rank':wire_rank,'input_sha256_le_u32':sha,'dispatch_monotonic_ns':time.monotonic_ns()}
    with lock:attempted+=1
    request=urllib.request.Request(f'http://172.16.12.{node}:'+('30300' if prefill else '30400')+'/v1/completions',
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json',
        'X-Request-Id':request_id,'X-data-parallel-rank':str(wire_rank)})
    try:
        with urllib.request.urlopen(request,timeout=90) as response:
            row['status']=response.status;row['response']=json.load(response)
        result=row['response'];choice=result['choices'][0]
        assert result['usage']['prompt_tokens']==512 and result['usage']['completion_tokens']==maximum
        assert choice['finish_reason']=='length' and choice['token_ids']==EXPECTED[:maximum]
        if prefill:
            params=result['kv_transfer_params'];assert params['remote_block_ids'] and params['remote_num_prompt_tokens']==512
            assert params['remote_host']==f'172.16.12.{node}' and params['remote_port']==26032+prank
        row['pass']=True
    except Exception as error:
        row['pass']=False;row['error']=repr(error)
        if isinstance(error,urllib.error.HTTPError):
            row['status']=error.code;row['error_body']=error.read().decode(errors='replace')
    finally:
        row['terminal_monotonic_ns']=time.monotonic_ns()
        with lock:print('GLM53_MATRIX_STEP='+json.dumps(row),flush=True)
    return row


def case(drank):
    prefill=leg(drank,True)
    decode=leg(drank,False,prefill['response']['kv_transfer_params']) if prefill['pass'] else None
    return {'D_DP':drank,'P_DP':drank%4,'prefill':prefill,'decode':decode,
            'pass':prefill['pass'] and decode is not None and decode['pass']}


results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures={pool.submit(case,rank):rank for rank in range(16)}
    for future in concurrent.futures.as_completed(futures):results.append(future.result())
result={'project_id':'glm53','namespace':NAMESPACE,'input_tokens':512,'input_sha256_le_u32':sha,
        'model_requests_attempted':attempted,'cases':sorted(results,key=lambda row:row['D_DP']),
        'client_output_pass':len(results)==16 and all(row['pass'] for row in results),
        'qualification':'MATRIX_OUTPUT_ONLY_REQUIRES_PER_DP_SERVER_SDMA_STATE_AUDIT'}
print('GLM53_MATRIX_RESULT='+json.dumps(result),flush=True)
assert result['client_output_pass'] and attempted==32
