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

NAMESPACE='glm53-pd-matrix-20261017-v1'
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
    rank=prank if prefill else drank;maximum=1 if prefill else 16
    request_id=f'{NAMESPACE}-d{drank:02}-'+('P' if prefill else 'D')
    body={'model':'glm53-w8a8','prompt':ids,'max_tokens':maximum,'min_tokens':maximum,
          'temperature':0,'seed':20261017,'ignore_eos':True,'return_token_ids':True,
          'stream':False,'request_id':request_id,'cache_salt':f'{NAMESPACE}-d{drank:02}',
          'kv_transfer_params':{'do_remote_decode':True,'do_remote_prefill':False} if prefill else params}
    row={'request_id':request_id,'D_DP':drank,'P_DP':prank,'node':node,'rank':rank,
         'input_sha256_le_u32':sha,'dispatch_monotonic_ns':time.monotonic_ns()}
    with lock:attempted+=1
    request=urllib.request.Request(f'http://172.16.12.{node}:'+('30300' if prefill else '30400')+'/v1/completions',
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json',
        'X-Request-Id':request_id,'X-data-parallel-rank':str(rank)})
    try:
        with urllib.request.urlopen(request,timeout=90) as response:
            row['status']=response.status;row['response']=json.load(response)
        result=row['response'];choice=result['choices'][0]
        assert result['usage']['prompt_tokens']==512 and result['usage']['completion_tokens']==maximum
        assert choice['finish_reason']=='length' and choice['token_ids']==EXPECTED[:maximum]
        if prefill:
            params=result['kv_transfer_params'];assert params['remote_block_ids'] and params['remote_num_prompt_tokens']==512
        row['pass']=True
    except Exception as error:
        row['pass']=False;row['error']=repr(error)
        if isinstance(error,urllib.error.HTTPError):row['error_body']=error.read().decode(errors='replace')
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
