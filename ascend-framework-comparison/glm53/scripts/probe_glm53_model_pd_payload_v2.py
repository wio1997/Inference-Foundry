"""Bounded greedy model/output probe; not a throughput or PD qualification alone."""
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import urllib.error

from tokenizers import Tokenizer

NAMESPACE = 'glm53-functional-pd-20261017-v2'
TOKENIZER = Path('/data/wio/models/GLM-5.3-w8a8/tokenizer.json')
assert hashlib.sha256(TOKENIZER.read_bytes()).hexdigest() == '19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'
tokenizer = Tokenizer.from_file(str(TOKENIZER))
text = ('Inventory verification. The code maple means 17, cedar means 23, and river means 41. '
        'Read the complete record and explain how to check that the numbers add up. '
        'A reliable answer uses each stated number exactly once and shows the result.\n') * 40
ids = tokenizer.encode(text, add_special_tokens=False).ids[:512]
assert len(ids) == 512
output = {'project_id':'glm53','namespace':NAMESPACE,'input_tokens':len(ids),
          'ids_sha256_le_u32':hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest(),
          'requests':[], 'model_requests_attempted':0,
          'qualification':'CLIENT_OUTPUT_ONLY_REQUIRES_SERVER_SDMA_STATE_BYTES_PROOF'}
for node in (30,31,43,45):
    port=30300 if node in (30,31) else 30400
    with urllib.request.urlopen(f'http://172.16.12.{node}:{port}/health',timeout=3) as response:
        assert response.status==200


def request(label,node,rank,maximum,salt,params=None):
    port=30300 if node in (30,31) else 30400
    request_id=NAMESPACE+'-'+label
    body={'model':'glm53-w8a8','prompt':ids,'max_tokens':maximum,'min_tokens':maximum,
          'temperature':0,'seed':20261017,'ignore_eos':True,'stream':False,
          'return_token_ids':True,'cache_salt':NAMESPACE+'-'+salt,'request_id':request_id}
    if params is not None:body['kv_transfer_params']=params
    row={'label':label,'node':node,'dp_rank':rank,'request_id':request_id,'body':body,
         'dispatch_monotonic_ns':time.monotonic_ns()}
    output['requests'].append(row);output['model_requests_attempted']+=1
    req=urllib.request.Request(f'http://172.16.12.{node}:{port}/v1/completions',
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json',
        'X-Request-Id':request_id,'X-data-parallel-rank':str(rank)})
    try:
        with urllib.request.urlopen(req,timeout=180) as response:
            row['status']=response.status;row['response']=json.load(response)
    except urllib.error.HTTPError as e:
        row['status']=e.code;row['error_body']=e.read().decode(errors='replace');raise
    finally:
        row['terminal_monotonic_ns']=time.monotonic_ns()
        print('GLM53_CLIENT_STEP='+json.dumps(row),flush=True)
    result=row['response'];assert result['usage']['prompt_tokens']==512
    assert result['usage']['completion_tokens']==maximum
    choice=result['choices'][0];assert choice['finish_reason']=='length'
    assert isinstance(choice.get('token_ids'),list) and len(choice['token_ids'])==maximum
    return result


try:
    direct_p=request('direct-P',30,0,16,'direct-P')
    direct_d=request('direct-D',43,0,16,'direct-D')
    prefill=request('PD-P',30,0,1,'PD',{'do_remote_decode':True,'do_remote_prefill':False})
    params=prefill.get('kv_transfer_params')
    assert params and params['remote_block_ids'] and params['remote_num_prompt_tokens']==512
    assert params['do_remote_prefill'] and not params['do_remote_decode']
    decoded=request('PD-D',43,0,16,'PD',params)
    p_tokens=direct_p['choices'][0]['token_ids'];d_tokens=direct_d['choices'][0]['token_ids']
    pd_tokens=decoded['choices'][0]['token_ids']
    output['direct_P_equals_direct_D']=p_tokens==d_tokens
    output['PD_equals_direct_D']=pd_tokens==d_tokens
    output['PD_equals_direct_P']=pd_tokens==p_tokens
    output['client_output_pass']=p_tokens==d_tokens==pd_tokens
    assert output['client_output_pass'],'Greedy output mismatch; retain all outputs and do not benchmark'
finally:
    print('GLM53_CLIENT_RESULT='+json.dumps(output),flush=True)
