#!/usr/bin/env python3
"""Validate the live unmodified official proxy against frozen native outputs."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();assert args.execute
    base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32');gold=json.loads((base/'recovery-v2/original64/pairs.json').read_text());refs={r['input_tokens']:r for r in gold['rows'] if r['mode']=='control'};out=args.output;assert out.is_relative_to(base);out.mkdir(exist_ok=False);report={'completed':False,'rows':[],'scope':'Current stock official proxy end-to-end native PD smoke, cold salts, no seed opt-in; no new performance claim.'}
    try:
        for n in (256,257):
            ref=refs[n]
            for stream in (False,True):
                body={k:v for k,v in ref['D_request'].items() if k!='kv_transfer_params'};body.update(max_tokens=64,stream=stream,cache_salt='proxy-native-'+uuid.uuid4().hex)
                if not stream:body.pop('stream_options',None)
                req=urllib.request.Request('http://172.16.12.32:29880/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json'});row={'input_tokens':n,'stream':stream,'request':body,'sent_ns':time.monotonic_ns()};report['rows'].append(row)
                with urllib.request.urlopen(req,timeout=120) as response:
                    row['http_status']=response.status
                    if stream:
                        chunks=[];ids=[]
                        for line in response:
                            if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):continue
                            chunk=json.loads(line[6:]);assert not chunk.get('error'),chunk;chunks.append(chunk)
                            for c in chunk.get('choices',[]):ids.extend(c.get('token_ids') or [])
                        row['chunks']=chunks;row['token_ids']=ids
                    else:
                        row['response']=json.load(response);row['token_ids']=row['response']['choices'][0]['token_ids']
                row['received_ns']=time.monotonic_ns();row['exact_match']=row['token_ids']==ref['token_ids']
                (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
                assert row['http_status']==200 and row['exact_match'] and len(row['token_ids'])==64
        report['completed']=True
    except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END');assert report['completed'],report.get('failure')



if __name__=='__main__':main()
