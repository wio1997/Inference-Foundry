"""Scoped HTTP/load helpers; contains no deployment or benchmark entry point."""

import json
import time
import urllib.request

from bench_profile import LEADERS

def get(url, method='GET', timeout=15):
    req=urllib.request.Request(url,method=method)
    with urllib.request.urlopen(req,timeout=timeout) as r:
        body=r.read().decode()
        try: return json.loads(body)
        except json.JSONDecodeError: return body

def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); tmp.replace(path)

def load_counts(obj):
    result=[]
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k in ('num_reqs','num_running_reqs','num_waiting_reqs','num_prefill_reqs','num_decode_reqs','prefill_inflight_queue_reqs','prefill_bootstrap_queue_reqs','decode_prealloc_queue_reqs','decode_transfer_queue_reqs','decode_retracted_queue_reqs') and isinstance(v,(int,float)):
                result.append((k,v))
            result += load_counts(v)
    elif isinstance(obj,list):
        for v in obj: result += load_counts(v)
    return result

def monitor(stop, path):
    with path.open('a') as f:
        while not stop.is_set():
            rec={'epoch':time.time(),'monotonic':time.perf_counter()}
            for role,url,*_ in LEADERS:
                try: rec[role]=get(url+'/v1/loads',timeout=3)
                except Exception as e: rec[role]={'error':str(e)}
            f.write(json.dumps(rec)+'\n'); f.flush(); stop.wait(5)
