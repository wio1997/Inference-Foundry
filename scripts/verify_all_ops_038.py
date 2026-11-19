#!/usr/bin/env python3
"""Verify sanitized 32-rank device-kernel summaries, not performance claims."""
import gzip,hashlib,json,math,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES={'p':('p_all_kernels.json.gz','3b5a99bf6882e2c8c13f1c9e5540528ccff429758e2509117ad5d9b2430b6622'),
         'd':('d_all_kernels_sanitized.json.gz','596ce5ef9a04d8c9f6fce073dcfe15d4a40e5ee96c92380cacdfe8ea192a65e8')}
IP=re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')
PATH=re.compile(r'(?<!\w)/(?:home|root|data|data1|task|tmp|models)(?:/|\b)')
DTYPE={'BOOL','DOUBLE','DT_BF16','DT_UNDEFINED','FLOAT','FLOAT16','INT32','INT64','INT8','UINT64','unknown'}
SHAPE=re.compile(r'^[0-9,;\[\]()\-\s]+$')


def require(condition,message):
    if not condition:raise ValueError(message)


def scan_strings(value):
    if isinstance(value,dict):
        for k,v in value.items():scan_strings(k);scan_strings(v)
    elif isinstance(value,list):
        for v in value:scan_strings(v)
    elif isinstance(value,str):
        require(not IP.search(value) and not PATH.search(value),'private address or path')
        require(not re.search(r'(?i)(password|bearer\s|authorization:|secret)',value),'credential-like text')


def validate_shape(v):
    return v in ('unknown','<unknown>','<not-recorded>','') or isinstance(v,str) and SHAPE.fullmatch(v) is not None


def validate_dtype(v):
    return isinstance(v,str) and all(x.strip() in DTYPE for x in v.split(';'))


def verify():
    out={}
    for role,(name,digest) in SOURCES.items():
        raw=gzip.decompress((ROOT/'evidence/all_ops_038'/name).read_bytes());require(hashlib.sha256(raw).hexdigest()==digest,'expanded SHA mismatch '+role)
        d=json.loads(raw);scan_strings(d);rows=d['results'] if role=='p' else d['ranks'];require(len(rows)==16,'rank count')
        keys=set();total_rows=0;groups=0
        for entry in rows:
            r=entry['result'] if role=='p' else entry;fw=r['framework'] if role=='p' else r['engine'];rank=r['rank'];key=(fw,rank);require(key not in keys and rank in range(8),'duplicate rank');keys.add(key)
            kernels=r['events'] if role=='p' else r['exact_kernels'];n=r['rows'] if role=='p' else r['csv_row_count'];subtotal=0
            for k in kernels:
                name=k['event_name_exact'] if role=='p' else k['kernel_name'];require(isinstance(name,str) and bool(name),'empty kernel name')
                stats=k if role=='p' else k['duration_us'];count=k['count'] if role=='p' else stats['count'];require(type(count) is int and count>0,'kernel count')
                for f in ('min_us','p50_us','p90_us','p99_us','max_us'):
                    v=stats.get(f);require(v is None or isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0,'invalid duration')
                if role=='p':
                    require(k['duration_invalid_count']==0 and k['duration_valid_count']==count,'P duration count')
                    require(all(validate_shape(s) for s in k['shape_counts']),'P shape')
                else:
                    for f in ('input_shape_counts','output_shape_counts'):require(all(validate_shape(s) for s in k[f]),'D shape')
                    for f in ('input_dtype_counts','output_dtype_counts'):require(all(validate_dtype(s) for s in k[f]),'D dtype')
                    require(sum(x['count'] for x in k['shape_dtype_stream_counts'])==count,'D joint signature mass')
                    for f in ('input_shape_counts','output_shape_counts','input_dtype_counts','output_dtype_counts','stream_id_counts'):require(sum(k[f].values())==count,'D marginal mass')
                subtotal+=count
            require(subtotal==n,'rank kernel event count mismatch');total_rows+=n;groups+=len(kernels)
        expected={('sglang' if role=='p' else 'sg',i) for i in range(8)}|{('vllm',i) for i in range(8)}
        require(keys==expected,'rank keys')
        require((total_rows,groups)==((755447,31890) if role=='p' else (556800,15040)),'total event count')
        out[role]={'ranks':len(rows),'csv_rows':total_rows,'rank_kernel_groups':groups,'expanded_sha256':digest}
    return out


if __name__=='__main__':
    try:print(json.dumps({'status':'pass','scope':'single 32K diagnostic; event sums are not wall-clock','roles':verify()},ensure_ascii=False,indent=2))
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
