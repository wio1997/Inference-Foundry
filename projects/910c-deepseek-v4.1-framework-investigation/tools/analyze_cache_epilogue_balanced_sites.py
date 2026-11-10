#!/usr/bin/env python3
"""Bind typed native cache publication chains in two saved balanced windows."""
from collections import Counter, defaultdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json, re, statistics, zlib

ROOT = Path('audit/execution-search')

def shapes(r):
    return [tuple(map(int, re.findall(r'-?\d+', p))) for p in r['data']['Input Shapes'].strip('"').split(';')]

def typ(r):
    return r['data']['Type']

def start(r):
    return Decimal(r['data']['Start Time(us)'].strip())

def duration(r):
    return Decimal(r['data']['Duration(us)'])

def union(intervals):
    end = None
    out = Decimal(0)
    for a,b in sorted(intervals):
        if end is None or a >= end:
            out += b-a
            end = b
        elif b > end:
            out += b-end
            end = b
    return out

def describe(seq):
    return dict(stream=seq[0]['data']['Stream ID'], csv_rows=[r['csv_row'] for r in seq],
                task_ids=[r['data']['Task ID'] for r in seq], types=[typ(r) for r in seq],
                shapes=[shapes(r) for r in seq], input_dtypes=[r['data']['Input Data Types'] for r in seq],
                native_sum_us=float(sum(map(duration,seq))), span_us=float(start(seq[-1])+duration(seq[-1])-start(seq[0])),
                task_ids_consecutive=all(int(b['data']['Task ID'])==int(a['data']['Task ID'])+1 for a,b in zip(seq,seq[1:])))

def main():
    archive=ROOT/'cache-epilogue-balanced-calls.json.zlib'
    data=json.loads(zlib.decompress(archive.read_bytes()))
    supp=json.loads((ROOT/'engram-storage-balanced-supplement.json').read_text())
    results=[]
    for p in data:
        streams=defaultdict(list)
        for r in p['selected']: streams[r['data']['Stream ID']].append(r)
        found=defaultdict(list)
        for stream,rows in streams.items():
            rows.sort(key=start)
            for i,r in enumerate(rows):
                seq=rows[i:i+3]
                if len(seq)==3 and list(map(typ,seq))==['RmsNorm','InplacePartialRotaryMul','ScatterNdUpdateSk']:
                    s=list(map(shapes,seq));n=s[0][0][0]
                    if s[0]==[(n,512),(512,)] and s[1]==[(n,1,1,512),(n,1,1,64),(n,1,1,64)] and s[2][1:]==[(n,2),(n,512)]:
                        found['norm_rope_cache512'].append(seq)
                seq=rows[i:i+10]
                grammar=['RmsNorm','MatMulV2','RmsNorm','InplacePartialRotaryMul','DynamicQuant','ScatterNdUpdateSk','Cast','ScatterNdUpdateSk','InplacePartialRotaryMul','ScatterNdUpdateSk']
                if len(seq)==10 and list(map(typ,seq))==grammar:
                    s=list(map(shapes,seq));n=s[0][0][0]
                    if (s[0]==[(n,512),(512,)] and s[1][:2]==[(n,512),(128,512)] and s[2]==[(n,128),(128,)]
                        and s[3]==[(n,1,1,128),(n,1,1,64),(n,1,1,64)] and s[4]==[(n,128)]
                        and s[5][1:]==[(n,2),(n,128)] and s[7][1:]==[(n,2),(n,1)]
                        and s[8]==[(n,1,1,512),(n,1,1,64),(n,1,1,64)] and s[9][1:]==[(n,2),(n,512)]):
                        found['index_long_joint10'].append(seq)
        window=next(w['device_window_ms']*1000 for w in supp['windows'] if w['phase']==p['phase'] and w['ep']==p['ep'])
        families=[]
        for name,seqs in found.items():
            intervals=[(start(r),start(r)+duration(r)) for seq in seqs for r in seq]
            spans=[(start(seq[0]),start(seq[-1])+duration(seq[-1])) for seq in seqs]
            detail=[describe(seq) for seq in seqs]
            perop=Counter()
            aiv=Counter()
            aiv_missing=Counter()
            for seq in seqs:
                for r in seq:
                    perop[typ(r)]+=float(duration(r))
                    val=r['data'].get('aiv_time(us)')
                    if val and val!='N/A':aiv[typ(r)]+=float(val)
                    else:aiv_missing[typ(r)]+=1
            sensitivity=[]
            for fraction in [0.25,0.5,1.0]:
                for h in [0,5,20]:
                    saving=float(union(intervals))*fraction-len(seqs)*h
                    sensitivity.append(dict(native_fraction_saved=fraction,added_us_per_chain=h,local_net_us=saving,window_pct=100*saving/window))
            families.append(dict(name=name,chains=len(seqs),chains_by_row_count=dict(Counter(shapes(s[0])[0][0] for s in seqs)),
                chains_by_stream=dict(Counter(s[0]['data']['Stream ID'] for s in seqs)),
                all_task_ids_consecutive=all(d['task_ids_consecutive'] for d in detail),
                native_sum_us=sum(d['native_sum_us'] for d in detail),native_union_us=float(union(intervals)),
                native_union_window_pct=float(union(intervals))*100/window,span_union_us=float(union(spans)),
                span_union_window_pct=float(union(spans))*100/window,
                per_operator_sum_us=dict(perop),per_operator_aiv_sum_us=dict(aiv),aiv_missing_calls=dict(aiv_missing),
                conditional_removed_intermediate_payload_bytes=(sum(shapes(s[0])[0][0] for s in seqs)*2304 if name=='norm_rope_cache512' else None),
                sensitivity=sensitivity,chains_detail=detail))
        results.append(dict(ep=p['ep'],dp=p['dp'],phase=p['phase'],window_us=window,csv_sha256=p['sha256'],families=families))
    out=dict(status='TYPED_SAVED_WINDOW_BINDING_NOT_REQUEST_ESTIMATE',profiles=results,
        interpretation=[
            'Two representative ranks only. Typed source-consistent grammars do not prove layer IDs, tensor aliases, dependency exposure or exact loaded binary/key.',
            'Rows are adjacent among selected native types on one stream; excluded CSV task types may intervene. Task-ID adjacency is reported independently.',
            'norm_rope_cache512 and index_long_joint10 are disjoint grammar instances; neither includes complete query/GEMM/input-ready parent.',
            'All-free native/span figures are conditional upper envelopes. Sensitivity pays one added cost per chain; no request credit or measured speedup.',
            'New mirrored balanced cohort stays separate from old eight-distinct-prompt full-request denominators; no cross-cohort extrapolation.',
            'Every replacement primitive must be bitwise identical; no primitive or performance admission from this analysis.'],
        inputs=[dict(path=str(archive),sha256=sha256(archive.read_bytes()).hexdigest()),dict(path=str(ROOT/'engram-storage-balanced-supplement.json'),sha256=sha256((ROOT/'engram-storage-balanced-supplement.json').read_bytes()).hexdigest())],
        new_profile=0,new_NPU_work=0,whole_chain_complete=False)
    out['analyzer_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    dest=ROOT/'cache-epilogue-balanced-sites.json';dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:p[k] for k in ['ep','window_us']}|{'families':[{k:f[k] for k in ['name','chains','chains_by_row_count','all_task_ids_consecutive','native_union_window_pct','span_union_window_pct','per_operator_sum_us']} for f in p['families']]} for p in results],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
