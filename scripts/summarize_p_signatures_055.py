#!/usr/bin/env python3
"""Offline only: compare recorded P signature quantiles across ranks.

8 ranks belong to one capture, not independent repetitions. No requests,
stream IDs or private source locations are copied into the output.
"""
import argparse,gzip,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'evidence/p_formats_047/distributions.json.gz'
SRC_SHA='9ae081169cd00b9cebaa9bf2b922232c1def10d71db44ef5562da79946e15b2f'
OUT=ROOT/'evidence/p_signatures_055/summary.json'
KEYS=('kernel_name','input_shape','output_shape','input_dtype','output_dtype','input_format','output_format')


def need(v,m):
 if not v:raise ValueError(m)


def summarize():
 raw=gzip.decompress(SRC.read_bytes());need(hashlib.sha256(raw).hexdigest()==SRC_SHA,'source SHA mismatch');data=json.loads(raw);pairs=data['format_pairs'];need(len(pairs)==112 and data['rank_count']==8,'source coverage')
 groups={}
 for row in pairs:
  key=tuple(row[k] for k in KEYS);rank=row['rank'];need(type(rank) is int and 0<=rank<8 and all(isinstance(x,str) and x for x in key),'invalid signature');group=groups.setdefault(key,{})
  need(rank not in group,'duplicate rank/signature');entry={'rank':rank}
  for f in ('sg','vllm'):
   src=row[f];n=src['count'];p50=src['p50_us'];p90=src['p90_us'];need(type(n) is int and n>0 and 0<=p50<=p90,'invalid distribution');entry[f]={'count':n,'p50_us':p50,'p90_us':p90}
  group[rank]=entry
 need(len(groups)==14 and all(set(v)==set(range(8)) for v in groups.values()),'incomplete signature ranks');out=[]
 for key,group in sorted(groups.items()):
  ranks=[group[i] for i in range(8)];result={k:v for k,v in zip(KEYS,key)};result['ranks']=ranks
  for stat in ('p50_us','p90_us'):
   diff=[x['sg'][stat]-x['vllm'][stat] for x in ranks];result[stat+'_sg_minus_vllm']={'min':min(diff),'median':statistics.median(diff),'max':max(diff),'sg_larger_rank_count':sum(x>0 for x in diff),'equal_rank_count':sum(x==0 for x in diff),'sg_smaller_rank_count':sum(x<0 for x in diff)}
  out.append(result)
 return {'schema':'glm53-p-recorded-signature-rank-quantile-v1','source_decompressed_sha256':SRC_SHA,'ranks':8,'signature_count':len(out),'signatures':out,
         'limitations':['One independent diagnostic capture per engine; ranks are not independent repetitions','Matching recorded name/input-output shape/dtype/format does not prove matching strides, data, useful work, layer, logical step, request, runtime implementation or device frequency','Ranks and counts differ and quantiles are independently calculated; no event-level pairing, critical-path attribution, or TTFT contribution','Different service epochs, CANN/torch_npu versions, and concurrency limits']}


def checks(doc):
 need(doc['ranks']==8 and doc['signature_count']==14,'summary coverage');s=doc['signatures'];dq=[x for x in s if x['kernel_name']=='aclnnDynamicQuantV2_DynamicQuant_DynamicQuant'];need(len(dq)==4,'DQ group count')
 for x in dq:
  expected=8 if x['input_shape']=='8192,4096' else 0;need(x['p50_us_sg_minus_vllm']['sg_larger_rank_count']==expected,'DQ p50 direction')
 rms=[x for x in s if x['kernel_name']=='RmsNorm' and x['input_shape']=='8192,4096;4096'];need(len(rms)==1 and rms[0]['p50_us_sg_minus_vllm']['sg_smaller_rank_count']==8 and rms[0]['p90_us_sg_minus_vllm']['sg_larger_rank_count']==8,'RmsNorm percentile direction')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');ap.add_argument('--check',action='store_true');args=ap.parse_args();doc=summarize();checks(doc);text=json.dumps(doc,ensure_ascii=False,indent=2)+'\n'
 if args.write:
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with OUT.open('x',encoding='utf8') as f:f.write(text)
 if args.check:need(OUT.read_text(encoding='utf8')==text,'output mismatch')
 print(json.dumps({'status':'pass','signature_count':len(doc['signatures']),'rank_count':8,'sha256':hashlib.sha256(text.encode()).hexdigest(),'wrote':args.write,'checked':args.check},ensure_ascii=False))


if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
