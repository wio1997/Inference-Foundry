#!/usr/bin/env python3
"""Offline D signature quantile comparison; recorded format/stride unknown."""
import argparse,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'evidence/d_shape_040/matched_distributions.json'
SRC_SHA='3e910b859e4b97176b65553450df69e8721fb375f2f0bfa00acda5894c048347'
OUT=ROOT/'evidence/p_signatures_055/d_summary.json'
KEYS=('kernel_name','input_shape','output_shape','input_dtype','output_dtype')


def need(v,m):
 if not v:raise ValueError(m)


def summarize():
 raw=SRC.read_bytes();need(hashlib.sha256(raw).hexdigest()==SRC_SHA,'source SHA mismatch');data=json.loads(raw);need(len(data['ranks'])==16 and data['rank_count']==16 and data['all_signature_counts_match'],'source coverage')
 groups={}
 for source in data['ranks']:
  fw=source['engine'];rank=source['rank'];need(fw in ('sg','vllm') and type(rank) is int and 0<=rank<8,'engine/rank identity')
  for row in source['signatures']:
   key=tuple(row[k] for k in KEYS);need(all(isinstance(x,str) and x for x in key),'signature invalid');pair=groups.setdefault(key,{})
   need((fw,rank) not in pair,'duplicate signature/rank');s=row['duration_us'];n=s['count'];need(row['count_match'] and row['actual_count']==n and n>0 and 0<=s['p50_us']<=s['p90_us'],'count or percentile mismatch');pair[(fw,rank)]={'count':n,'p50_us':s['p50_us'],'p90_us':s['p90_us']}
 need(len(groups)==15 and all(set(g)=={(fw,i) for fw in ('sg','vllm') for i in range(8)} for g in groups.values()),'signature coverage')
 out=[]
 for key,g in sorted(groups.items()):
  ranks=[{'rank':i,'sg':g[('sg',i)],'vllm':g[('vllm',i)]} for i in range(8)];item={k:v for k,v in zip(KEYS,key)};item['ranks']=ranks
  for stat in ('p50_us','p90_us'):
   diff=[r['sg'][stat]-r['vllm'][stat] for r in ranks];item[stat+'_sg_minus_vllm']={'min':min(diff),'median':statistics.median(diff),'max':max(diff),'sg_larger_rank_count':sum(x>0 for x in diff),'equal_rank_count':sum(x==0 for x in diff),'sg_smaller_rank_count':sum(x<0 for x in diff)}
  out.append(item)
 return {'schema':'glm53-d-recorded-signature-rank-quantile-v1','source_sha256':SRC_SHA,'ranks':8,'signature_count':len(out),'signatures':out,
         'limitations':['One non-scored diagnostic capture per engine; ranks are not independent repetitions','Input/output format and strides unknown; same recorded shape/dtype/name does not prove same data, expert load, logical steps or requests','Counts and runtime versions differ; quantiles not paired and do not establish per-token, critical-path or end-to-end differences']}


def checks(doc):
 need(doc['ranks']==8 and doc['signature_count']==15,'summary coverage');s=doc['signatures'];top=[x for x in s if x['kernel_name']=='MoeGatingTopK'];need(len(top)==1 and top[0]['p50_us_sg_minus_vllm']['sg_larger_rank_count']==8 and top[0]['p90_us_sg_minus_vllm']['sg_larger_rank_count']==4,'TopK direction')
 dq=[x for x in s if 'DynamicQuantV2' in x['kernel_name']];need(len(dq)==2 and all(x['p50_us_sg_minus_vllm']['sg_smaller_rank_count']==8 for x in dq),'DQ direction')


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
