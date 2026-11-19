#!/usr/bin/env python3
"""Reconstruct endpoint means, not request chains, from formal snapshots.

Input P/D local vLLM snapshots contain HTTP metrics bodies: they are read in
memory, filtered and never printed. No prompt/output/individual identity is
read. Local snapshot SHAs were first pinned in 059, not historically frozen.
"""
import argparse,hashlib,json
from pathlib import Path
from summarize_stage_snapshots_056 import between
from summarize_vllm_formal_059 import SNAPSHOTS,read_regular
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/p_latency_scope_060/summary_v2.json'


def need(x,message):
 if not x:raise ValueError(message)


def sha(x):return hashlib.sha256(x).hexdigest()


def hist(scope,name):
 xs=[h for h in scope['histograms'] if h['metric']==name];need(len(xs)==1,'ambiguous metric');return xs[0]


def summarize(args):
 hashes={}
 for line in (ROOT/'SHA256SUMS').read_text().splitlines():
  h,name=line.split('  ',1);need(name not in hashes,'duplicate archive checksum');hashes[name]=h
 def source(name):
  p=Path(name);need(not p.is_absolute() and '..' not in p.parts and name in hashes,'unsafe source');raw=read_regular(ROOT/p);need(sha(raw)==hashes[name],'archive source mismatch');return json.loads(raw),sha(raw)
 sg,sgsha=source('evidence/stage_snapshots_056/summary_v7.json');v,vsha=source('evidence/vllm_formal_059/summary_v2.json');metrics,msha=source('records/metrics.json');roots={32768:Path(args.root32),65536:Path(args.root64)};results=[]
 for n in (32768,65536):
  for phase in ('cold','hot'):
   docs={}
   for bound in ('before','after'):
    raw=read_regular(roots[n]/f'{bound}_{phase}_{n}_server_snapshot.json');need(sha(raw)==SNAPSHOTS[(n,phase,bound)],'vLLM source SHA mismatch');docs[bound]=json.loads(raw)
   sgp=next(x for x in sg['scopes'] if x.get('framework')=='sglang' and x.get('capture')=='formal' and x.get('input_tokens')==n and x.get('phase')==phase and x.get('role')=='prefill');sge=hist(sgp,'sglang:e2e_request_latency_seconds');need(sge['metric_scope']=='global_streaming' and sge['delta_count']==32,'SG P scope mismatch')
   tag=('g228' if n==32768 else 'g273')+f'_{phase}_{n}';x=[z for z in metrics if z.get('phase')==tag and z.get('scored')];need(len(x)==1 and x[0]['requests']==x[0]['successful_requests']==32,'SG client scope mismatch');sgclient=x[0]['ttft_mean_ms']/1000
   vse={}
   for role in ('prefill','decode'):
    x=docs['before'][role]['metrics'];y=docs['after'][role]['metrics'];need(x['status']==y['status']==200,'metrics status');rows,process,_=between(x['body'],y['body'],('vllm:e2e_request_latency_seconds',),role,False);need(len(rows)==1 and rows[0]['delta_count']==32 and rows[0]['created_check']=='same_at_endpoints' and process=='same_at_endpoints','vLLM E2E scope mismatch');vs=next(x for x in v['results'] if x['input_tokens']==n and x['phase']==phase and x['role']==role);ttft=hist(vs,'vllm:time_to_first_token_seconds');eb={z['upper_s']:z['cumulative_count'] for z in rows[0]['buckets']};tb={z['upper_s']:z['cumulative_count'] for z in ttft['buckets']};shared=set(eb)&set(tb);vse[role]={'e2e':rows[0],'ttft_histogram_count':ttft['delta_count'],'ttft_mean_s':ttft['mean_s'],'e2e_and_ttft_same_count_and_sum':rows[0]['delta_count']==ttft['delta_count'] and rows[0]['delta_sum_s']==ttft['delta_sum_s'],'common_bucket_boundary_count':len(shared),'common_bucket_cumulative_deltas_equal':all(eb[k]==tb[k] for k in shared)}
   vp=next(x for x in v['results'] if x['input_tokens']==n and x['phase']==phase and x['role']=='prefill');results.append({'input_tokens':n,'phase':phase,'sg_client_ttft_mean_s':sgclient,'vllm_client_ttft_mean_s':vp['client_mean_ttft_s'],'sg_prefill_e2e_streaming':sge,'vllm_prefill':vse['prefill'],'vllm_decode':vse['decode']})
 return {'schema':'glm53-p-service-and-client-scopes-v1','source_sha256':{'sg_formal':sgsha,'vllm_formal':vsha,'client_records':msha},'results':results,'limitations':['Different metrics and services; each listed value is its own mean, no per-request service-client identity is established','P e2e and client TTFT do not share identical start/end points; their numerical proximity is exploratory, not route overhead or critical-path contribution','vLLM histogram count and sum equality does not establish same events, sample identity, endpoint or semantics','Raw vLLM snapshots first hashed offline in 059; SG exporter process epoch is not proven within windows','Independent service epochs, software versions, limits and batching confound causal comparison']}


def checks(d):
 need(len(d['results'])==4 and {(r['input_tokens'],r['phase']) for r in d['results']}=={(n,p) for n in (32768,65536) for p in ('cold','hot')},'coverage')
 for r in d['results']:
  need(r['sg_prefill_e2e_streaming']['delta_count']==32 and r['vllm_prefill']['e2e']['delta_count']==32,'invalid sample count');need(r['vllm_prefill']['e2e_and_ttft_same_count_and_sum'] and r['vllm_prefill']['common_bucket_boundary_count']==8 and r['vllm_prefill']['common_bucket_cumulative_deltas_equal'],'vLLM P endpoint mismatch')
 a=next(r for r in d['results'] if r['input_tokens']==32768 and r['phase']=='cold');need(abs(a['sg_prefill_e2e_streaming']['mean_s']-65.150760)<1e-5 and abs(a['vllm_prefill']['e2e']['mean_s']-40.595361)<1e-5,'unexpected mean')


def main():
 p=argparse.ArgumentParser();p.add_argument('--root32',required=True);p.add_argument('--root64',required=True);p.add_argument('--write',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args();d=summarize(a);checks(d);text=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
 if a.write:
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with OUT.open('x',encoding='utf8') as f:f.write(text)
 if a.check:need(OUT.read_text(encoding='utf8')==text,'output mismatch')
 print(json.dumps({'status':'pass','scopes':len(d['results']),'sha256':sha(text.encode()),'wrote':a.write,'checked':a.check}))


if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,StopIteration,json.JSONDecodeError):raise SystemExit('FAILED_CLOSED: scope, source, or parsing check failed')
