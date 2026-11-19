#!/usr/bin/env python3
"""Read local G4 formal snapshots, publish only allowlisted histogram deltas.

The source snapshot hashes were first recorded during this offline extraction;
they are not claimed as an earlier historical frozen snapshot index. Source
HTTP responses and local path names are never written to output.
"""
import argparse,hashlib,json
from pathlib import Path
from summarize_stage_snapshots_056 import between,duration,V_BASES
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/vllm_formal_059/summary_v2.json'
RUNNER_SHA={32768:'ac8e7408b3f8024a0c178896a4579f9592b25a764c323405a55042f60410a077',65536:'6bacd313eced19553d8c329cffe3fb55f14eab76672ac1d4140f6ab05ef310bb'}
RUN_MANIFEST_SHA={32768:'9f9a1cb0e1a339730215b11cb7dbf915cd1b7f44dceaea0b0ce631363da27490',65536:'5fe0e154134274b3f78bbeeec3ec870f4b2ad076afdd67b63776862f8ebfb03b'}
RESUME_NAME={32768:'resume_manifest_20261117T160850496552Z.json',65536:'resume_manifest_20261117T163949792088Z.json'}
SNAPSHOTS={
 (32768,'cold','before'):'e14f1fb0a9ae2f30a00818b7ae698900ec42502168ce79de49399a5bd37a058c',
 (32768,'cold','after'):'35d8fa682b6f35af0d87009fa18f96011de11ccf482c1187d1f956cad7a7dcfb',
 (32768,'hot','before'):'38134019ff8ce7bf2286f7750bdb990fbc82f543ef177973c3f67508f52af5b2',
 (32768,'hot','after'):'3611c8f950fa00b29e837fe2949151cf4d142add38f02fd446dda0c3cbce9a29',
 (65536,'cold','before'):'cae1b2298263bb6767e5b22dedf33b357a41d186f8e96d8172470518ef517917',
 (65536,'cold','after'):'bb52757a1f0a29ef60185f966ca3923a59a8b7869dcd6e8598039c99384ab938',
 (65536,'hot','before'):'5bf975bdecaf88c3321804d35c27ecdbe59c66dbdff9b27a7cf0f540eecc020f',
 (65536,'hot','after'):'87a7bb2898213033b2c06ee0433a0648e86f2424b9ab97a2d75d25fd11503329',
}
AUDIT_SHA={(32768,'cold'):'f46b93e0c6654e4fb85a5066e0a24288a88896d0a4d179de5eb016686d82e76a',(32768,'hot'):'bc1a78bf340cd568e2b64a1acba3014932802060a84ebbc1c61d371112a90bef',(65536,'cold'):'194d62450e5de00581437f4e66b5c716471e8d4a9c9f571dbaef9005d23cff29',(65536,'hot'):'0d93935c5ce8a74fdf9c7b0cd47fe5e4ed551fcaece20fc5aeb852248c254442'}


def need(v,m):
 if not v:raise ValueError(m)


def read_regular(p,max_size=10000000):
 need(p.is_file() and not p.is_symlink(),'source is not a regular file');before=p.stat();need(before.st_size<=max_size,'source too large');b=p.read_bytes();after=p.stat();need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'source changed during read');return b


def sha(b):return hashlib.sha256(b).hexdigest()


def contains(obj,target):
 if isinstance(obj,dict):return any(contains(v,target) for v in obj.values())
 if isinstance(obj,list):return any(contains(v,target) for v in obj)
 return obj==target


def summarize(args):
 hashes={}
 for line in (ROOT/'SHA256SUMS').read_text().splitlines():
  h,p=line.split('  ',1);need(p not in hashes,'duplicate checksum');hashes[p]=h
 def archive(name):
  p=Path(name);need(not p.is_absolute() and '..' not in p.parts and name in hashes,'unsafe archive path');raw=read_regular(ROOT/p);need(sha(raw)==hashes[name],'archive source mismatch');return json.loads(raw)
 metrics=archive('records/metrics.json');provenance=archive('provenance/artifacts.json');roots={32768:Path(args.root32),65536:Path(args.root64)};runners={32768:Path(args.runner32),65536:Path(args.runner64)};docs={};source={};controller={}
 for n in (32768,65536):
  runner_raw=read_regular(runners[n]);run_raw=read_regular(roots[n]/'run_manifest.json');resume_raw=read_regular(roots[n]/RESUME_NAME[n]);resume=json.loads(resume_raw)
  need(sha(runner_raw)==RUNNER_SHA[n] and sha(run_raw)==RUN_MANIFEST_SHA[n],'controller source mismatch');need(resume.get('runner_sha256')==RUNNER_SHA[n] and resume.get('original_manifest_sha256')==RUN_MANIFEST_SHA[n] and resume.get('same_actual_service_identities_verified') is True,'resume receipt mismatch')
  controller[n]={'runner_sha256':sha(runner_raw),'original_run_manifest_sha256':sha(run_raw),'resume_manifest_sha256_current':sha(resume_raw),'resume_recorded_same_service_identity':True,'scope':'resume is a historical receipt; no independent proof of entire-period worker/series continuity'}
 for key,expected in SNAPSHOTS.items():
  n,phase,bound=key;raw=read_regular(roots[n]/f'{bound}_{phase}_{n}_server_snapshot.json');need(sha(raw)==expected,'snapshot SHA mismatch');docs[key]=json.loads(raw);source[key]=sha(raw)
 results=[]
 for n in (32768,65536):
  for phase in ('cold','hot'):
   tag=('g47' if n==32768 else 'g48')+f'_{phase}_{n}';matching=[x for x in metrics if x.get('phase')==tag and x.get('scored')];need(len(matching)==1,'formal index mismatch');record=matching[0];need((record['input_tokens'],record['output_tokens'],record['requests'],record['successful_requests'],record['configured_concurrency'])==(n,1024,32,32,16),'formal workload mismatch');manifest=archive(record['manifest'])
   audit_raw=read_regular(roots[n]/f'{phase}_{n}_audit.json');audit_hash=sha(audit_raw);need(audit_hash==AUDIT_SHA[(n,phase)],'audit original SHA mismatch');p=[x for x in provenance if x.get('original_sha256')==audit_hash];need(len(p)==1 and contains(manifest,p[0]['artifact']),'audit provenance mismatch')
   a=docs[(n,phase,'before')];b=docs[(n,phase,'after')]
   for role in ('prefill','decode'):
    x=a[role]['metrics'];y=b[role]['metrics'];need(x['status']==y['status']==200,'metrics response unavailable');rows,proc,_=between(x['body'],y['body'],V_BASES,role,False);need(proc=='same_at_endpoints' and len(rows)==5 and all(z['created_check']=='same_at_endpoints' and z['delta_count']==32 for z in rows),'formal histogram endpoint mismatch')
    results.append({'input_tokens':n,'phase':phase,'role':role,'nominal_scored_requests':32,'snapshot_elapsed_s':duration(x['at'],y['at']),'source_snapshot_sha256':[source[(n,phase,'before')],source[(n,phase,'after')]],'source_snapshot_sha_basis':'first recorded during this offline extraction; not historical frozen source digest','source_audit_original_sha256':audit_hash,'audit_original_matches_archived_provenance_and_formal_manifest':True,'process_start_check':proc,'client_mean_ttft_s':record['ttft_mean_ms']/1000,'client_mean_e2e_s':record['e2el_mean_ms']/1000,'histograms':rows})
 return {'schema':'glm53-vllm-g4-formal-stage-v2','controllers':controller,'results':results,'limitations':['Local historical snapshots not previously frozen by source SHA; locally pinned and double-checked during this extraction','Snapshots enclose formal run but counters are exporter endpoint aggregates; count equality does not prove same 32 requests, attempts, or batch identities','Same exporter process-start and histogram-created at endpoints and nonnegative bucket deltas do not rule out unseen reset or independent worker restart','Local wall-clock scrape intervals are not request latency, exact benchmark boundaries, or cross-machine calibration','P/D and different histograms can overlap; no sum or request-level chain implied','SG and vLLM use different metrics definitions and runtime versions; same unit/name does not establish equivalent stages or causal framework comparison']}


def checks(doc):
 need(len(doc['results'])==8,'role/phase coverage');keys={(x['input_tokens'],x['phase'],x['role']) for x in doc['results']};need(keys=={(n,phase,role) for n in (32768,65536) for phase in ('cold','hot') for role in ('prefill','decode')},'duplicate or missing role scope')
 for x in doc['results']:
  need(x['snapshot_elapsed_s']>0 and len(x['histograms'])==5,'empty scope');need(all(y['delta_count']==32 for y in x['histograms']),'sample count')
 p=next(x for x in doc['results'] if x['input_tokens']==32768 and x['phase']=='cold' and x['role']=='prefill');q=next(x for x in p['histograms'] if x['metric']=='vllm:request_queue_time_seconds');need(abs(q['mean_s']-28.238173)<1e-5,'32K queue mean mismatch')
 p=next(x for x in doc['results'] if x['input_tokens']==65536 and x['phase']=='cold' and x['role']=='prefill');q=next(x for x in p['histograms'] if x['metric']=='vllm:request_queue_time_seconds');need(abs(q['mean_s']-80.198494)<1e-5,'64K queue mean mismatch')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root32',required=True);ap.add_argument('--root64',required=True);ap.add_argument('--runner32',required=True);ap.add_argument('--runner64',required=True);ap.add_argument('--write',action='store_true');ap.add_argument('--check',action='store_true');args=ap.parse_args();doc=summarize(args);checks(doc);text=json.dumps(doc,ensure_ascii=False,indent=2)+'\n'
 if args.write:
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with OUT.open('x',encoding='utf8') as f:f.write(text)
 if args.check:need(OUT.read_text(encoding='utf8')==text,'output mismatch')
 print(json.dumps({'status':'pass','scopes':len(doc['results']),'sha256':sha(text.encode()),'wrote':args.write,'checked':args.check}))


if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError):raise SystemExit('FAILED_CLOSED: formal source or histogram validation failed')
