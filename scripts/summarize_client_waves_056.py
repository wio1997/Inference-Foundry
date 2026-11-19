#!/usr/bin/env python3
"""Summarize local AISBench client timings in arrival-order groups.

Existing source request data is loaded in memory only; output contains no
request text, per-request timings, IDs, absolute timestamps, or private paths.
No attribution to server stages or cross-run ordinal pairing is implied.
"""
import argparse,ast,hashlib,json,math,sqlite3,statistics,struct
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/client_waves_056/summary_v2.json'
PHASES={
 (32768,'cold','sglang'):'g228_cold_32768', (32768,'cold','vllm'):'g47_cold_32768',
 (32768,'hot','sglang'):'g228_hot_32768', (32768,'hot','vllm'):'g47_hot_32768',
 (65536,'cold','sglang'):'g273_cold_65536', (65536,'cold','vllm'):'g48_cold_65536',
 (65536,'hot','sglang'):'g273_hot_65536', (65536,'hot','vllm'):'g48_hot_65536',
}


def need(x,m):
 if not x:raise ValueError(m)


def verified(path,sums):
 p=Path(path);need(not p.is_absolute() and '..' not in p.parts,'unsafe source path');key=p.as_posix();need(key in sums,'source absent from manifest');b=(ROOT/p).read_bytes();need(hashlib.sha256(b).hexdigest()==sums[key],'source SHA mismatch');return b


def read_timing_arrays(path):
 with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as conn:
  tables={x[0] for x in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")};need(tables=={'numpy_store','sqlite_sequence'},'unexpected timing schema');values=[]
  for (blob,) in conn.execute('SELECT arr_blob FROM numpy_store'):
   need(blob[:6]==b'\x93NUMPY' and blob[6] in (1,2,3),'invalid timing array');major=blob[6];offset=10 if major==1 else 12;headerlen=struct.unpack('<H' if major==1 else '<I',blob[8:offset])[0];header=ast.literal_eval(blob[offset:offset+headerlen].decode('utf-8' if major==3 else 'latin1'));need(len(header['shape'])==1 and header['descr'] in ('<f8','<f4'),'invalid timing header');n=header['shape'][0];need(type(n) is int and n>=2,'invalid timing length');points=struct.unpack('<'+str(n)+('d' if header['descr']=='<f8' else 'f'),blob[offset+headerlen:]);need(all(math.isfinite(x) for x in points) and all(a<=b for a,b in zip(points,points[1:])),'invalid timing ordering');values.append((points[0],points[1],points[-1]))
 return values


def q(values,quantile):
 x=sorted(values);need(x and 0<=quantile<=1,'invalid quantile');i=(len(x)-1)*quantile;lo=math.floor(i);hi=math.ceil(i);return x[lo]+(x[hi]-x[lo])*(i-lo)


def stats(xs):return {'min':min(xs),'p50':q(xs,.5),'p90':q(xs,.9),'max':max(xs)}


def summarize():
 sums={}
 for line in (ROOT/'SHA256SUMS').read_text().splitlines():
  digest,path=line.split('  ',1);need(path not in sums,'duplicate SHA entry');sums[path]=digest
 metrics=json.loads(verified('records/metrics.json',sums));results=[]
 for (size,phase,fw),tag in PHASES.items():
  selected=[x for x in metrics if x['phase']==tag and x['scored']];need(len(selected)==1,'phase ambiguity');record=selected[0];need((record['input_tokens'],record['output_tokens'],record['requests'],record['successful_requests'],record['configured_concurrency'])==(size,1024,32,32,16),'workload mismatch')
  manifest=json.loads(verified(record['manifest'],sums));evidence=manifest['raw_evidence'];dbs=evidence['timing_databases'];need(len(dbs)==1,'unexpected timing DB count');verified(dbs[0],sums)
  durations=read_timing_arrays(ROOT/dbs[0]);need(len(durations)==32,'client timing count')
  for start,first,end in durations:need(start<=first<=end,'invalid client timing')
  durations.sort();origin=durations[0][0];groups=[];boundary_ties=sum(durations[i-1][0]==durations[i][0] for i in (8,16,24))
  for i in range(4):
   sample=durations[i*8:(i+1)*8];need(len(sample)==8,'group denominator')
   groups.append({'arrival_order_group':i+1,'count':len(sample),'start_offset_s':stats([s-origin for s,f,e in sample]),'first_frame_offset_s':stats([f-origin for s,f,e in sample]),'last_frame_offset_s':stats([e-origin for s,f,e in sample]),'ttft_s':stats([f-s for s,f,e in sample]),'post_first_s':stats([e-f for s,f,e in sample])})
  sweep=[]
  for start,first,end in durations:sweep.extend(((start,1),(end,-1)))
  active=0;peak=0
  for _,delta in sorted(sweep,key=lambda x:(x[0],x[1])):active+=delta;need(active>=0,'invalid client concurrency');peak=max(peak,active)
  need(active==0 and peak<=16,'client concurrency bound');ttft=[f-s for s,f,e in durations];e2e=[e-s for s,f,e in durations]
  need(abs(statistics.mean(ttft)*1000-record['ttft_mean_ms'])<0.11 and abs(statistics.mean(e2e)*1000-record['e2el_mean_ms'])<0.11,'archived mean mismatch')
  results.append({'input_tokens':size,'phase':phase,'framework':fw,'requests':32,'configured_concurrency':16,'peak_timed_spans_overlapping':peak,'arrival_group_boundary_ties':boundary_ties,'groups':groups,'overall_ttft_s':stats(ttft),'overall_e2e_s':stats(e2e)})
 return {'schema':'glm53-client-arrival-quartiles-v1','group_size':8,'results':results,'limits':['Client timing only: before POST, first recorded nonempty stream frame, last recorded nonempty stream frame; not HTTP close or server stages','Peak overlapping measured spans is not necessarily full HTTP concurrency','Each run has an independent local time origin; start-order groups are not same request/step across frameworks','Groups are post-hoc fixed-size quantile bins, not proven scheduler waves','Only archived run metrics and timing-array databases are read; no request or response contents are opened','Timing database has 32 arrays and recomputed means match aggregate; this check alone does not establish request ID/semantic pairing','No request text, IDs, individual event times or absolute timestamps in this output','Single cold/hot run per configuration; concurrency and data, versions, epochs and queue behavior limit causal attribution']}


def checks(d):
 need(len(d['results'])==8,'run coverage')
 for x in d['results']:
  need(len(x['groups'])==4 and all(y['count']==8 for y in x['groups']),'group coverage');need(x['groups'][0]['start_offset_s']['min']==0,'time origin')
  need(all(g['start_offset_s']['min']<=g['start_offset_s']['p50']<=g['start_offset_s']['max'] for g in x['groups']),'start monotonic');need(x['peak_timed_spans_overlapping']<=16,'concurrency')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');ap.add_argument('--check',action='store_true');a=ap.parse_args();doc=summarize();checks(doc);text=json.dumps(doc,ensure_ascii=False,indent=2)+'\n'
 if a.write:
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with OUT.open('x',encoding='utf8') as f:f.write(text)
 if a.check:need(OUT.read_text(encoding='utf8')==text,'summary mismatch')
 print(json.dumps({'status':'pass','runs':len(doc['results']),'groups_per_run':4,'group_size':8,'sha256':hashlib.sha256(text.encode()).hexdigest(),'wrote':a.write,'checked':a.check}))


if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,AssertionError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
