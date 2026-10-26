"""Same CPU43 client clocks only; role HTTP legs are not pure GPU compute."""
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def pct(values,p):
 values=sorted(values);at=(len(values)-1)*p;low=math.floor(at);return values[low]+(values[math.ceil(at)]-values[low])*(at-low)
def main(run_id):
 path=ROOT/'runs'/run_id/'formal-result.remote.json';report=json.loads(path.read_text());assert report['phase']=='COMPLETE';rows=[]
 for item in report['formal']['rows']:
  gateway=item['result']['terminal']['gateway'];valid=item['result']['native']['extra_perf_data']['glm53_valid_output_chunk_times'];start=item['outer_dispatch_perf_counter_ns']/1e9;first=valid[0]
  moments=[start,gateway['prefill_dispatch_monotonic_ns']/1e9,gateway['prefill_terminal_monotonic_ns']/1e9,gateway['decode_dispatch_monotonic_ns']/1e9,first]
  assert all(a<=b for a,b in zip(moments,moments[1:]));parts=dict(zip(['client_to_P_dispatch','P_HTTP_leg','P_to_D_dispatch','D_dispatch_to_first_valid_chunk'],[b-a for a,b in zip(moments,moments[1:])]))
  assert math.isclose(sum(parts.values()),item['result']['ttft_seconds'],rel_tol=1e-9,abs_tol=1e-7)
  rows.append({'request_id':item['request_id'],'parts_seconds':parts,'P_HTTP_fraction_of_TTFT':parts['P_HTTP_leg']/(first-start)})
 stats={key:{name:pct([r['parts_seconds'][key] for r in rows],p) for name,p in [('P50',.5),('P90',.9),('P99',.99)]} for key in rows[0]['parts_seconds']}
 result={'project_id':'glm53','framework':report['framework'],'concurrency':report['concurrency'],'run_id':run_id,'formal_requests':len(rows),'scope':'All boundaries are CPU43 client/gateway clocks. P_HTTP_leg includes native P scheduling, KV lookup/read/preparation and prefill/output, not pure GPU compute. D leg includes admission/metadata/SDMA/first inference plus receive. Per-leg quantiles must not be summed. No P/D host clocks subtracted.','parts_quantiles_seconds':stats,'P_HTTP_TTFT_fraction':{'P50':pct([x['P_HTTP_fraction_of_TTFT'] for x in rows],.5),'P90':pct([x['P_HTTP_fraction_of_TTFT'] for x in rows],.9)},'formal_report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'optimization_causal_gain_claimed':False}
 target=ROOT/f'results/{report["framework"]}-formal-C{report["concurrency"]}-role-leg-timing-v1.json';target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('run_id');args=parser.parse_args();main(args.run_id)
