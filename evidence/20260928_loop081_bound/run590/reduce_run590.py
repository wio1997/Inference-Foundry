#!/usr/bin/env python3
"""Recompute Run590 static inventory admission; never query devices."""
from pathlib import Path
import re,json,hashlib
P=Path(__file__).resolve().parent
m=json.loads((P/'manifest.json').read_text())
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
need((P/'acquire.exit').read_text()=='0\n' and (P/'acquire.stderr').read_bytes()==b'', 'collector exit/stderr')
acquired=json.loads((P/'acquire.stdout').read_text())
need(acquired['status']==m['status'] and acquired['inventory_exits']==[0]*8 and acquired['input_hashes_unchanged'], 'collector stdout gate')
need(m['status']=='read_only_all8_inventory_acquired' and m['current_cli_hccs_help_gate'],'acquisition gate')
rows=[]
for c in m['commands']:
 for name in ('stdout','stderr'):need(sha(P/c[name])==c[name+'_sha256'],'raw command output drift')
 if not c['stdout'].startswith('hccs_rank'):continue
 rank=int(c['stdout'].removeprefix('hccs_rank').split('.')[0]);need(c['argv']==['/usr/local/sbin/npu-smi','info','-t','hccs','-i',str(rank),'-c','0'],'inventory argv')
 need(c['exit_code']==0 and c['host_end_monotonic_ns']>c['host_start_monotonic_ns'],'exit/clock bracket')
 fields={}
 for line in (P/c['stdout']).read_text().splitlines():
  key,sep,value=line.partition(':')
  if sep:
   key=key.strip();need(key not in fields,'duplicate field');fields[key]=value.strip()
 need(fields['hccs health status']=='OK' and fields['hccs first error lane']=='NA','health')
 def array(name):
  s=fields[name];need(s.startswith('[') and s.endswith(']'),'array syntax')
  a=[int(x) for x in s[1:-1].split()];need(len(a)==7 and all(0<=x<2**64 for x in a),'seven valid entries');return a
 a={name:array('hccs '+name) for name in ('lane mode','link lane list','link speed','tx packets','tx bytes','rx packets','rx bytes','retry count','error count')}
 need(a['lane mode']==[4]*7 and a['link lane list']==[1111]*7 and a['link speed']==[224]*7,'observed link inventory changed')
 need(a['retry count']==a['error count']==[0]*7,'observed errors')
 for direction in ('tx','rx'):need(a[direction+' bytes']==[20*x for x in a[direction+' packets']],'bytes/count relation')
 rows.append({'rank':rank,'health':'OK','reported_links':7,'reported_lane_mode':a['lane mode'],'reported_link_lane_list':a['link lane list'],'reported_standard_speed_gbps':a['link speed'],'twenty_bytes_per_reported_packet':True,'retry_count':a['retry count'],'error_count':a['error count'],'host_bracket_ns':[c['host_start_monotonic_ns'],c['host_end_monotonic_ns']],'elapsed_s':c['elapsed_s']})
need(sorted(x['rank'] for x in rows)==list(range(8)),'all8 uniqueness')
need(m['input_hashes_unchanged'] and all(x.get('unchanged_after',True) for x in m['files'].values()),'acquisition identity drift')
need(m['files']['/usr/local/bin/npu-smi']['sha256']==m['files']['/usr/local/sbin/npu-smi']['sha256'],'CLI alias bytes differ')
result={'status':'scoped_read_only_static_inventory_admitted','ranks':rows,'total_reported_directed_local_port_records':56,'scope':'one static query per rank; no simultaneous-snapshot, freshness or interval-traffic claim','strict_capacity_upper':None,'cumulative_service_B':None,'physical_traffic_interval_bytes':None,'compulsory_communication_bytes':None,'resource_hardware_endpoint_s':None,'scheduling_execution_endpoint_s':None,'product_e2e_endpoint':None,'formal_current_tps':571.681,'input_manifest_sha256':sha(P/'manifest.json'),'acquirer_sha256':sha(P/'acquire_run590.py'),'reducer_sha256':sha(Path(__file__))}
(P/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'ranks':len(rows),'local_port_records':56,'finite_endpoints':0}))
