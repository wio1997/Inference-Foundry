#!/usr/bin/env python3
"""Run590 one supported all8 read-only HCCS inventory; never reset counters."""
from pathlib import Path
import json,subprocess,hashlib,time,datetime,os
OUT=Path(__file__).resolve().parent
if (OUT/'manifest.json').exists():raise SystemExit('Refusing to overwrite or rerun Run590')
CLI=Path('/usr/local/sbin/npu-smi')
DRIVER=Path('/usr/local/Ascend/driver')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def cmd(tag,argv,timeout=30):
 start=time.monotonic_ns();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
  rc=p.returncode;stdout=p.stdout;stderr=p.stderr
 except subprocess.TimeoutExpired as e:
  rc=124;stdout=e.stdout or b'';stderr=e.stderr or b''
 end=time.monotonic_ns()
 (OUT/(tag+'.stdout')).write_bytes(stdout);(OUT/(tag+'.stderr')).write_bytes(stderr)
 row={'argv':argv,'exit_code':rc,'host_start_monotonic_ns':start,'host_end_monotonic_ns':end,'host_start_utc':utc,'elapsed_s':(end-start)/1e9,'stdout':tag+'.stdout','stderr':tag+'.stderr','stdout_sha256':hashlib.sha256(stdout).hexdigest(),'stderr_sha256':hashlib.sha256(stderr).hexdigest()}
 (OUT/(tag+'.json')).write_text(json.dumps(row,indent=2)+'\n')
 return row,stdout.decode(errors='replace')
manifest={'schema':1,'scope':'offline identity and once-per-rank read-only HCCS static inventory; no reset/write/workload; counters are one snapshot, never deltas','commands':[],'files':{},'strict_capacity_upper':None,'physical_traffic_interval_bytes':None}
files=[CLI,Path('/usr/local/bin/npu-smi'),DRIVER/'tools/npu-smi',DRIVER/'version.info',DRIVER/'lib64/driver/libdcmi.so',DRIVER/'lib64/driver/libdrvdsmi_host.so',DRIVER/'device/ascend_910b_device_sw.img',DRIVER/'device/ascend_910b_device_sw.bin',DRIVER/'device/ascend_910b_device_boot.img',DRIVER/'device/ascend_910b_device_config.bin',DRIVER/'include/dcmi_interface_api.h',DRIVER/'include/dsmi_common_interface.h',DRIVER/'kernel/dms/hccs/dms_hccs_feature.c',DRIVER/'kernel/dms/hccs/dms_hccs.mk',DRIVER/'kernel/dms/hccs/dms_hccs_init.c']
for path in files:
 if not path.exists():manifest['files'][str(path)]={'exists':False};continue
 manifest['files'][str(path)]={'exists':True,'resolved_path':str(path.resolve()),'size':path.stat().st_size,'sha256':sha(path),'symlink':path.is_symlink()}
 if path.suffix in ('.img','.bin'):
  with path.open('rb') as f:header=f.read(256)
  manifest['files'][str(path)]['header256_hex']=header.hex()
version=(DRIVER/'version.info').read_text()
(OUT/'driver_version.txt').write_text(version)
if 'Version=26.0.rc1' not in version:raise SystemExit('Exact installed driver changed; no inventory')
row,help_text=cmd('cli_info_help',[str(CLI),'info','-h']);manifest['commands'].append(row)
if row['exit_code'] or 'hccs' not in [x.strip() for x in help_text.replace('\n',',').replace(' ', ',').split(',')]:raise SystemExit('Current CLI does not explicitly support hccs')
manifest['current_cli_hccs_help_gate']=True
for tag,path in [('cli',CLI),('dcmi',DRIVER/'lib64/driver/libdcmi.so'),('dsmi',DRIVER/'lib64/driver/libdrvdsmi_host.so')]:
 for mode,arg in [('notes','-n'),('dynamic','-d')]:
  row,_=cmd(tag+'_'+mode,['readelf',arg,str(path)]);manifest['commands'].append(row)
 row,_=cmd(tag+'_dynsymbols',['readelf','--dyn-syms','--wide',str(path)]);manifest['commands'].append(row)
row,_=cmd('cli_ldd',['ldd',str(CLI)]);manifest['commands'].append(row)
for rank in range(8):
 row,_=cmd('hccs_rank'+str(rank),[str(CLI),'info','-t','hccs','-i',str(rank),'-c','0'])
 manifest['commands'].append(row)
 # Unsupported/error output is preserved; do not retry another API.
for p,r in manifest['files'].items():
 if r.get('exists'):r['unchanged_after']=Path(p).exists() and sha(Path(p))==r['sha256']
manifest['inventory_exits']=[r['exit_code'] for r in manifest['commands'] if r['stdout'].startswith('hccs_rank')]
manifest['input_hashes_unchanged']=all(r.get('unchanged_after',True) for r in manifest['files'].values())
manifest['status']='read_only_all8_inventory_acquired' if manifest['inventory_exits']==[0]*8 and manifest['input_hashes_unchanged'] else 'partial_read_only_inventory_no_retry'
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':manifest['status'],'inventory_exits':manifest['inventory_exits'],'input_hashes_unchanged':manifest['input_hashes_unchanged']}))
