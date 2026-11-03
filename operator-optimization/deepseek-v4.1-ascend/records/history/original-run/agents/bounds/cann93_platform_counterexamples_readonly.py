#!/usr/bin/env python3
"""Capture nonsecret CANN platform INI fields from own stage container; no NPU access."""
import base64, hashlib, json, pathlib, shlex, subprocess
remote=r'''import configparser, glob, hashlib, json, pathlib
result=[]
for s in sorted(glob.glob('/usr/local/Ascend/cann-9.2.0/aarch64-linux/data/platform_config/Ascend910_93*.ini')):
 p=pathlib.Path(s); c=configparser.ConfigParser(interpolation=None); c.read(s)
 fields={'SoC_version':c.get('version','SoC_version'),'Short_SoC_version':c.get('version','Short_SoC_version'),'NpuArch':c.get('version','NpuArch'),'ai_core_cnt':c.getint('SoCInfo','ai_core_cnt'),'cube_core_cnt':c.getint('SoCInfo','cube_core_cnt'),'vector_core_cnt':c.getint('SoCInfo','vector_core_cnt'),'memory_size':c.getint('SoCInfo','memory_size'),'cube_freq':c.getint('AICoreSpec','cube_freq'),'DT_INT8':c.get('DtypeMKN','DT_INT8')}
 result.append({'path':s,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'fields':fields})
p=pathlib.Path('/usr/local/Ascend/driver/include/dcmi_interface_api.h')
print(json.dumps({'files':result,'driver_header_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))'''
cmd='docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+base64.b64encode(remote.encode()).decode()+"'))")
p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=30,check=True)
x=json.loads(p.stdout)
assert len(x['files'])>=7
x['scope']='current stage image CANN configuration and mounted driver header, not proof of hardware limits or historical kernel binding'
out=pathlib.Path(__file__).with_name('cann93-platform-counterexamples.json');out.write_text(json.dumps(x,indent=2)+'\n')
print(out); print('files',len(x['files']),'driver header sha256',x['driver_header_sha256'])
