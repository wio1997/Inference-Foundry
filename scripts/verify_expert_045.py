#!/usr/bin/env python3
"""Offline check of existing G181/G240 expert-recording evidence scope."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'evidence/matched_SG_G181'
FILES={
 'readiness':('d/readiness.json','85af82116c37a846c493e729656f449ebf8f73c141e7fdd54c8dafa9e2d34829'),
 'g182':('p/functional_PD_g182/startup_d_server_info.json','f1e28417587f009d57b4a3815a06ac90f955dae0b65fc1aac802e4c4d99e6a4d'),
 'g204':('p/functional_PD_g204/after_d_server_info.json','7f58961dd190a0538b025b14ef308bf96ac1db2a08a2bfd07994ba298a43ac5e'),
 'stability':('profile_G240/CPU_d/review_g252/raw_stability.json','d7c03f54a484883b349c24480a5622fe956092bd76805c6d5a3fc2334841c4ed'),
 'start':('profile_G240/control/d_start_profile.json','dad4ee0d0a78b7e2c180f4cd65915e2098f6968cc111b9d3b0c92655d9009dd4'),
 'job':('profile_G240/control/job.py','a2a18f2e5ede26a4567c4174f7db66c8306adc3824ff8725fe9d4eb880ea56e9'),
}


def need(ok,message):
 if not ok:raise ValueError(message)


def load(name):
 rel,digest=FILES[name];raw=(BASE/rel).read_bytes();need(hashlib.sha256(raw).hexdigest()==digest,'source SHA: '+name);return raw


def check():
 out={}
 for name in ('readiness','g182','g204'):
  d=json.loads(load(name));d=d['API']['/server_info']['body'] if name=='readiness' else d
  need('expert_distribution_recorder_mode' in d and d['expert_distribution_recorder_mode'] is None,'recorder mode')
  need(d['expert_balancedness_report_mode']=='off' and d['ep_size']==8 and d['moe_a2a_backend']=='deepep','service parameters')
  out[name]={'mode':None,'balancedness':'off','ep_size':8,'deepep_mode':d['deepep_mode']}
 i=json.loads(load('stability'));need(len(i['files'])==1207 and i['two_SHA_scans_equal'] is True,'index count')
 matches=sum('expert' in key.lower() or key.lower().endswith('.pt') for key in i['files']);need(matches==0,'unexpected indexed expert filename')
 start=json.loads(load('start'));need(start['role']=='d' and start['status']==200,'start status')
 tree=ast.parse(load('job'));values=[n.value for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='record_shapes'];need(len(values)==1 and isinstance(values[0],ast.Constant) and values[0].value is False,'shape capture flag')
 return {'snapshots':out,'indexed_filenames':len(i['files']),'expert_or_pt_names':matches,'profile_record_shapes':False,'profile_start_status':200,'scope':'historical snapshots and indexed names only; cannot prove capture-time dynamic state or contents of unindexed files'}


if __name__=='__main__':
 try:print(json.dumps(check(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError,SyntaxError) as e:raise SystemExit('FAILED: '+str(e))
