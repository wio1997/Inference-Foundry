# SPDX-License-Identifier: Apache-2.0
"""Read-only private generated-ID evidence audit, no model/SDK/NPU calls."""
import pathlib,json,hashlib,subprocess,shlex
P=pathlib.Path(__file__).resolve().parents[2];R=P/'runs/20261017-r139-actual-flash-padded-vocab-dataset';x=json.loads(json.loads((R/'node14.json').read_text())['stdout']);actual=x['result'];expected={d['concurrency']:d['manifest_sha256'] for d in actual['datasets']};assert actual['passed'] and x['post_inspect']['stdout'].strip()=='false 0'
code=r'''import pathlib,json,hashlib,socket
assert socket.gethostname()=='A6-15U-Atlas900RCKA3-14'
R=pathlib.Path('/data/wio/inference-foundry/glm53-flash/runs/20261017-r139-actual-flash-padded-vocab-dataset');expected=__EXPECTED__;out={'project_id':'glm53-flash','passed':False,'actual_model_or_SDK_or_NPU_calls':0,'remote_writes':0,'raw_IDs_exported':0,'datasets':[]};seen_full=set()
def h(v):return hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
def lcp(a,b):
 for i,(x,y) in enumerate(zip(a,b)):
  if x!=y:return i
 return min(len(a),len(b))
for c,digest in expected.items():
 d=R/f'private-C{c}';b=(d/'manifest.json').read_bytes();assert hashlib.sha256(b).hexdigest()==digest;m=json.loads(b);assert m['actual_tokenizer_vocab']==154856 and m['model_embedding_vocab']==154880 and len(m['rows'])==c;prefixes=[];max_id=0;requests=0
 for row in m['rows']:
  f=d/row['private_file'];assert not f.is_symlink() and f.stat().st_size<1024*1024;p=json.loads(f.read_text());prefix=p['prefix_ids'];assert len(prefix)==74400 and h(prefix)==p['prefix_ids_sha256']==row['prefix_ids_sha256'] and p['namespace']==m['namespace'];prefixes.append(prefix);first=[]
  for value in prefix:assert type(value)is int and 0<=value<154856
  for v,public in zip(p['variants'],row['variants']):
   suffix=v['suffix_ids'];full=prefix+suffix;assert len(suffix)==5600 and len(full)==80000 and h(suffix)==v['suffix_ids_sha256']==public['suffix_ids_sha256'];assert h(full)==v['input_ids_sha256']==public['input_ids_sha256'];assert v['required_output_tokens']==600 and v['input_ids_sha256'] not in seen_full;seen_full.add(v['input_ids_sha256']);first.append(suffix[0]);max_id=max(max_id,max(full));requests+=1
   for value in suffix:assert type(value)is int and 0<=value<154856
  assert len(p['variants'])==4 and len(set(first))==4
 cross=[lcp(prefixes[i],prefixes[j]) for i in range(c) for j in range(i+1,c)];assert (max(cross) if cross else None)==m['cross_prefix_LCP_max'] and (not cross or max(cross)<=256)
 out['datasets'].append({'concurrency':c,'actual_prefixes':c,'exact80000_full_IDs_verified':requests,'four_suffix_first_tokens_distinct':True,'actual_token_ID_max':max_id,'cross_prefix_LCP_max':max(cross) if cross else None,'manifest_sha256':digest})
assert len(seen_full)==452;out['passed']=True;out['unique_full_ID_sequences']=len(seen_full);out['formal_requests_issued']=0;out['cache_warmup_requests_issued']=0;print(json.dumps(out,indent=2))
'''.replace('__EXPECTED__',repr(expected))
ssh=['ssh','-S','/Users/wio/work/sglang-glm53-pd/ssh-control','-p','60011','-oBatchMode=yes','-oConnectTimeout=15','root@61.241.77.34'];cmd=['kubectl','exec','-i','-n','default','zhixing-dockerctl-25xz2','-c','hostctl','--','nsenter','-t1','-m','--','python3','-'];q=subprocess.run(ssh+[shlex.join(cmd)],input=code,capture_output=True,text=True,timeout=50);assert q.returncode==0,q.stderr;v=json.loads(q.stdout);assert v['passed'];(R/'independent-ID-readback.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
