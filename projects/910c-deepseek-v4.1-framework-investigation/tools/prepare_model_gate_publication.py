#!/usr/bin/env python3
"""Preserve fresh public base bytes; append this model-guard/consumer review."""
from pathlib import Path
from hashlib import sha1,sha256
import json,re
from update_model_gate_maps import ROOT,TITLE,REPORT,BODY,NEXT,PATTERN,banner,frontier
PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def blob(b):return sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def main():
 checks=json.loads(Path('/tmp/model-gate-public-base-checks.json').read_text());prior=json.loads(Path('/tmp/worker-protocol-publication-payload.json').read_text());prior_files={f['path']:f for f in prior['files']};receipt=json.loads((ROOT/'worker-protocol-publication.json').read_text());assert checks['base_commit']==receipt['commit']=='58eddc81886e4fe4eac1408741b4981d5f5f60ed' and receipt['all_content_equal'];files=[];base_checks=[]
 def add(path,content,source):
  b=content.encode();files.append(dict(path=PREFIX+path,content=content,local_source=source,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b)))
 evidence=['model-gate-dependencies.json','model-gate-consumer-dependencies.json','model-gate-state.json','model-gate-selector-lexical.json','model-gate-reader-a1-invalid.json','model-gate-source-controls.json','model-gate-control-source-excerpts.json','model-gate-source-review.json','model-gate-parent-prices.json','model-gate-review-history.json','model-pad-row-parent-primary.json','model-rotated-engram-parent-primary.json','model-rotated-engram-parent-a1-label-correction.json','model-gate-frontier.json','model-gate-integrity.json',REPORT]
 for n in evidence:add('handoff/evidence/'+n,(ROOT/n).read_text(),str(ROOT/n))
 for n in ['read_model_gate_dependencies.py','read_model_gate_consumers.py','read_model_gate_state.py','read_model_gate_lexical.py','probe_model_gate_source_controls.py','review_model_gates.py','update_model_gate_maps.py','verify_model_gate_evidence.py','prepare_model_gate_publication.py','stage_model_gate_blobs.py','verify_model_gate_publication.py']:add('tools/'+n,(Path('tools')/n).read_text(),'tools/'+n)
 for p in sorted((ROOT/'model-gate-source').rglob('*.py')):add('handoff/evidence/'+str(p.relative_to(ROOT)),p.read_text(),str(p))
 for item in checks['files']:
  name=item['path']
  if PREFIX+name in prior_files:
   f=prior_files[PREFIX+name];b=f['content'].encode();assert blob(b)==f['git_blob_sha1'] and sha256(b).hexdigest()==f['sha256']
  else:
   assert name=='modules/M06.md';b=Path('/tmp/model-gate-public-M06.md').read_bytes()
   if blob(b)!=item['sha'] and b.endswith(b'\n') and blob(b[:-1])==item['sha']:b=b[:-1]
  assert blob(b)==item['sha'];base_checks.append(dict(path=name,sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),bytes=len(b)));s=b.decode()
  if name.endswith('.html'):
   old=s;s,n=re.subn(PATTERN,lambda _:banner('../'+REPORT),s,count=1,flags=re.S);assert n==1 and re.sub(PATTERN,'',old,flags=re.S)==re.sub(PATTERN,'',s,flags=re.S)
  elif name.endswith('.md'):
   body=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY;i=s.index('\n')+1;assert '## 当前断点 · '+TITLE not in s;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+body+'\n'+s[i:]
   if name=='handoff/README.md':s=re.sub(r'^当前源码覆盖：.*$',lambda _:'当前源码覆盖：原156位122受限源码绑定/34动态待追；27当前身份、5模型guard源码条件、85原Python/布尔行/opaque控制。21进程仅初始单项env，非loaded cache；generic runner/cache两backend派发分列。A3 hash/vision已有native，当前rotated Engram不套unrotated gate。大row/metadata/owner候选保留，全链路未闭合。',s,count=1,flags=re.M)
  else:
   d=json.loads(s);d.update(status=frontier()['status'],model_gate_review=frontier(),next_action=NEXT,whole_chain_complete=False,completeness_established=False,new_framework_replacement_admissions=0)
   if name=='progress.json':d.update(active_loop='loop-047',overall_complete=False,latest_review='Run170051Z bounded5modelguards/source dispatch/old-qualified-cost PASSNA;122bound34pending; Run174546Z PLANNED',previous_goal_turn_classification='PROGRESS:27 current source pins,21 whitelisted initial-env records,85 original source/geometry/opaque controls;3 new constraints,122/34 ledger. Actual generic selector/cache paths and A3 hash/vision/rotated Engram selection resolved in source scope; independent larger pad-row primary retained, historical6source/type/cost intersection reuses stopped small recipes. Initial path/harness/rank-count/quant-label corrections retained. No newprimitive/framework admission or full-chain claim.')
   s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
  add(name,s,'Fresh SHA-checked exact public base '+checks['base_commit']+'; preserve other content')
 dest='handoff/evidence/model-gate-publication-manifest.json';manifest=dict(base_commit=checks['base_commit'],base_tree_sha=receipt['tree'],paths=[f['path'] for f in files]+[PREFIX+dest],base_checks=base_checks,source_pins=27,source_bindings=122,pending_original_sites=34,model_guard_source_legs=5,original_Python_controls=85,old_DFC_representatives_only=[0,8],HTML_only_own_status_aside=True,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False)
 p=ROOT/'model-gate-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(dest,p.read_text(),str(p));assert len(files)==len({f['path'] for f in files});out=Path('/tmp/model-gate-publication-payload.json');out.write_text(json.dumps(dict(base_commit=checks['base_commit'],base_tree_sha=receipt['tree'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),payload_bytes=out.stat().st_size,max_file_bytes=max(f['bytes'] for f in files))))
if __name__=='__main__':main()
