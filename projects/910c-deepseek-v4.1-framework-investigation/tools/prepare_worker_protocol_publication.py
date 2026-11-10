#!/usr/bin/env python3
"""Apply only this protocol delta to a freshly SHA-checked public base."""
from pathlib import Path
from hashlib import sha1,sha256
import json,re
from update_worker_protocol_maps import ROOT,TITLE,REPORT,BODY,NEXT,PATTERN,banner,frontier
PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def blob(b):return sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def main():
 checks=json.loads(Path('/tmp/worker-protocol-public-base-checks.json').read_text());prior=json.loads(Path('/tmp/target-draft-publication-payload.json').read_text());prior_files={f['path']:f for f in prior['files']};receipt=json.loads((ROOT/'target-draft-metadata-publication.json').read_text());assert checks['base_commit']==receipt['commit']=='91a211abde98ac80255c0824dfb301db61d03f15' and receipt['all_content_equal'];files=[];base_checks=[]
 def add(path,content,source):
  b=content.encode();files.append(dict(path=PREFIX+path,content=content,local_source=source,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b)))
 for name in ['worker-protocol-source-review.json','worker-protocol-dependencies.json','worker-protocol-aux-dependencies.json','worker-protocol-launch-flags.json','worker-execution-protocol-controls.json','worker-execution-protocol-source-excerpts.json','worker-protocol-gate-source-excerpts.json','worker-position-parent-primary.json','worker-position-parent-price.json','worker-protocol-control-harness-history.json','worker-protocol-frontier.json','worker-protocol-integrity.json',REPORT]:add('handoff/evidence/'+name,(ROOT/name).read_text(),str(ROOT/name))
 for name in ['read_worker_protocol_dependencies.py','read_worker_protocol_aux.py','read_worker_protocol_flags.py','probe_worker_execution_protocol.py','review_worker_protocol.py','update_worker_protocol_maps.py','verify_worker_protocol.py','prepare_worker_protocol_publication.py','stage_worker_protocol_blobs.py','verify_worker_protocol_publication.py']:add('tools/'+name,(Path('tools')/name).read_text(),'tools/'+name)
 for p in sorted((ROOT/'worker-protocol-source').rglob('*.py')):add('handoff/evidence/'+str(p.relative_to(ROOT)),p.read_text(),str(p))
 for item in checks['files']:
  name=item['path'];old=prior_files[PREFIX+name];b=old['content'].encode();assert blob(b)==item['sha']==old['git_blob_sha1'] and sha256(b).hexdigest()==old['sha256'];base_checks.append(dict(path=name,sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),bytes=len(b)));s=b.decode()
  if name.endswith('.html'):
   old=s;s,n=re.subn(PATTERN,lambda _:banner('../'+REPORT),s,count=1,flags=re.S);assert n==1 and re.sub(PATTERN,'',old,flags=re.S)==re.sub(PATTERN,'',s,flags=re.S)
  elif name.endswith('.md'):
   body=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY;i=s.index('\n')+1;assert '## 当前断点 · '+TITLE not in s;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+body+'\n'+s[i:]
   if name=='handoff/README.md':s=re.sub(r'^当前源码覆盖：.*$',lambda _:'当前源码覆盖：原156位119受限源码绑定/37动态待追；本段24身份、61 worker guard源码leg与人口分列、13 opaque协议+13默认/factory控制。29新增固定gate排除不当原生覆盖。低值早期CPU影子/等待后移小边界仅同配平诊断停止，大frame/gid12保留，全链路未闭合。',s,count=1,flags=re.M)
  else:
   d=json.loads(s);d.update(status=frontier()['status'],worker_protocol_review=frontier(),next_action=NEXT,whole_chain_complete=False,completeness_established=False,new_framework_replacement_admissions=0)
   if name=='progress.json':d.update(active_loop='loop-047',overall_complete=False,latest_review='Run163605Z bounded worker source/protocol/old-budget PASSNA;119bound37pending; Run170051Z PLANNED',previous_goal_turn_classification='PROGRESS:24current source pins;61 worker guard source legs recorded separately from population;29 new frozen gate constraints,119/37 original-site ledger.13 opaque original worker protocol cases plus13 source default/factory controls. Source-first whole early-shadow/wait-retiming family stopped only in balanced diagnostic using relaxed full execute prepare-self budget; larger parents kept. No newprimitive/framework admission.')
   s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
  add(name,s,'Exact verified public base '+checks['base_commit']+';fresh connector fileSHA;other content preserved')
 dest='handoff/evidence/worker-protocol-publication-manifest.json';manifest=dict(base_commit=checks['base_commit'],base_tree_sha=receipt['tree'],paths=[f['path'] for f in files]+[PREFIX+dest],base_checks=base_checks,source_pins=24,source_bindings=119,pending_original_sites=37,source_leg_notes_not_population=61,HTML_only_own_status_aside=True,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False)
 p=ROOT/'worker-protocol-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(dest,p.read_text(),str(p));assert len(files)==len({f['path'] for f in files});out=Path('/tmp/worker-protocol-publication-payload.json');out.write_text(json.dumps(dict(base_commit=checks['base_commit'],base_tree_sha=receipt['tree'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),payload_bytes=out.stat().st_size,max_file_bytes=max(f['bytes'] for f in files))))
if __name__=='__main__':main()
