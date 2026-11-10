#!/usr/bin/env python3
"""Use the exact previously verified public bytes plus fresh connector SHAs."""
from pathlib import Path
from hashlib import sha1,sha256
import json,re
from update_target_draft_metadata_maps import ROOT,TITLE,REPORT,BODY,NEXT,PATTERN,banner,frontier
PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def blob(b):return sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def main():
 checks=json.loads(Path('/tmp/target-draft-public-base-checks.json').read_text());prior=json.loads(Path('/tmp/input-resource-publication-payload.json').read_text());prior_files={f['path']:f for f in prior['files']};receipt=json.loads((ROOT/'input-resource-publication.json').read_text());assert checks['base_commit']==receipt['commit']=='bd4d8bedf1b9c09b02d6d0852208d82dfae535cc';assert receipt['all_content_equal']
 files=[];base_checks=[]
 def add(path,content,source):
  b=content.encode();files.append(dict(path=PREFIX+path,content=content,local_source=source,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b)))
 names=['target-draft-metadata-primary.json','target-draft-metadata-dependencies.json','target-draft-graph-dependencies.json','target-draft-metadata-review.json','target-draft-metadata-price-raw.json','target-draft-metadata-price.json','target-draft-metadata-source-excerpts.json','target-draft-metadata-frontier.json','target-draft-metadata-integrity.json',REPORT]
 for n in names:add('handoff/evidence/'+n,(ROOT/n).read_text(),str(ROOT/n))
 for n in ['read_target_draft_metadata_dependencies.py','read_target_draft_metadata_price.py','review_target_draft_metadata.py','verify_target_draft_metadata.py','update_target_draft_metadata_maps.py','prepare_target_draft_metadata_publication.py','stage_target_draft_metadata_blobs.py','verify_target_draft_metadata_publication.py']:add('tools/'+n,(Path('tools')/n).read_text(),'tools/'+n)
 for p in sorted((ROOT/'target-draft-metadata-source').rglob('*.py')):add('handoff/evidence/'+str(p.relative_to(ROOT)),p.read_text(),str(p))
 for item in checks['files']:
  name=item['path'];old=prior_files[PREFIX+name];b=old['content'].encode();assert blob(b)==item['sha']==old['git_blob_sha1'];assert sha256(b).hexdigest()==old['sha256'];base_checks.append(dict(path=name,sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),bytes=len(b)));s=b.decode()
  if name.endswith('.html'):
   old=s;s,n=re.subn(PATTERN,lambda _:banner('../'+REPORT),s,count=1,flags=re.S);assert n==1 and re.sub(PATTERN,'',old,flags=re.S)==re.sub(PATTERN,'',s,flags=re.S)
  elif name.endswith('.md'):
   body=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY;i=s.index('\n')+1;assert '## 当前断点 · '+TITLE not in s;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+body+'\n'+s[i:]
   if name=='handoff/README.md':
    s=re.sub(r'^当前源码覆盖：.*$',lambda _:'当前源码覆盖：原156位仍90受限绑定/66待追；新增gid12整段递归交付判断、15身份/34原Python控制与176旧调用，不增加原位完成数。target draft metadata完整body值得深入，图/状态/native位与无探针净价仍待确认。全链路未闭合。',s,count=1,flags=re.M)
  else:
   d=json.loads(s);d.update(status=frontier()['status'],target_draft_metadata_review=frontier(),next_action=NEXT,whole_chain_complete=False,completeness_established=False,new_framework_replacement_admissions=0)
   if name=='progress.json':d.update(active_loop='loop-047',overall_complete=False,latest_review='Run155707Z bounded gid12 source/old interval PASSNA;90bound66pending; Run163605Z PLANNED',previous_goal_turn_classification='PROGRESS:independent whole target gid12 delivery;15 current source pins,34 opaque Python controls,176 exact old last-of26 calls with all nested FT boundary union and saved clock guard. Whole body material, retained concrete graph/state/native/net gates. No newprimitive/framework admission; original90/66 unchanged.')
   s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
  add(name,s,'Exact verified public base '+checks['base_commit']+'; fresh connector blob SHA; preserve all other content')
 manifest=dict(base_commit=checks['base_commit'],base_tree_sha='04d66ae6660fff16b76717ad398353721b3972f5',paths=[f['path'] for f in files]+[PREFIX+'handoff/evidence/target-draft-metadata-publication-manifest.json'],base_checks=base_checks,full_diagnostic_call_records=176,all_call_intervals_and_pins_included=True,whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,HTML_only_own_status_aside=True)
 p=ROOT/'target-draft-metadata-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add('handoff/evidence/'+p.name,p.read_text(),str(p));assert len(files)==len({f['path'] for f in files});out=Path('/tmp/target-draft-publication-payload.json');out.write_text(json.dumps(dict(base_commit=checks['base_commit'],base_tree_sha=manifest['base_tree_sha'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),payload_bytes=out.stat().st_size,max_file_bytes=max(f['bytes'] for f in files))))
if __name__=='__main__':main()
