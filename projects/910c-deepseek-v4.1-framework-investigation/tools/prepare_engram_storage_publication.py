#!/usr/bin/env python3
"""Explicit public owner/price payload; large sources use disclosed exact excerpts."""
from pathlib import Path
from hashlib import sha256,sha1
import json
from update_engram_storage_maps import SUMMARY,NEXT

if __name__=='__main__':
 root=Path('audit/execution-search');prefix='projects/910c-deepseek-v4.1-framework-investigation/';files=[]
 def add(p,dest=None):
  p=Path(p);files.append(dict(path=prefix+(dest or 'handoff/evidence/'+p.name),content=p.read_text(),local_source=str(p)))
 for name in ['Engram成对UVA寿命与整段lookup价值判断.md','engram-storage-source-review.json','engram-storage-frontier.json','engram-storage-price.json','engram-storage-catalog.json','engram-storage-dependencies.json','engram-storage-sdk.json','engram-storage-exploration-notes.json','cache-epilogue-identity.json','cache-epilogue-entry.json','engram-storage-balanced-supplement.json']:add(root/name)
 prior=json.loads((root/'smla-direct-l1-public-source-manifest.json').read_text());old={r['runtime_path']:r for r in prior['sources']};records={}
 reviews=json.loads((root/'engram-storage-source-review.json').read_text())['sources'];entry=json.loads((root/'cache-epilogue-entry.json').read_text())['sources']
 for row in reviews+entry:
  rp=row['runtime_path']
  if rp in records:
   assert row['sha256']==records[rp]['sha256'];records[rp]['reviewed_ranges']=sorted({tuple(v) for v in records[rp]['reviewed_ranges']+row['reviewed_ranges']});continue
  records[rp]=dict(row)
 public=[]
 for rp,row in records.items():
  p=Path(row['path']);raw=p.read_bytes();assert sha256(raw).hexdigest()==row['sha256'];r=dict(runtime_path=rp,sha256=row['sha256'],reviewed_ranges=row['reviewed_ranges'],raw_local_path=str(p),raw_bytes=len(raw));prev=old.get(rp)
  if prev and prev['sha256']==r['sha256']:
   r.update(public_path=prev['public_path'],public_representation='prior_full_body',prior_commit=prev.get('prior_source_commit') or '70b68809c1443f3372455812d3701941f010dc7a',expected_git_blob=sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
  elif rp.endswith('/vllm_ascend/ops/layernorm.py'):
   r.update(public_path=prefix+'handoff/evidence/index-key-producer-runtime-source/vllm-ascend/vllm_ascend/ops/layernorm.py',public_representation='prior_full_body',prior_commit='10693f19dca6531edf332bbaa1aa10f7bb986146',expected_git_blob=sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
  elif not row['reviewed_ranges']:
   r.update(public_path=None,public_representation='identity_only_no_body_published',reason='No body reviewed; full local raw source retained, source pin only.')
  elif len(raw)>50000:
   lines=raw.decode().splitlines();ex=dict(runtime_path=rp,full_file_sha256=r['sha256'],full_file_bytes=len(raw),full_body_public=False,reviewed_ranges=row['reviewed_ranges'],excerpts=[dict(first=a,last=b,text='\n'.join(lines[a-1:b])+'\n') for a,b in row['reviewed_ranges']],scope='Exact listed line text from complete current raw source; full raw body remains local. No unseen ranges claimed inspected.')
   name='engram-storage-excerpts/'+rp.lstrip('/')+'.json';files.append(dict(path=prefix+'handoff/evidence/'+name,content=json.dumps(ex,ensure_ascii=False,indent=2)+'\n',local_source=str(p)+' (exact reviewed excerpts only)'));r.update(public_path=prefix+'handoff/evidence/'+name,public_representation='exact_reviewed_excerpts_only')
  else:
   dest='handoff/evidence/engram-storage-selected-source/'+rp.lstrip('/');add(p,dest);r.update(public_path=prefix+dest,public_representation='new_full_body')
  public.append(r)
 manifest=dict(strategy='Reuse6 prior immutable full sources, publish exact reviewed excerpts for large bodies and complete smaller sources; identity-only hasno body. All full raw source/header bodies remain local withSHA. Sourcecount/ranges do not prove alllegalbranches/loadedbinary equivalence.',sources=public,raw_price=dict(path='handoff/evidence/engram-storage-price.json',sha256=sha256((root/'engram-storage-price.json').read_bytes()).hexdigest(),projection=False),whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,late_profile_supplement='engram-storage-balanced-supplement.json',frozen_loop036_report_scope='Pre-profile-completion snapshot; newcompletedbalancedcohort keptseparate, oldLoop036artifacts unchanged.')
 p=root/'engram-storage-public-source-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(p)
 for name in ['read_engram_storage_dependencies.py','read_engram_storage_catalog.py','read_engram_storage_sdk.py','analyze_engram_storage_price.py','analyze_engram_storage_frontier.py','read_cache_epilogue_entry.py','update_engram_storage_maps.py','prepare_engram_storage_publication.py','analyze_engram_balanced_supplement.py']:add(Path('tools')/name,'tools/'+name)
 current=json.loads((root/'source-frontier.json').read_text());value=dict(files=files,summary=SUMMARY,next=NEXT,engram_storage_review=current['engram_storage_review'],cache_epilogue_entry=current['cache_epilogue_entry'],Engram_review=current['Engram_review'],source_base_commit='2729de670e52dc2f55103b8e96793beddd1fddd1',reused_sources=[r for r in public if r['public_representation']=='prior_full_body'],engram_storage_balanced_supplement=current['engram_storage_balanced_supplement'])
 p=Path('/tmp/engram-storage-publication-payload.json');p.write_text(json.dumps(value,ensure_ascii=False));n=len(p.read_text());print(json.dumps(dict(files=len(files),characters=n,chunks6000=(n+5999)//6000,representations={k:sum(r['public_representation']==k for r in public) for k in sorted({r['public_representation'] for r in public})},path=str(p))))
