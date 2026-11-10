#!/usr/bin/env python3
"""Create explicit publication payload and raw-artifact/source provenance manifest."""
from pathlib import Path
from hashlib import sha256
import json
from update_smla_direct_l1_maps import SUMMARY,NEXT

if __name__=='__main__':
 root=Path('audit/execution-search');pf='projects/910c-deepseek-v4.1-framework-investigation/';files=[]
 def add(p,dest=None):
  p=Path(p);files.append(dict(path=pf+(dest or 'handoff/evidence/'+p.name),content=p.read_text(),local_source=str(p)))
 for name in ['SMLA原序直达L1与尾块复用判断.md','smla-direct-l1-source-review.json','smla-direct-l1-frontier.json','smla-direct-l1-exploration-notes.json','smla-direct-l1-price.json','smla-l1-catalog.json','smla-l1-dependencies.json','smla-direct-producer-identity.json','engram-storage-identity.json','engram-storage-entry.json']:add(root/name)
 for folder,names in [('smla-transfer-plan-cpu-a1',['build.log','driver.cpp']),('smla-transfer-plan-cpu-a2',['build.log','driver.cpp','result.json']),('smla-l1-layout-cpu-a1',['field.cpp','build-2201.log','build-3510.log','result.json'])]:
  for name in names:add(root/folder/name,'handoff/evidence/'+folder+'/'+name)
 previous=json.loads((root/'smla-schedule-public-source-manifest.json').read_text());old={r['runtime_path']:r for r in previous['sources']};review=json.loads((root/'smla-direct-l1-source-review.json').read_text());entry=json.loads((root/'engram-storage-entry.json').read_text());source_records=[]
 for row in review['sources']+entry['sources']:
  p=Path(row['path']);assert sha256(p.read_bytes()).hexdigest()==row['sha256'];prior=old.get(row['runtime_path'])
  if prior and prior['sha256']==row['sha256']:
   dest=prior['public_path'];commit='c589af115c9bfda4b7d47cc12d2818a1f9b1f235';new=False
  else:
   dest=pf+'handoff/evidence/smla-direct-l1-selected-source/'+row['runtime_path'].lstrip('/');add(p,dest[len(pf):]);commit=None;new=True
  source_records.append(dict(runtime_path=row['runtime_path'],sha256=row['sha256'],reviewed_ranges=row['reviewed_ranges'],public_path=dest,prior_source_commit=commit,new_body_published=new))
 omitted=[]
 for rel in ['smla-transfer-plan-cpu-a2/address-traces.json.zlib','smla-transfer-plan-cpu-a2/driver','smla-l1-layout-cpu-a1/field-2201','smla-l1-layout-cpu-a1/field-3510']:
  p=root/rel;omitted.append(dict(local_path=str(p),bytes=p.stat().st_size,sha256=sha256(p.read_bytes()).hexdigest(),reason='Local binary/compressed trace; full reproducible source, compiler command and result are public. Omitted body is not claimed publicly uploaded.'))
 manifest=dict(source_strategy='9new exact current source bodies plus5previous at c589af1; listed ranges only, not full-file/all-branch/loadedbinary proof. Includes3Engram next-entry bodies, not a completed Engram owner review.',sources=source_records,omitted_artifacts=omitted,raw_price_artifact=dict(path='handoff/evidence/smla-direct-l1-price.json',sha256=sha256((root/'smla-direct-l1-price.json').read_bytes()).hexdigest(),projection=False),whole_chain_complete=False,every_replacement_bitwise=True)
 p=root/'smla-direct-l1-public-source-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(p)
 for name in ['read_smla_l1_catalog.py','read_smla_l1_dependencies.py','read_smla_direct_producer_identity.py','probe_smla_transfer_plan_cpu.py','probe_smla_l1_layout_cpu.py','analyze_smla_direct_l1_price.py','analyze_smla_direct_l1_frontier.py','read_engram_storage_entry.py','update_smla_direct_l1_maps.py','prepare_smla_direct_l1_publication.py']:add(Path('tools')/name,'tools/'+name)
 current=json.loads((root/'source-frontier.json').read_text());payload=dict(files=files,summary=SUMMARY,next=NEXT,smla_direct_l1_review=current['smla_direct_l1_review'],engram_storage_entry=current['engram_storage_entry'],frontiers=current['frontiers'],source_base_commit='d756ab19eb5bf045a2c55055efa3f38526f5b68b')
 p=Path('/tmp/smla-direct-l1-publication-payload.json');p.write_text(json.dumps(payload,ensure_ascii=False));raw=p.read_text();print(json.dumps(dict(files=len(files),characters=len(raw),chunks6000=(len(raw)+5999)//6000,path=str(p),source_bodies_new=sum(r['new_body_published'] for r in source_records),source_bodies_prior=sum(not r['new_body_published'] for r in source_records))))
