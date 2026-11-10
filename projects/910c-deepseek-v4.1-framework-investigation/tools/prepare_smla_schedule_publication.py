#!/usr/bin/env python3
"""Build an explicit public evidence payload while preserving prior public maps."""
from pathlib import Path
from hashlib import sha256
import json
from update_smla_schedule_maps import SUMMARY,NEXT

if __name__=='__main__':
 root=Path('audit/execution-search');pf='projects/910c-deepseek-v4.1-framework-investigation/';files=[]
 names=['SMLA分核包与原顺序交付判断.md','smla-schedule-catalog.json','smla-schedule-dependencies.json','smla-schedule-source-review.json','smla-schedule-frontier.json','smla-schedule-exploration-notes.json','smla-address-plan-paid-price.json','smla-direct-kv-entry.json','smla-package-recheck.json']
 for name in names:
  p=root/name;files.append(dict(path=pf+'handoff/evidence/'+name,content=p.read_text(),local_source=str(p)))
 for name in ['result.json','driver.cpp','stdout.jsonl','build.stdout','build.stderr','stderr.log']:
  p=root/'smla-schedule-cpu-a1'/name;files.append(dict(path=pf+'handoff/evidence/smla-schedule-cpu-a1/'+name,content=p.read_text(),local_source=str(p)))
 for name in ['reader.py','error.json']:
  p=root/'smla-package-recheck-a1'/name;files.append(dict(path=pf+'handoff/evidence/smla-package-recheck-a1/'+name,content=p.read_text(),local_source=str(p)))
 deps=json.loads((root/'smla-schedule-dependencies.json').read_text())
 for row in deps['sources']:
  p=Path(row['local_path']);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];files.append(dict(path=pf+'handoff/evidence/smla-schedule-selected-source/'+row['runtime_path'].lstrip('/'),content=p.read_text(),local_source=str(p)))
 for name in ['read_smla_schedule_catalog.py','read_smla_schedule_dependencies.py','analyze_smla_address_plan_paid_price.py','probe_smla_schedule_cpu.py','analyze_smla_schedule_frontier.py','update_smla_schedule_maps.py','read_smla_package_recheck.py','prepare_smla_schedule_publication.py']:
  p=Path('tools')/name;files.append(dict(path=pf+'tools/'+name,content=p.read_text(),local_source=str(p)))
 review=json.loads((root/'smla-schedule-source-review.json').read_text());current={r['runtime_path'] for r in deps['sources']};records=[]
 for row in review['source_pins']:
  new=row['runtime_path'] in current;records.append(dict(runtime_path=row['runtime_path'],sha256=row['sha256'],reviewed_ranges=row['reviewed_ranges'],public_path=pf+'handoff/evidence/'+('smla-schedule-selected-source/' if new else 'smla-metadata-selected-source/')+row['runtime_path'].lstrip('/'),prior_source_commit=None if new else 'd65d1e060fa26ad60a5c57b9f168d8675bd18925'))
 manifest=dict(strategy='13 newly referenced dependency bodies published;8 previous exact source snapshots retained at d65d1e0. Only reviewed ranges count as inspected; no loaded binary/source equivalence or percall deterministic proof.',source_review_sha256=sha256((root/'smla-schedule-source-review.json').read_bytes()).hexdigest(),sources=records)
 p=root/'smla-schedule-public-source-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');files.append(dict(path=pf+'handoff/evidence/'+p.name,content=p.read_text(),local_source=str(p)))
 current=json.loads((root/'source-frontier.json').read_text());payload=dict(files=files,summary=SUMMARY,next=NEXT,smla_schedule_review=current['smla_schedule_review'],smla_direct_kv_entry=current['smla_direct_kv_entry'])
 p=Path('/tmp/smla-schedule-publication-payload.json');p.write_text(json.dumps(payload,ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),path=str(p))))
