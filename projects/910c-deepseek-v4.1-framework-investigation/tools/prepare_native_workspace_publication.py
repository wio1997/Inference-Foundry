#!/usr/bin/env python3
"""Prepare explicit local evidence payload; fresh public maps are merged separately."""
from pathlib import Path
from hashlib import sha256
import json
from update_native_workspace_maps import SUMMARY,NEXT

def main():
 prefix='projects/910c-deepseek-v4.1-framework-investigation/'
 root=Path('audit/execution-search')
 names=['原生workspace寿命与融合容量判断.md','native-workspace-catalog.json','native-workspace-backend-records.json','native-workspace-source-review.json','native-workspace-source-review-v2.json','native-workspace-source-review-correction.json','native-workspace-frontier.json','native-workspace-exploration-notes.json','native-workspace-upstream-retrieval.json','smla-metadata-execution-identity.json','smla-metadata-delivery-entry.json']
 payload=[]
 for n in names:
  p=root/n;payload.append(dict(path=prefix+'handoff/evidence/'+n,content=p.read_text(),local_source=str(p)))
 entry=json.loads((root/'smla-metadata-delivery-entry.json').read_text())
 for r in entry['sources']:
  p=Path(r['path']);assert sha256(p.read_bytes()).hexdigest()==r['sha256']
  payload.append(dict(path=prefix+'handoff/evidence/smla-metadata-selected-source/'+r['runtime_path'].lstrip('/'),content=p.read_text(),local_source=str(p)))
 tools=['read_native_workspace_catalog.py','read_native_workspace_backend_records.py','analyze_native_workspace_frontier.py','read_smla_metadata_identity.py','update_native_workspace_maps.py','prepare_native_workspace_publication.py','correct_native_workspace_review_ranges.py']
 for n in tools:
  p=Path('tools')/n;payload.append(dict(path=prefix+'tools/'+n,content=p.read_text(),local_source=str(p)))
 upstream=json.loads((root/'native-workspace-source-review-v2.json').read_text())
 manifest=dict(strategy='All24 exact upstream references retained locally; public immutable upstream commit URLs, SHA256 and reviewed ranges in native-workspace-source-review.json. Upstream source bodies are not duplicated in this publication.',source_review_sha256=sha256((root/'native-workspace-source-review.json').read_bytes()).hexdigest(),upstream_references=[dict(path=r['path'],url=r['url'],sha256=r['sha256'],bytes=r['bytes'],reviewed_ranges=r['reviewed_ranges']) for r in upstream['sources']],actual_SMLA_runtime_snapshots=[dict(path=x['path'],local_source=x['local_source'],sha256=sha256(Path(x['local_source']).read_bytes()).hexdigest()) for x in payload if '/smla-metadata-selected-source/' in x['path']])
 p=root/'native-workspace-public-source-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');payload.append(dict(path=prefix+'handoff/evidence/'+p.name,content=p.read_text(),local_source=str(p)))
 local=json.loads((root/'source-frontier.json').read_text())
 out=dict(files=payload,summary=SUMMARY,next=NEXT,native_workspace_review=local['native_workspace_review'],smla_metadata_delivery_entry=local['smla_metadata_delivery_entry'])
 p=Path('/tmp/native-workspace-publication-payload.json');p.write_text(json.dumps(out,ensure_ascii=False))
 print(json.dumps(dict(path=str(p),files=len(payload),characters=len(p.read_text()),source_snapshots=sum('/smla-metadata-selected-source/' in r['path'] for r in payload))))

if __name__=='__main__':main()
