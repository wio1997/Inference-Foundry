#!/usr/bin/env python3
"""Filename/line-number discovery only in two fixed source roots; no source excerpts."""
import json
import os
from pathlib import Path
import re

ROOTS = ('/vllm-workspace/vllm/vllm', '/vllm-workspace/vllm-ascend/vllm_ascend')
MARKERS = {
    'metadata_constructor': re.compile(r'\bSpecDecodeMetadata\s*\('),
    'metadata_ref': re.compile(r'\bspec_decode_metadata\b'),
    'draft_count': re.compile(r'\bnum_draft_tokens\b'),
    'sampled_ids': re.compile(r'\bsampled_token_ids\b'),
    'rejection_sampler': re.compile(r'\brejection_sampler\b'),
    'spec_limit': re.compile(r'\bnum_speculative_tokens\b'),
    'output_update': re.compile(r'\b(?:update_from_output|_update_from_output|process_outputs|append_output_token_ids|_append_output_token_ids|sample_tokens)\b'),
}
MAX_FILE_BYTES = 1024*1024
MAX_RESULT_BYTES = 24000
MAX_RESULTS = 70
rows=[]
scanned=0
too_large=0
errors=0
root_status={}
for root in ROOTS:
    root_status[root] = 'OK' if Path(root).is_dir() else 'MISSING'
    if root_status[root] != 'OK':
        continue
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in ('__pycache__','node_modules'))
        for name in sorted(files):
            if not name.endswith('.py'):
                continue
            path=Path(parent)/name
            try:
                if path.is_symlink():
                    continue
                if path.stat().st_size > MAX_FILE_BYTES:
                    too_large+=1
                    continue
                text=path.read_text(encoding='utf-8', errors='replace')
            except OSError:
                errors+=1
                continue
            scanned+=1
            hits={}
            for line_no,line in enumerate(text.splitlines(),1):
                for key,pattern in MARKERS.items():
                    if pattern.search(line):
                        hits.setdefault(key,[]).append(line_no)
            keys=set(hits)
            candidate=('metadata_constructor' in keys or
                       {'draft_count','sampled_ids'} <= keys or
                       {'metadata_ref','sampled_ids'} <= keys or
                       {'rejection_sampler','sampled_ids'} <= keys or
                       {'output_update','sampled_ids'} <= keys)
            if not candidate:
                continue
            lower=str(path).lower()
            score=(100 if 'metadata_constructor' in keys else 0)+\
                  (25 if 'runner' in lower else 0)+(20 if 'scheduler' in lower else 0)+\
                  (15 if 'output' in lower else 0)+(10 if 'spec_decode' in lower else 0)+len(keys)
            rows.append({'path':str(path),'score':score,
                         'markers':{k:{'count':len(v),'first_line_numbers':v[:8]} for k,v in sorted(hits.items())}})
rows.sort(key=lambda x:(-x['score'],x['path']))
result={'scope':'filename and line numbers only; fixed source roots; no content, prompts, env or credentials',
        'root_status':root_status,'python_files_scanned':scanned,'files_too_large_skipped':too_large,
        'read_errors':errors,'candidate_files_total':len(rows),'limit':MAX_RESULTS,'candidates':rows[:MAX_RESULTS]}
result['omitted_candidates']=len(rows)-len(result['candidates'])
while len(json.dumps(result,ensure_ascii=True).encode())>MAX_RESULT_BYTES and result['candidates']:
    result['candidates'].pop()
    result['omitted_candidates']+=1
print(json.dumps(result,ensure_ascii=True,separators=(',',':')))
