#!/usr/bin/env python3
"""Filename/line-number discovery only in one fixed source subtree; no source excerpts."""
import json
import os
from pathlib import Path
import re

ROOTS = ('/vllm-workspace/vllm/vllm/entrypoints/openai',)
MARKERS = {
    'completions_route': re.compile(r'/v1/completions|["\']/?completions["\']'),
    'completion_request': re.compile(r'\bCompletionRequest\b'),
    'to_sampling_params': re.compile(r'\bto_sampling_params\b'),
    'structured_outputs': re.compile(r'\bstructured_outputs\b'),
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
            candidate=bool(keys & {'completions_route','completion_request'}) and bool(keys & {'to_sampling_params','structured_outputs','completions_route'})
            if not candidate:
                continue
            lower=str(path).lower()
            score=len(keys)
            rows.append({'path':str(path),'score':score,
                         'markers':{k:{'count':len(v),'first_line_numbers':v[:8]} for k,v in sorted(hits.items())}})
rows.sort(key=lambda x:(-x['score'],x['path']))
result={'scope':'filename and line numbers only; fixed source subtree; no content, prompts, env or credentials',
        'root_status':root_status,'python_files_scanned':scanned,'files_too_large_skipped':too_large,
        'read_errors':errors,'candidate_files_total':len(rows),'limit':MAX_RESULTS,'candidates':rows[:MAX_RESULTS]}
result['omitted_candidates']=len(rows)-len(result['candidates'])
while len(json.dumps(result,ensure_ascii=True).encode())>MAX_RESULT_BYTES and result['candidates']:
    result['candidates'].pop()
    result['omitted_candidates']+=1
print(json.dumps(result,ensure_ascii=True,separators=(',',':')))
