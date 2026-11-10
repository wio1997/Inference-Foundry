#!/usr/bin/env python3
"""Read the owned Ascend package terms, without claiming dynamic-call absence."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_input_resource_callers import REMOTE
ROOT=Path('audit/execution-search')
def main():
 terms=['AttentionSelectorConfig','_cached_get_attn_backend','VLLM_MOE_SKIP_PADDING','engram_rotated','is_padding']
 start=REMOTE.index('terms=');end=REMOTE.index('\ninventory=',start);remote=REMOTE[:start]+'terms='+repr(terms)+REMOTE[end:]
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',remote],capture_output=True,text=True,timeout=150)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout);d.update(base_reader_sha256=sha256(Path('tools/read_input_resource_callers.py').read_bytes()).hexdigest(),reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),query_terms=terms,whole_chain_complete=False)
 (ROOT/'model-gate-selector-lexical.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(python_files=d['python_files'],matched_files=len(d['matches']),matched=[dict(path=x['runtime_path'],terms=sorted({t for h in x['hits'] for t in h['terms']})) for x in d['matches']],remote_writes=[]),ensure_ascii=False))
if __name__=='__main__':main()
