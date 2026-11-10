#!/usr/bin/env python3
"""Read current disk identities for common-device projection and consumers."""
from pathlib import Path
from hashlib import sha256
import json, subprocess
from read_swa_query_producer_entry import REMOTE
ROOT = Path('audit/execution-search')

def main():
    names = ['common-frame-key-dependencies.json', 'metadata-readers-execution-identity.json',
             'index-key-producer-execution-identity.json']
    prior = {}
    for name in names:
        for r in json.loads((ROOT/name).read_text())['sources']:
            prior[r['runtime_path']] = r
    wanted = json.loads((ROOT/names[0]).read_text())['sources']
    paths = [r['runtime_path'] for r in wanted]
    paths += ['/vllm-workspace/vllm-ascend/vllm_ascend/'+n for n in [
        'models/deepseek_v41/indexer.py', 'models/deepseek_v41/compressor.py',
        'models/deepseek_v41/mixed_quant_attention.py', 'ascend_forward_context.py',
        'attention/dsa_v1.py']]
    spec = []
    for name in paths:
        r = prior[name]; p = Path(r.get('local_path', r.get('path')))
        spec.append(dict(runtime_path=name, local_path=str(p), local_sha256=sha256(p.read_bytes()).hexdigest()))
    r = subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)], capture_output=True, text=True, timeout=115)
    if r.returncode:
        (ROOT/'common-device-entry-error.json').write_text(json.dumps(dict(returncode=r.returncode,error=r.stderr),indent=2)+'\n')
        raise RuntimeError(r.stderr)
    d = json.loads(r.stdout)
    for row in d['sources']:
        body = row.pop('content', None)
        if body is not None:
            p = ROOT/'common-device-runtime-source'/row['runtime_path'].lstrip('/')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
            assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256']
            row['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current disk identities, no claim of full bodies/native execution; reviewed ranges separate.',whole_chain_complete=False)
    (ROOT/'common-device-entry.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['sources']),matches=sum(x['matches_local'] for x in d['sources']),container_state=d['container_state'],new_NPU_work=0,remote_writes=[])))

if __name__=='__main__':main()
