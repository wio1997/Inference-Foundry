#!/usr/bin/env python3
"""Extend the safe saved-launch whitelist; never print environment/credentials."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_input_frozen_flags import FLAGS
ROOT=Path('audit/execution-search')
def main():
 remote=FLAGS.replace("'--ec-transfer-config'};out={}","'--ec-transfer-config','--async-scheduling','--no-async-scheduling','--distributed-executor-backend','--disable-cascade-attn'};out={}")
 remote=remote.replace("keys=['multistream_engram_overlap'","keys=['enable_kvpp','multistream_engram_overlap'")
 remote=remote.replace("{'all','align',True,False}","{'all','align','none','mp','ray','uni','external_launcher','True','False',True,False}")
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',remote],capture_output=True,text=True,timeout=60)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout);assert d['sha256']==json.loads((ROOT/'input-frozen-flags.json').read_text())['saved_launch']['sha256'];d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),base_reader_sha256=sha256(Path('tools/read_input_frozen_flags.py').read_bytes()).hexdigest(),new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
 (ROOT/'worker-protocol-launch-flags.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(d,ensure_ascii=False))
if __name__=='__main__':main()
