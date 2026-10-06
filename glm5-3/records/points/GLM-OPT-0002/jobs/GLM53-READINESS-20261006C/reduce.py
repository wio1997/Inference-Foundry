"""Resolve JobB's total_size assumption without rewriting its raw or Result."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

SMALL = r'''
import hashlib,json,os,subprocess
from pathlib import Path
from collections import Counter
root=Path('/data/tiankuan/wio/GLM-5.3-w8a8')
index=json.loads((root/'quant_model_weights.safetensors.index.json').read_text())
rot_names=[k for k,v in index['weight_map'].items() if v=='rot.safetensors']
tokenizer=json.loads((root/'tokenizer.json').read_text())
q=json.loads((root/'quant_model_description.json').read_text())
source={}
for filename in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','quant_model_description.json'):
 p=root/filename;b=p.read_bytes();source[filename]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
procs=[]
ps=subprocess.run(['ps','-eo','pid,stat,comm'],text=True,capture_output=True,timeout=10)
for line in ps.stdout.splitlines()[1:]:
 parts=line.split(None,2)
 if len(parts)!=3 or 'Z' in parts[1]:continue
 pid,state,comm=parts
 if not any(k in comm.lower() for k in ('python','vllm','mainthread')):continue
 p=Path('/proc')/pid
 try:
  args=p.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
  if not any(k in ' '.join(args) for k in ('native_acl_lifecycle.py','/bin/vllm','vllm.entrypoints','native_identity_observer.py','glm5-3/runtime/controller.py')):continue
  if any(k in a.lower() for a in args for k in ('password','api-key','token=','bearer')):args=['REDACTED']
  stats=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
  procs.append(dict(pid=int(pid),state=state,start_ticks=stats[19],argv=args,PYTHONPATH=[v.decode(errors='replace') for v in p.joinpath('environ').read_bytes().split(b'\0') if v.startswith(b'PYTHONPATH=')]))
 except (OSError,ValueError):pass
print(json.dumps(dict(rot_index_keys=rot_names,metadata_hashes=source,tokenizer_json_valid=True,tokenizer_model_type=tokenizer['model']['type'],tokenizer_vocab_entries=len(tokenizer['model']['vocab']),added_tokens=len(tokenizer.get('added_tokens',[])),quant_description_types=dict(Counter(v for v in q.values() if isinstance(v,str))),native_roots=procs)))
'''


def main(job_path):
    job=json.loads(Path(job_path).read_text());dest=Path(job['result']['path']).parent
    previous=dest.parent/'GLM53-READINESS-20261006B';observations={};evidence=[]
    for host in ('166','167'):
        p=previous/('observations_'+host+'.raw');raw=p.read_bytes()
        d=json.loads(raw.decode().split('\nSTDERR\n')[0]);m=d['model']
        evidence.append(dict(id='headers'+host,path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),locator='model.shards/model.structural_errors/index.metadata'))
        assert m['structural_errors']==[dict(error='index metadata.total_size differs from tensor payload offsets')]
        rotation=next(s for s in m['shards'] if s['file']=='rot.safetensors')
        rotation_payload=rotation['bytes']-rotation['header_bytes']
        delta=m['total_payload_bytes']-m['index']['metadata']['total_size']
        assert delta==rotation_payload and m['tensors']==m['index']['tensor_count'] and not m['changed']
        argv=['python3','-'] if host=='166' else ['ssh','-o','BatchMode=yes','root@172.16.10.167','python3','-']
        proc=subprocess.run(argv,input=SMALL,text=True,capture_output=True,timeout=60)
        out=dest/('metadata_'+host+'.raw');out.write_text(proc.stdout+'\nSTDERR\n'+proc.stderr)
        if proc.returncode:raise RuntimeError('small metadata probe exit '+str(proc.returncode))
        small=json.loads(proc.stdout)
        observations[host]=dict(structural_complete=True,coverage='all182 index-referenced files,177474 tensors;exact header/index membership,contiguous offsets,shape/dtype sizes,file length,unchanged stat',payload_bytes_read=0,cryptographic_payload_integrity='unknown',index_metadata_delta=delta,rotation_payload=rotation_payload,rotation_keys=small['rot_index_keys'],tokenizer_json_valid=small['tokenizer_json_valid'],tokenizer_vocab_entries=small['tokenizer_vocab_entries'],quant_description_types=small['quant_description_types'],metadata_hashes=small['metadata_hashes'],native_roots=small['native_roots'],uploader_terminal='unknown')
        b=out.read_bytes();evidence.append(dict(id='metadata'+host,path=str(out),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),locator='rot_index_keys/tokenizer/quant types/native_roots'))
    # Cross-host headers/size identities; this is not a payload checksum.
    a=json.loads((previous/'observations_166.raw').read_text().split('\nSTDERR\n')[0])['model']
    b=json.loads((previous/'observations_167.raw').read_text().split('\nSTDERR\n')[0])['model']
    fields=('file','bytes','header_sha256','header_bytes','tensors')
    assert [{k:s[k] for k in fields}for s in a['shards']]==[{k:s[k] for k in fields}for s in b['shards']]
    reduction=dict(at=datetime.now(timezone.utc).isoformat(),observations=observations,cross_host_header_and_size_identity=True,interpretation='Per-file structural completeness passes; metadata.total_size excludes exactly the rotation tensor payload arithmetically. Exporter intent is inference; no payload checksum or successful model load is claimed.',prior_job='GLM53-READINESS-20261006B',prior_job_unchanged=True)
    p=dest/'reduction.json';p.write_text(json.dumps(reduction,indent=2)+'\n');raw=p.read_bytes();evidence.append(dict(id='reduction',path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),locator='all observations and interpretation'))
    result=dict(schema_version=1,job_id=job['job_id'],status='completed',summary='All referenced tensor files pass structural checks on both hosts. JobB metadata-size warning equals rotation payload exactly; original JobB retained. Tokenizer/quant metadata parsed, native roots freshly read. No payload/model/service mutation.',execution=dict(inner_exit_code=0,acceptance='passed',processes=[]),findings=[dict(kind='fact',text='182 referenced files/177474 tensors complete by header/offset/shape/length;both host headers/sizes identical;metadata delta75497472 bytes equals rot payload.',scope=dict(hosts=['166','167'],tensor_payload_reads=0),evidence_ids=['headers166','headers167','reduction']),dict(kind='inference',text='Index metadata.total_size was computed without the exported rotation tensor; exporter source/intent not yet verified.',scope=dict(model='GLM-5.3 W8A8'),evidence_ids=['reduction'])],evidence=evidence,unknowns=['Uploader terminal confirmation','Cryptographic payload integrity','GLM-5.3 loading/fit/full PD/correctness/E2E'],decision_request=None,next_check_at=None)
    p=dest/'result.tmp';p.write_text(json.dumps(result,indent=2)+'\n');os.replace(p,dest/'result.json')


if __name__=='__main__':
    main(sys.argv[1])
