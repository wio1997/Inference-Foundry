"""Render a relay-local pipe for accepted prefix files; no controller prompt copies."""
import ast, re, shlex

SENDER = r'''import socket,pathlib,json,hashlib,tarfile,io,gzip,sys,struct
assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T0615-formal-inputs-v3/artifacts')/('C'+str(CONCURRENCY))
raw=(root/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==MANIFEST_SHA
manifest=json.loads(raw);assert manifest['concurrency']==CONCURRENCY and len(manifest['prefixes'])==CONCURRENCY
names=['manifest.json']
for item in manifest['prefixes']:
 name='prefix-'+str(item['prefix_id']).zfill(2)+'.json';entry=json.loads((root/name).read_text());ids=entry['input_ids']
 assert len(ids)==74400 and hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()==entry['ids_sha256_le_u32']==item['ids_sha256_le_u32']
 names.append(name)
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as archive:
 for name in names:archive.add(root/name,arcname=name,recursive=False)
assert len(buffer.getvalue())<=128*1024**2
sys.stdout.buffer.write(gzip.compress(buffer.getvalue(),mtime=0));sys.stdout.buffer.flush()
'''
RECEIVER = r'''import socket,pathlib,json,hashlib,tarfile,io,gzip,sys,struct
assert socket.gethostname()=='B5-15U-Atlas900RCKA3-34'
raw=sys.stdin.buffer.read(64*1024**2+1);assert len(raw)<=64*1024**2
unpacked=gzip.decompress(raw);assert len(unpacked)<=128*1024**2
root=pathlib.Path(DESTINATION);assert not root.exists(),'No overwrite/replay'
expected={'manifest.json'}|{'prefix-'+str(i).zfill(2)+'.json' for i in range(CONCURRENCY)}
with tarfile.open(fileobj=io.BytesIO(unpacked),mode='r:') as archive:
 members=archive.getmembers();assert len(members)==CONCURRENCY+1 and {member.name for member in members}==expected
 assert all(member.isfile() and member.size<2*1024**2 for member in members)
 root.mkdir(parents=True)
 for member in members:
  value=archive.extractfile(member).read();assert len(value)==member.size
  (root/member.name).write_bytes(value)
raw_manifest=(root/'manifest.json').read_bytes();assert hashlib.sha256(raw_manifest).hexdigest()==MANIFEST_SHA
manifest=json.loads(raw_manifest);assert manifest['concurrency']==CONCURRENCY
for item in manifest['prefixes']:
 entry=json.loads((root/('prefix-'+str(item['prefix_id']).zfill(2)+'.json')).read_text());ids=entry['input_ids']
 assert len(ids)==74400 and hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()==entry['ids_sha256_le_u32']==item['ids_sha256_le_u32']
print(json.dumps({'project_id':'glm53','concurrency':CONCURRENCY,'destination':str(root),'compressed_bytes':len(raw),'tar_bytes':len(unpacked),'manifest_sha256':MANIFEST_SHA,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())}}))
'''

def render(concurrency, manifest_sha, run_id, source_pod, destination_pod):
    assert concurrency in [16,32,64] and re.fullmatch('[a-f0-9]{64}',manifest_sha)
    assert re.fullmatch('[A-Za-z0-9-]+',run_id)
    destination='/data/wio/inference-foundry/glm53/runs/20261017T1720-sglang-shared-pool-v2/formal-gates/'+run_id+'/data'
    values={'CONCURRENCY':concurrency,'MANIFEST_SHA':manifest_sha,'DESTINATION':destination}
    def replace(text):
        for key,value in values.items():text=re.sub(r'\b'+key+r'\b',lambda _:repr(value),text)
        ast.parse(text);return text
    def command(pod, code, stdin=False):
        return shlex.join(['kubectl','exec',*(['-i'] if stdin else []),'-n','default',pod,'-c','hostctl',
                           '--','nsenter','-t','1','-m','--','python3','-c',code])
    pipeline=command(source_pod,replace(SENDER))+' | '+command(destination_pod,replace(RECEIVER),True)
    return shlex.join(['bash','-o','pipefail','-c',pipeline]),destination
