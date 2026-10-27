"""Archive the terminal actual204 map and verify every extracted member by SHA."""
from pathlib import Path
import gzip,hashlib,json,os,sys,tarfile,time

D=Path(__file__).resolve().parent
R=Path('PRIVATE_PATH_695a4dffb8d9')
sys.path.insert(0,str(R/'executor-run-gen90/consumer-v2'))
from common import sha,rec,atomic
S=R/'executor-run-gen103/final-integration/map'
manifest_path=D/'distribution-manifest.json';archive=D/'q27-actual204-complete-map.tar.gz'
assert not archive.exists() and not manifest_path.exists()
def version(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
composition=json.loads((S/'composition-receipt.json').read_text())
assert composition['verified_identity_count']==204 and not composition['bundle_errors']
integration=json.loads((S/'integration-receipt.json').read_text())
assert integration['actual204_executed'] and len(integration['restored_postfreeze_keys'])==12
assert integration['combined_map']==rec(S/'combined-map.json') and integration['html']==rec(S/'index.html')
entries=[]
for p in sorted(S.rglob('*')):
 assert not p.is_symlink(),str(p)
 if not p.is_file():continue
 v=version(p);h=sha(p);assert version(p)==v
 entries.append(dict(path=str(p.relative_to(S)),bytes=v[2],sha256=h,version=v))
assert sum(e['path'].startswith('native-rank-details/') and e['path'].endswith('.json.gz.js')for e in entries)==204
assert sum(e['path'].startswith('native-rank-details/') and e['path'].endswith('.json.gz')for e in entries)>=204
manifest=dict(task_id='Q27-FULL-MAP-20261025',generation=104,created_ns=time.time_ns(),source=str(S),files=entries,source_composition=rec(S/'composition-receipt.json'),source_integration=rec(S/'integration-receipt.json'),actual204_executed=True,browser_QA='separate_required_gate',overall_complete=False)
atomic(manifest_path,manifest)
with archive.open('xb')as raw:
 with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=1)as gz:
  with tarfile.open(fileobj=gz,mode='w|')as tar:
   for e in entries:
    p=S/e['path'];assert version(p)==e['version'];info=tar.gettarinfo(str(p),arcname='q27-map/'+e['path']);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
    with p.open('rb')as inp:tar.addfile(info,inp)
    assert version(p)==e['version']
   info=tar.gettarinfo(str(manifest_path),arcname='q27-map/distribution-manifest.json');info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
   with manifest_path.open('rb')as inp:tar.addfile(info,inp)
# Full decompression / every member SHA without creating a second full map tree.
expected={'q27-map/'+e['path']:e for e in entries}
expected['q27-map/distribution-manifest.json']=rec(manifest_path)
seen=set()
with tarfile.open(archive,'r|gz')as tar:
 for m in tar:
  assert m.isfile() and m.name in expected and m.name not in seen and not Path(m.name).is_absolute() and '..'not in Path(m.name).parts
  h=hashlib.sha256();size=0
  with tar.extractfile(m)as inp:
   for b in iter(lambda:inp.read(1<<20),b''):h.update(b);size+=len(b)
  e=expected[m.name];assert size==m.size==e['bytes'] and h.hexdigest()==e['sha256'],m.name;seen.add(m.name)
assert seen==set(expected)
for e in entries:assert version(S/e['path'])==e['version']
receipt=dict(status='complete_actual204_distribution_packaged_full_member_SHA_verified_pending_browser_QA_root_review',created_ns=time.time_ns(),archive=rec(archive),manifest=rec(manifest_path),file_count=len(entries),source_bytes=sum(e['bytes']for e in entries),archive_members_verified=len(seen),native_sidecars=204,all_member_SHAs_verified=True,actual204_executed=True,browser_QA='separate_required_gate',overall_complete=False)
atomic(D/'distribution-receipt.json',receipt);print(json.dumps(receipt))
