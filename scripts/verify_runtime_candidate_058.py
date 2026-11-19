#!/usr/bin/env python3
"""Verify a local candidate artifact chain, NOT historical runtime imports.

Pass local source paths explicitly; output only fixed file names, SHA digests,
and booleans. Never extract archives or print candidate content/paths.
"""
import argparse,hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/runtime_candidate_058/source_binding_v3.json'
TAR_SHA='f98cfbd3e825a8f08431541fe342053a84639a84781b002253e070b9fad11a8a'
TREE_SHA='d3f04bbccce8d5ac7ed7ba6221bd3439f4dcd37bac1956f7e18048e95305c7af'
MANIFEST_SHA='d5f891be7693bd14e77edb58b137ba4788c9ed48c33ad691570c36dea63f9ec0'
CANDIDATE_SHA={'p':'f18c2a615949b33270ab39d220005eeb9d53bd7e1c13fd044a4a47594445e7a8','d':'bda88af9a5bc3e4b00329a92d409ed9afa9a0d102aa0e999c5d24aec0f7f4c1d'}
TARGETS=('sglang/srt/observability/req_time_stats.py','sglang/srt/disaggregation/decode.py','sglang/srt/disaggregation/common/conn.py','sglang/srt/disaggregation/ascend/conn.py','sglang/srt/disaggregation/mooncake/conn.py','sglang/srt/managers/scheduler_components/metrics_reporter.py')


def need(v,m):
 if not v:raise ValueError(m)


def read_regular(path,max_bytes):
 p=Path(path);need(p.is_file() and not p.is_symlink(),'input not regular');before=p.stat();need(before.st_size<=max_bytes,'input too large');raw=p.read_bytes();after=p.stat();need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'input changed');return raw


def sha(b):return hashlib.sha256(b).hexdigest()


def candidates(obj):
 out=[]
 def scan(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if k=='runtime_tar_sha256':out.append(v)
    scan(v)
  elif isinstance(x,list):
   for v in x:scan(v)
 scan(obj);return out


def summarize(args):
 tree_raw=read_regular(args.tree,5000000);need(sha(tree_raw)==TREE_SHA,'tree manifest mismatch');tree=json.loads(tree_raw);need(isinstance(tree,dict) and len(tree)==6336,'tree coverage');tar_raw=read_regular(args.tar,100000000);need(sha(tar_raw)==TAR_SHA,'candidate tar mismatch')
 records={}
 for line in (ROOT/'SHA256SUMS').read_text().splitlines():
  h,name=line.split('  ',1);need(name not in records,'duplicate archive checksum');records[name]=h
 launches={}
 for role in ('p','d'):
  candidate_raw=read_regular(getattr(args,'candidate_'+role),10000000);need(sha(candidate_raw)==CANDIDATE_SHA[role] and candidates(json.loads(candidate_raw))==[TAR_SHA],'candidate content mismatch');launch_path=f'evidence/matched_SG_G181/{role}/launch_manifest.json';launch_raw=read_regular(ROOT/launch_path,10000000);need(records.get(launch_path)==sha(launch_raw),'launch archive checksum mismatch');launch=json.loads(launch_raw);need(launch['candidate_sha256']==sha(candidate_raw) and launch['runtime_manifest_sha256']==MANIFEST_SHA,'launch record mismatch');launches[role]={'launch_manifest_sha256':sha(launch_raw),'candidate_sha256':sha(candidate_raw),'declared_tar_sha256':TAR_SHA,'recorded_manifest_sha256':launch['runtime_manifest_sha256']}
 members={};import io
 with tarfile.open(fileobj=io.BytesIO(tar_raw),mode='r:gz') as tf:
  infos=tf.getmembers()
  for suffix in ('integrated_g18_manifest.json',)+TARGETS:
   found=[m for m in infos if m.name==suffix or m.name.endswith('/'+suffix)];need(len(found)==1 and found[0].isfile() and found[0].size<5000000,'missing or ambiguous tar member');stream=tf.extractfile(found[0]);need(stream is not None,'tar read failed');raw=stream.read();need(len(raw)==found[0].size,'short tar member');members[suffix]=raw
 manifest=json.loads(members['integrated_g18_manifest.json']);need(sha(members['integrated_g18_manifest.json'])==MANIFEST_SHA and manifest['source_tree_manifest_sha256']==TREE_SHA and manifest['source_file_count']==6336,'manifest binding mismatch')
 files=[]
 for name in TARGETS:
  h=sha(members[name]);need(tree.get(name)==h,'target source differs');need(not any(k.endswith(name) or name.endswith(k) for k in manifest['output_sha256']),'target in output override list');files.append({'relative_source':name,'sha256':h,'matches_tree_and_tar':True,'not_in_integrated_override_list':True})
 return {'schema':'glm53-g181-candidate-source-binding-v1','source_tree_manifest_sha256':TREE_SHA,'source_tree_file_count':len(tree),'runtime_tar_sha256':TAR_SHA,'integrated_manifest_sha256':MANIFEST_SHA,'launch_candidate_records':launches,'target_files':files,'limitations':['Artifact chain binds launch records to local candidate bytes and source tar; it does not prove historical deployed files or actual Python imports','Read-only container rootfs does not establish read-only host bind source or rule out prior mutations/shadowed imports','Candidate source semantics remain conditional pending runtime evidence']}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tar',required=True);ap.add_argument('--tree',required=True);ap.add_argument('--candidate-p',required=True);ap.add_argument('--candidate-d',required=True);ap.add_argument('--write',action='store_true');ap.add_argument('--check',action='store_true');a=ap.parse_args();d=summarize(a);text=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
 if a.write:
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with OUT.open('x',encoding='utf8') as f:f.write(text)
 if a.check:need(OUT.read_text(encoding='utf8')==text,'result mismatch')
 print(json.dumps({'status':'pass','targets':len(d['target_files']),'sha256':sha(text.encode()),'wrote':a.write,'checked':a.check}))


if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,tarfile.TarError,json.JSONDecodeError):raise SystemExit('FAILED_CLOSED: candidate integrity or parsing validation failed')
