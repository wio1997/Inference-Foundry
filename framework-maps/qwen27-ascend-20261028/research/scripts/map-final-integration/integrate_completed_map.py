"""Merge frozen complete supplements with terminal204 native evidence, in new output."""
from pathlib import Path
import json,sys,os,hashlib,importlib.util,time,shutil
D=Path(__file__).resolve().parent
R=Path('PRIVATE_PATH_695a4dffb8d9')
sys.path.insert(0,str(R/'executor-run-gen90/consumer-v2'));from common import sha,rec,atomic

def load(n):
 spec=importlib.util.spec_from_file_location(n,D/(n+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def build(source,inputs,out):
 source=Path(source);inputs=Path(inputs);out=Path(out);assert not out.exists()
 package=json.loads((inputs/'package-manifest.json').read_text())
 for e in package['manifest']:
  p=inputs/e['path'];assert p.stat().st_size==e['bytes']and sha(p)==e['sha256']
 cpath=source/'combined-map.json';old=json.loads(cpath.read_text());base=json.loads((inputs/'base/combined-map.json').read_text())
 # The verified source differs only by the exact204 native component and restored postfreeze keys.
 assert set(old)==set(base)
 for k in old:
  if k not in ['new_measurements','metadata']:assert old[k]==base[k],k
 for k in set(base['metadata'])|set(old['metadata']):
  if k not in ['status','preview_not_final','validity']:assert old['metadata'].get(k)==base['metadata'].get(k),k
 assert old['metadata']['status']=='REVIEWABLE_NATIVE_EVIDENCE_PENDING_ROOT_ACCEPTANCE'and old['metadata']['preview_not_final']is True
 assert old['metadata']['validity'].startswith(str(base['metadata'].get('validity','')))
 base['metadata']=old['metadata']
 on=old['new_measurements'];bn=base['new_measurements'];assert set(on)-set(bn)=={'native_sharded_map'}
 for k in bn:
  if k not in ['postfreeze_root_audit','cpu_lifecycle_map']:assert bn[k]==on[k],k
 assert {k:v for k,v in bn['cpu_lifecycle_map'].items()if k!='native_bundle'}=={k:v for k,v in on['cpu_lifecycle_map'].items()if k!='native_bundle'}
 assert bn['cpu_lifecycle_map']['native_bundle']['status']=='pending_input_not_supplied'
 assert on['cpu_lifecycle_map']['native_bundle']['identity_count']==204 and on['cpu_lifecycle_map']['native_bundle']['bundle_errors']==0
 bn['cpu_lifecycle_map']['native_bundle']=on['cpu_lifecycle_map']['native_bundle']
 op=on['postfreeze_root_audit'];bp=bn['postfreeze_root_audit'];assert all(bp[k]==v for k,v in op.items()),'existing_postfreeze_values_changed'
 missing=sorted(set(bp)-set(op));assert len(missing)==12
 native=on['native_sharded_map'];ix=native['index'];assert ix['identity_count']==len(ix['per_rank'])==204 and not ix['bundle_errors']
 assert len({r['rank_key']for r in ix['per_rank']})==204
 composition=json.loads((source/'composition-receipt.json').read_text());assert composition['verified_identity_count']==204 and composition['output_combined_map']['sha256']==sha(cpath)
 fileproof=json.loads((source/'native-file-build-receipt.json').read_text());assert fileproof['actual204_executed']and fileproof['native_sidecar_count']==204 and fileproof['html_sha256']==sha(source/'index.html') and fileproof['combined_map_sha256']==sha(cpath)
 bn['native_sharded_map']=native;assert bn['native_sharded_map']==on['native_sharded_map']
 changed={'combined-map.json','index.html','index-HTTP.html','composition-receipt.json','portable-build-receipt.json','native-file-build-receipt.json'}
 out.mkdir(parents=True);reused=[]
 for p in source.rglob('*'):
  assert not p.is_symlink(),'symlink_refused'
  if not p.is_file()or str(p.relative_to(source))in changed:continue
  q=out/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True);before=p.stat();os.link(p,q);after=p.stat();qs=q.stat();assert(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)==(qs.st_dev,qs.st_ino,qs.st_size,qs.st_mtime_ns)
  reused.append(dict(path=str(p.relative_to(source)),device=qs.st_dev,inode=qs.st_ino,bytes=qs.st_size,mtime_ns=qs.st_mtime_ns))
 # Exact frozen attachments, collision allowed only for byte-identical content.
 for e in package['manifest']:
  if not e['path'].startswith('base/')or e['path']=='base/combined-map.json':continue
  rel=Path(e['path']).relative_to('base');p=inputs/e['path'];q=out/rel;q.parent.mkdir(parents=True,exist_ok=True)
  if q.exists():assert sha(q)==e['sha256'],('attachment_collision',str(rel))
  else:os.link(p,q)
 atomic(out/'combined-map.json',base)
 template=(inputs/'template.html').read_text();transport=load('build_native_file_map');assert template.count(transport.OLD)==1;template=template.replace(transport.OLD,transport.NEW,1);tp=out/'template.html';assert not tp.exists();tp.write_text(template)
 renderer=load('renderer');(out/'index-HTTP.html').write_text(renderer.render(base,tp));portable=load('build_portable_map');portable.build(out/'index-HTTP.html',out/'index.html')
 # Reuse existing native verification verbatim and bind the changed map separately.
 atomic(out/'native-composition-source-receipt.json',composition)
 proof=dict(status='actual204_complete_supplements_integrated_pending_browser_QA',created_ns=time.time_ns(),source_combined_map=rec(cpath),input_package_manifest=rec(inputs/'package-manifest.json'),source_native_composition=rec(source/'composition-receipt.json'),source_native_file_build=rec(source/'native-file-build-receipt.json'),restored_postfreeze_keys=missing,baseline_and_protected_arrays_exact_equal=True,native_status_and_metadata_preserved_from_verified_composition=True,native_index_exact_equal=True,source_native_sidecars_preserved=204,immutable_hardlinks=reused,combined_map=rec(out/'combined-map.json'),html=rec(out/'index.html'),actual204_executed=True,overall_complete=False)
 atomic(out/'integration-receipt.json',proof)
 composition.update(status='native_verification_reused_complete_supplements_integrated_pending_root_review',composition_mode='reuse_exact_native_index_and_immutable_files_with_frozen_supplement_merge',source_composition=rec(source/'composition-receipt.json'),integration_receipt=rec(out/'integration-receipt.json'),output_combined_map=rec(out/'combined-map.json'))
 atomic(out/'composition-receipt.json',composition)
 fileproof.update(status='actual204_file_native_transport_reused_supplements_integrated_pending_browser_QA',source_build_receipt=rec(source/'native-file-build-receipt.json'),source=str(source),output=str(out),html_sha256=sha(out/'index.html'),combined_map_sha256=sha(out/'combined-map.json'),sidecar_reuse='Exact original sidecars linked; no regenerated/native evidence changes',sidecars=[dict(e,path=e['path'])for e in fileproof['sidecars']])
 atomic(out/'native-file-build-receipt.json',fileproof)
 return {k:v for k,v in proof.items()if k!='immutable_hardlinks'}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.source,a.inputs,a.output)))
