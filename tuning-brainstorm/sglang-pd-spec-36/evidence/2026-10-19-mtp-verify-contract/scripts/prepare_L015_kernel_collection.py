#!/usr/bin/env python3
"""Prepare read-only package-file discovery, without importing NPU modules."""
import ast,hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1]
INNER=r'''
import hashlib,importlib.metadata,json,pathlib,subprocess,sys
j=json.loads(sys.argv[1]);sgl=pathlib.Path('/sgl-workspace/sglang');head=subprocess.check_output(['git','-C',str(sgl),'rev-parse','HEAD'],text=True).strip();assert head==j['expected_commit']
roots=[]
for entry in sys.path:
 p=pathlib.Path(entry or '.').resolve()
 if p not in roots:roots.append(p)
v={'source_commit':head,'files':{},'read_only':True,'generation_requests':0,'selection_basis':'sys.path file scan without importing package; candidate file identity only, not actual loaded native operator','distribution_versions':{}}
for dist in importlib.metadata.distributions():
 name=dist.metadata.get('Name','')
 if name.lower().replace('-','_')=='sgl_kernel_npu':v['distribution_versions'][name]=dist.version
for rel in j['paths']:
 assert rel.startswith('sgl_kernel_npu/') and '..' not in pathlib.PurePosixPath(rel).parts and rel.endswith('.py')
 candidates=[]
 for root in roots:
  p=root/rel
  if p.is_file() and p.resolve() not in candidates:candidates.append(p.resolve())
 if not candidates:continue
 p=candidates[0];b=p.read_bytes();git={};q=subprocess.run(['git','-C',str(p.parent),'rev-parse','--show-toplevel'],capture_output=True,text=True,timeout=15)
 if q.returncode==0:
  repo=pathlib.Path(q.stdout.strip());revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();relative=str(p.relative_to(repo));show=subprocess.run(['git','-C',str(repo),'show',revision+':'+relative],capture_output=True,timeout=15);git={'root':str(repo),'commit':revision,'relative_path':relative,'matches_git_head':show.returncode==0 and show.stdout==b}
 v['files'][rel]={'sha256':hashlib.sha256(b).hexdigest(),'path_candidate':str(p),'all_path_candidates':[str(x) for x in candidates],'git':git,'content':b.decode()}
v['probe_status']='SOURCE_FOUND' if 'sgl_kernel_npu/sample/verify_tree_greedy.py' in v['files'] else 'UNAVAILABLE_IN_PYTHON_PATH_SCAN';v['missing_files']=sorted(set(j['paths'])-v['files'].keys());print(json.dumps(v))
'''
base=R/'scripts/collect_mtp_contract.py';s=base.read_text();node=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='INNER');lines=s.splitlines(keepends=True);s=''.join(lines[:node.lineno-1])+"INNER=r'''\n"+INNER+"\n'''\n"+''.join(lines[node.end_lineno:]);s=s.replace('L015-R001-mtp-sources','L015-R002-kernel-source').replace('specs/L015-source-request.json','specs/L015-kernel-request.json').replace("R/'sources/installed'/n","R/'sources/installed-kernel-npu'/n");ast.parse(s);controller=R/'scripts/collect_mtp_kernel_sources.py';controller.open('x').write(s)
request={'expected_commit':'ce0a8fea5f4245839d174b2b8c55692be1810a7f','expected_container_id':'f42cd0a62f5f0778c0e6576ff8177f3d8c71f0e77c03719e22aa9af7b2e7a9c8','paths':['sgl_kernel_npu/sample/verify_tree_greedy.py','sgl_kernel_npu/sample/__init__.py','sgl_kernel_npu/__init__.py'],'scope':'Read-only file discovery and source identity; no package import, native launch or model request.'};(R/'specs/L015-kernel-request.json').open('x').write(json.dumps(request,indent=2)+'\n');inputs=['scripts/collect_mtp_kernel_sources.py','specs/L015-kernel-request.json','scripts/execute_frozen.py'];spec={'command':['python3','scripts/collect_mtp_kernel_sources.py'],'receipt_dir':'runs/L015-R002-frozen-controller','timeout_seconds':160,'inputs':{f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in inputs},'scope':request['scope']};(R/'specs/L015-R002.json').open('x').write(json.dumps(spec,indent=2)+'\n');print(json.dumps({'collector_sha256':hashlib.sha256(controller.read_bytes()).hexdigest(),'frozen_inputs':len(inputs)}))
