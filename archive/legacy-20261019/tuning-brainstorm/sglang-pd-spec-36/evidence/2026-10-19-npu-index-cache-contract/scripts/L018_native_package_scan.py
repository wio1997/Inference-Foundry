"""Bounded file/metadata scan only: never import torch_npu or execute an operator."""
import hashlib,importlib.metadata,json,pathlib,subprocess,sys
j=json.loads(sys.argv[1]);root=pathlib.Path('/sgl-workspace/sglang');head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==j['expected_commit']
v={'source_commit':head,'read_only':True,'generation_requests':0,'native_executed':False,'files':{},'distributions':{},'scan':{'text_files':0,'text_bytes':0,'skipped_large':[],'skipped_budget':[],'matching_paths':[]},'selection_basis':'Installed distribution RECORD and explicit optional CANN header directories. File candidates only; actual loaded runtime registry/library unknown.','package_git':'UNKNOWN'}
needles=[x.encode() for x in j['needles']];max_read=j['max_read_bytes'];max_file=j['max_file_bytes'];max_out=j['max_output_content_bytes'];out_bytes=0;seen=set()
for package in j['packages']:
 try:dist=importlib.metadata.distribution(package)
 except importlib.metadata.PackageNotFoundError:v['distributions'][package]={'status':'NOT_FOUND'};continue
 base=pathlib.Path(dist.locate_file('')).resolve();files=sorted(dist.files or [],key=str);v['distributions'][package]={'name':dist.metadata.get('Name'),'version':dist.version,'base':str(base),'record_entries':len(files)}
 for rel in files:
  name=str(rel);parts=pathlib.PurePosixPath(name).parts
  if not parts or parts[0]!=package or '..' in parts:continue
  p=pathlib.Path(dist.locate_file(rel)).resolve()
  if not p.is_relative_to(base/package) or not p.is_file() or p in seen:continue
  seen.add(p)
  if p.suffix.lower() not in j['text_suffixes']:continue
  size=p.stat().st_size
  if size>max_file:v['scan']['skipped_large'].append({'path':name,'bytes':size});continue
  if v['scan']['text_bytes']+size>max_read:v['scan']['skipped_budget'].append(name);continue
  b=p.read_bytes();v['scan']['text_files']+=1;v['scan']['text_bytes']+=len(b)
  if not any(n in b.lower() for n in needles):continue
  v['scan']['matching_paths'].append(name)
  item={'sha256':hashlib.sha256(b).hexdigest(),'path_candidate':str(p),'bytes':len(b),'git':'UNKNOWN','content_saved':False}
  if out_bytes+len(b)<=max_out:
   try:item['content']=b.decode('utf-8');item['content_saved']=True;out_bytes+=len(b)
   except UnicodeDecodeError:item['decode']='NON_UTF8'
  else:item['content_limit']='MAX_OUTPUT_CONTENT_BYTES'
  v['files'][name]=item
v['optional_cann_headers']={}
for dirname in j['optional_cann_header_dirs']:
 p=pathlib.Path(dirname);found=[]
 if p.is_dir():
  for q in sorted(p.glob('*.h')):
   if 'lightning' not in q.name.lower():continue
   size=q.stat().st_size
   if size>max_file:continue
   b=q.read_bytes();name='cann_headers/'+q.name;found.append({'path':str(q),'sha256':hashlib.sha256(b).hexdigest()})
   if name not in v['files'] and out_bytes+len(b)<=max_out:v['files'][name]={'path_candidate':str(q),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'git':'UNKNOWN','content_saved':True,'content':b.decode('utf-8')};out_bytes+=len(b)
 v['optional_cann_headers'][dirname]={'exists':p.is_dir(),'matching_headers':found}
v['native_packages_in_sys_modules']=[x for x in ['torch_npu','sgl_kernel_npu'] if x in sys.modules];v['scan']['output_content_bytes']=out_bytes;v['probe_status']='SOURCE_FOUND' if v['files'] else 'UNAVAILABLE_IN_BOUNDED_SCAN';print(json.dumps(v))
