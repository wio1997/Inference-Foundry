"""AST symbol/callsite index only for source locations actually in profile stacks."""
import ast,collections,hashlib,json,re,time
from pathlib import Path
ROOT=Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis/joint169')
review=json.loads((ROOT/'startup-creators/REVIEW.json').read_text());pattern=re.compile(r'^(.*\.py)\((\d+)\): (.*)$');wanted=collections.defaultdict(dict)
for p in review['profiles']:
 for frame,count in p['source_frames'].items():
  match=pattern.match(frame)
  if not match:continue
  filename,line,symbol=match.groups();path=Path(filename)
  if not path.is_relative_to('/vllm-workspace'):continue
  wanted[filename][frame]=(int(line),symbol)
files=[];frames={}
for filename,entries in sorted(wanted.items()):
 p=Path(filename)
 if not p.exists() or not p.resolve().is_relative_to('/vllm-workspace'):
  files.append({'path':filename,'status':'SOURCE_NOT_AVAILABLE_WITHIN_ALLOWED_ROOT'});continue
 raw=p.read_bytes();tree=ast.parse(raw.decode());symbols=[];calls=[]
 def visit(node,scope):
  nextscope=scope
  if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
   nextscope=scope+[node.name];symbols.append((node.lineno,node.end_lineno,'.'.join(nextscope),type(node).__name__))
  if isinstance(node,ast.Call):calls.append((node.lineno,node.end_lineno,ast.unparse(node.func)[:240]))
  for child in ast.iter_child_nodes(node):visit(child,nextscope)
 visit(tree,[])
 for frame,(line,symbol) in entries.items():
  scopes=[r for r in symbols if r[0]<=line<=r[1]];scopes.sort(key=lambda r:(r[1]-r[0],-r[0]));on_line=[r for r in calls if r[0]<=line<=r[1]];on_line.sort(key=lambda r:(r[1]-r[0],r[0],r[2]));frames[frame]={'path':filename,'line':line,'reported_symbol':symbol,'enclosing_qualified_symbol':scopes[0][2] if scopes else None,'call_expressions_on_profile_line':[r[2] for r in on_line[:6]],'source_sha256':hashlib.sha256(raw).hexdigest()}
 files.append({'path':filename,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'observed_source_locations':len(entries),'status':'SYMBOL_ONLY_AST_INDEXED'})
result={'status':'PROFILE_DRIVEN_SYMBOL_ONLY_SOURCE_INDEX','generated_unix':time.time(),'files':files,'source_frame_index':frames,'scope':'Only file locations actually present in exactstartupCPU/native associations. ASTnames/callsites,currentimmutable-image source hashes;no semantic closure or per-formal-invocation identity inferred. No sourcebody dump or repositorywide scan.','overall_complete':False};(ROOT/'PROFILE-SOURCE-SYMBOL-INDEX.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'files':len(files),'frames':len(frames)}))
