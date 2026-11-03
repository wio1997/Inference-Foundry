"""Index class/function identities for new actual profile frames only."""
import ast
import collections
import hashlib
import json
import re
from pathlib import Path

O=Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis/joint172')
frontier=json.loads((O/'SOURCE-FRAME-FRONTIER.json').read_text())
old=json.loads((O.parent/'joint169/PROFILE-SOURCE-SYMBOL-INDEX.json').read_text())['source_frame_index']
by_file=collections.defaultdict(list)
for frame in frontier['unknown_frames']:
    m=re.fullmatch(r'(.+\.py)\((\d+)\): (.+)',frame)
    if m:by_file[m[1]].append((frame,int(m[2]),m[3]))
result={};files=[];unresolved=[]
for name,frames in sorted(by_file.items()):
    p=Path(name)
    if not name.startswith('/vllm-workspace/') or not p.is_file():
        unresolved.extend(f for f,_,_ in frames);continue
    b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();tree=ast.parse(b);ranges=[];calls=[]
    def visit(node,parent=''):
        is_scope=isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))
        qualified=parent+'.'+node.name if parent and is_scope else node.name if is_scope else parent
        if is_scope:ranges.append({'qualified':qualified,'kind':type(node).__name__,'first_line':node.lineno,'last_line':node.end_lineno})
        if isinstance(node,ast.Call):calls.append(node)
        for child in ast.iter_child_nodes(node):visit(child,qualified)
    visit(tree)
    oldhashes=sorted({v['source_sha256'] for v in old.values() if v['path']==name})
    files.append({'path':name,'sha256':sha,'bytes':len(b),'new_requested_frames':len(frames),'old_frame_hashes':oldhashes,'matches_prior_index_hash':not oldhashes or oldhashes==[sha],'symbol_ranges':ranges})
    for frame,line,reported in frames:
        scopes=[x for x in ranges if x['kind']!='ClassDef' and x['first_line']<=line<=x['last_line']]
        scope=min(scopes,key=lambda x:x['last_line']-x['first_line']) if scopes else None
        expr=sorted({ast.unparse(x.func)[:1000] for x in calls if x.lineno<=line<=x.end_lineno})
        result[frame]={'path':name,'line':line,'reported_symbol':reported,'enclosing_qualified_symbol':scope['qualified'] if scope else None,'reported_name_matches_enclosing':bool(scope and scope['qualified'].split('.')[-1]==reported),'call_expressions_on_profile_line':expr,'source_sha256':sha}
all_index=dict(old);all_index.update(result)
r={'status':'PROFILE_FRAME_SYMBOL_DELTA_RETURNED','new_files_indexed':len(files),'new_frames_indexed':len(result),'unresolved':unresolved,'files':files,'new_frame_index':result,'all_frame_index':all_index,'scope':'Onlynewprofilelocations;ASTclasses/functions/callnames,notframeworkbodyreview. Fullsymbolrangespersistedtoavoidre-readingthesefilesforfuturelinequeries.','overall_complete':False}
(O/'PROFILE-FRAME-SYMBOL-DELTA.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['files','new_frame_index','all_frame_index']}))
