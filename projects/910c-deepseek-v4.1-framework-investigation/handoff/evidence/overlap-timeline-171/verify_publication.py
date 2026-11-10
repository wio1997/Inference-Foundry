from pathlib import Path
import base64,concurrent.futures,hashlib,json
import stage_publication as s
P=Path(__file__).resolve().parent;plan=json.load(open(P/'publication-plan.json'));commit=json.load(open(P/'published-commit.json'))['sha'];expected=json.load(open(P/'publication-expected.json'))
tree=s.api('git/trees/'+commit+'?recursive=1');lookup={r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'}
assert all(lookup.get(r['path'])==r['sha'] for r in plan['entries'])
def verify(row):
 content=base64.b64decode(s.api('git/blobs/'+row['sha'])['content']);e=expected[row['path']]
 assert len(content)==e['bytes'] and hashlib.sha256(content).hexdigest()==e['sha256'],row['path']
 return row['path']
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(verify,plan['entries']))
head=s.api('git/ref/heads/'+s.BRANCH)['object']['sha'];included=head==commit or s.api('compare/'+commit+'...'+head)['status']=='ahead';assert included
r={'commit':commit,'parent':plan['head'],'head_at_readback':head,'readback_equal':True,'paths_match_committed_tree':True,'head_contains_commit':included,'files':paths,'new_npu_runs':0,'new_model_requests':0,'service_actions':0}
(P/'publication.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='files'}|{'files_count':len(paths)}))
