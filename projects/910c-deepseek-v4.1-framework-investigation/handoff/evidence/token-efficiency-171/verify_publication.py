from pathlib import Path
import concurrent.futures,hashlib,json
import stage_publication as s
P=Path(__file__).resolve().parent;plan=json.loads((P/'publication-plan.json').read_text());expected=json.loads((P/'publication-expected.json').read_text());commit=json.loads((P/'published-commit.json').read_text())['sha']
lookup={}
for folder in sorted({str(Path(r['path']).parent) for r in plan['entries']}):
    lookup.update({r['path']:r['sha'] for r in s.api('contents/'+folder+'?ref='+commit) if r['type']=='file'})
assert all(lookup.get(r['path'])==r['sha'] for r in plan['entries'])
def verify(row):
    b=s.raw(row['path'],commit);e=expected[row['path']]
    assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],row['path']
    assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['sha']
    return row['path']
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(verify,plan['entries']))
head=s.api('git/ref/heads/'+s.BRANCH)['object']['sha'];included=head==commit or s.api('compare/'+commit+'...'+head)['status']=='ahead';assert included
r={'commit':commit,'parent':plan['head'],'head_at_readback':head,'paths_match_committed_tree':True,'readback_equal':True,'sha256_and_git_blob_sha_match':True,'head_contains_commit':included,'files':paths,'new_profiles':0,'new_model_requests':0,'service_actions':0}
(P/'publication.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='files'}|{'file_count':len(paths)}))
