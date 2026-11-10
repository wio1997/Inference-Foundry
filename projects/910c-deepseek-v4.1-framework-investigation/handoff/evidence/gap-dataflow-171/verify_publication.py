from pathlib import Path
import base64,concurrent.futures,hashlib,json
import stage_publication as s
P=Path(__file__).resolve().parent
plan=json.loads((P/'publication-plan.json').read_text())
commit=json.loads((P/'published-commit.json').read_text())['sha']
expected=json.loads((P/'publication-expected.json').read_text())
folders=sorted({str(Path(r['path']).parent) for r in plan['entries']})
lookup={}
for folder in folders:
    for row in s.api('contents/'+folder+'?ref='+commit):
        if row['type']=='file':lookup[row['path']]=row['sha']
assert all(lookup.get(r['path'])==r['sha'] for r in plan['entries'])
def verify(row):
    b=base64.b64decode(s.api('git/blobs/'+row['sha'])['content']);e=expected[row['path']]
    assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],row['path']
    return row['path']
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(verify,plan['entries']))
head=s.api('git/ref/heads/'+s.BRANCH)['object']['sha']
included=head==commit or s.api('compare/'+commit+'...'+head)['status']=='ahead';assert included
r={'commit':commit,'parent':plan['head'],'head_at_readback':head,'readback_equal':True,'paths_match_committed_tree':True,
   'head_contains_commit':included,'files':paths,'new_npu_runs':0,'new_model_requests':0,'new_profiles':0,'service_actions':0}
(P/'publication.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in r.items() if k!='files'}|{'files_count':len(paths)}))
