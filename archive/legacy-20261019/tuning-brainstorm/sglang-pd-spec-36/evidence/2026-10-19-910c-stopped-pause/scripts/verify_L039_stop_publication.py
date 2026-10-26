"""Verify immutable uploaded tree, commit parent, branch head and exact changed files."""
import concurrent.futures, datetime, hashlib, json, pathlib, subprocess, sys, time
R=pathlib.Path(__file__).resolve().parents[1];P=(R/sys.argv[1]).resolve();M=R.parent/'repo-export';repo='wio1997/Inference-Foundry';target=sys.argv[2]
base=json.loads((P/'base.json').read_text());verified=json.loads((P/'tree-verification.json').read_text());manifest=json.loads((P/'manifest.json').read_text());tree=verified['tree']
assert verified['parent']==base['head'] and verified['verified_blobs']==len(manifest) and not verified['truncated']
for x in manifest:
    data=(M/x['path']).read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==x['git_blob_sha1']
def api(path):
    for attempt in range(2):
        p=subprocess.run(['gh','api','repos/'+repo+'/'+path],capture_output=True,text=True,timeout=90)
        if p.returncode==0:return json.loads(p.stdout)
        if attempt==0:time.sleep(1)
    raise RuntimeError(p.stderr[:1500])
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    head,commit,diff=list(pool.map(api,['git/ref/heads/'+base['branch'],'git/commits/'+target,'compare/'+base['head']+'...'+target]))
assert head['object']['sha']==target and [x['sha'] for x in commit['parents']]==[base['head']] and commit['tree']['sha']==tree
assert diff['status']=='ahead' and diff['ahead_by']==1 and {x['filename'] for x in diff['files']}==set(json.loads((P/'changed-files.json').read_text()))
v={'repository':repo,'branch':base['branch'],'head':target,'parent':base['head'],'tree':tree,'verified_blobs':len(manifest),'force':False,'expected_sha_lease':base['head'],'commit_url':'https://github.com/'+repo+'/commit/'+target,'evidence_url':'https://github.com/'+repo+'/tree/'+target+'/tuning-brainstorm/sglang-pd-spec-36/evidence/2026-10-19-910c-stopped-pause','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_changed_files_exact':True}
q=P/'published.json';assert not q.exists();q.write_text(json.dumps(v,indent=2)+'\n')
for p in [R/'tasks/sglang-pd-spec-36/index.json',R/'notes/L039-handoff.json']:
    x=json.loads(p.read_text());x['last_publication' if p.name=='index.json' else 'publication']=v;p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(v))
