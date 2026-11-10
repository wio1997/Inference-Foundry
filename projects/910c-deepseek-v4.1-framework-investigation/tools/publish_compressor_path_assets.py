#!/usr/bin/env python3
"""Prepare exact blobs and verify content; ref movement uses the connector."""
from pathlib import Path
from hashlib import sha1,sha256
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from urllib.parse import quote
import base64,json,re,subprocess,sys
from update_compressor_path_maps import R,REPORT,TITLE,BODY,NEXT,PATTERN,banner,frontier
PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
PAYLOAD=Path('/tmp/compressor-path-publication-payload.json')
def blob(b):return sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def prepare():
    base=json.loads(Path('/tmp/compressor-path-public-base-checks.json').read_text());prior=json.loads(Path('/tmp/aux-context-final-publication-payload.json').read_text());receipt=json.loads((R/'aux-context-publication.json').read_text());assert base['base_commit']==receipt['commit']=='5f015695873448533b8e382a40851a70363b89f1' and receipt['all_content_equal']
    old={f['path']:f for f in prior['files']};files=[];checks=[]
    def add(path,s,origin):
        b=s.encode();files.append(dict(path=PREFIX+path,content=s,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),local_source=origin))
    for n in ['compressor-path-dependencies.json','compressor-path-parent-primary.json','compressor-path-controls.json','compressor-path-price.json','compressor-path-source-excerpts.json','compressor-path-edge-dispositions.json','compressor-path-review.json','compressor-path-integrity.json','compressor-path-frontier.json','compressor-path-recursive-intersections.json',REPORT]:add('handoff/evidence/'+n,(R/n).read_text(),str(R/n))
    for n in ['read_compressor_path_sources','probe_compressor_path_controls','analyze_compressor_path_price','review_compressor_path','update_compressor_path_maps','publish_compressor_path_assets']:add('tools/'+n+'.py',Path('tools',n+'.py').read_text(),'tools/'+n+'.py')
    for p in sorted((R/'compressor-path-source').rglob('*.py')):add('handoff/evidence/'+str(p.relative_to(R)),p.read_text(),str(p))
    for f in base['files']:
        name=f['path'];key=PREFIX+name
        if key in old:
            item=old[key];b=item['content'].encode();assert blob(b)==item['git_blob_sha1'] and sha256(b).hexdigest()==item['sha256']
        else:
            u='https://raw.githubusercontent.com/wio1997/Inference-Foundry/'+base['base_commit']+'/'+quote(key,safe='/')
            with urlopen(Request(u,headers={'User-Agent':'Inference-Foundry-source-audit'}),timeout=45) as response:b=response.read()
        assert blob(b)==f['sha'];s=b.decode();checks.append(dict(path=name,sha=f['sha'],bytes=len(b),sha256=sha256(b).hexdigest(),full_pinned_base_equal=True))
        if name.endswith('.html'):
            before=s;s,n=re.subn(PATTERN,lambda _:banner('../'+REPORT),s,count=1,flags=re.S);assert n==1 and re.sub(PATTERN,'',before,flags=re.S)==re.sub(PATTERN,'',s,flags=re.S)
        elif name.endswith('.md'):
            body=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY;i=s.index('\n')+1;assert '## 当前断点 · '+TITLE not in s;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+body+'\n'+s[i:]
            if name=='handoff/README.md':s=re.sub(r'^当前源码覆盖：.*$',lambda _:'当前源码覆盖：16当前身份、30具名body/range、11条receiver路由（原工作表15边交集）、15原Python/有限控制。C1/C2投影→完整ring与K/scale、长KV→QLI/SMLA读者；96完整源交付定价，同预算独立小方案停止，较大输入/query/SWA/QLI/SMLA保留。原122/34未变，每替换原语逐位一致，native新增0，全链路未闭合。',s,count=1,flags=re.M)
        else:
            d=json.loads(s);d.update(status=frontier()['status'],compressor_path_review=frontier(),next_action=NEXT,whole_chain_complete=False,completeness_established=False,new_framework_replacement_admissions=0)
            if name=='progress.json':d.update(active_loop='loop-047',overall_complete=False,latest_review='Run202744Z Compressor complete state/dualcache source/value bounded PASSNA; Run210932Z Engram embedding/UVA delivery PLANNED',previous_goal_turn_classification='PROGRESS:16 current source pins,30 body/range walks,11 selected receiver routes/15 old edge intersections,15 Python/finite controls.96 complete mirrored source deliveries; exactSHA old pool/ring native history reused. Whole source native0.709665%window, retainedGEMM paid scenario~0.189%window; originalC8 complete-source standalone native maximum under old1%budget. Stop same small children, retain larger input/query/SWA/QLI/SMLA and original candidates. Unknown request credit0, no native/fullchain admission.')
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Fresh connector blob SHA + pinned full base '+base['base_commit'])
    name='handoff/evidence/compressor-path-publication-manifest.json';manifest=dict(base_commit=base['base_commit'],base_tree_sha=receipt['tree'],paths=[x['path'] for x in files]+[PREFIX+name],base_checks=checks,HTML_only_own_aside=True,current_source_pins=16,source_walks=30,receiver_routes=11,matched_original_edges=15,controls=15,old_bound_inputs=122,old_unbound_inputs=34,new_native_admissions=0,every_replacement_bitwise=True,whole_chain_complete=False)
    p=R/'compressor-path-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(name,p.read_text(),str(p));assert len(files)==len({x['path'] for x in files});PAYLOAD.write_text(json.dumps(dict(base_commit=base['base_commit'],base_tree_sha=receipt['tree'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),max_bytes=max(x['bytes'] for x in files))))

def stage():
    d=json.loads(PAYLOAD.read_text())
    def one(f):
        b=f['content'].encode();assert sha256(b).hexdigest()==f['sha256'] and blob(b)==f['git_blob_sha1'];p=subprocess.run(['gh','api','repos/wio1997/Inference-Foundry/git/blobs','-X','POST','--input','-'],input=json.dumps(dict(content=base64.b64encode(b).decode(),encoding='base64')).encode(),capture_output=True,timeout=90)
        if p.returncode:raise RuntimeError('Blob stage failed: '+f['path'])
        h=json.loads(p.stdout)['sha'];assert h==f['git_blob_sha1'];return dict(path=f['path'],mode='100644',type='blob',sha=h)
    with ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(one,d['files']))
    (R/'compressor-path-staged-blobs.json').write_text(json.dumps(dict(base_commit=d['base_commit'],base_tree_sha=d['base_tree_sha'],entries=entries,payload_SHA256=sha256(PAYLOAD.read_bytes()).hexdigest(),branch_mutation=False),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(files=len(entries),all_blob_SHAs_match=True,branch_mutation=False)))
def verify(commit):
    assert re.fullmatch('[0-9a-f]{40}',commit);d=json.loads(PAYLOAD.read_text())
    def one(f):
        u='https://raw.githubusercontent.com/wio1997/Inference-Foundry/'+commit+'/'+quote(f['path'],safe='/')
        with urlopen(Request(u,headers={'User-Agent':'Inference-Foundry-source-audit'}),timeout=45) as response:b=response.read()
        assert b==f['content'].encode() and sha256(b).hexdigest()==f['sha256'] and blob(b)==f['git_blob_sha1'];return dict(path=f['path'],bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),content_equal=True)
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(one,d['files']))
    (R/'compressor-path-publication.json').write_text(json.dumps(dict(commit=commit,base_commit=d['base_commit'],files=checks,all_content_equal=True,payload_SHA256=sha256(PAYLOAD.read_bytes()).hexdigest(),qualification='Pinned full raw HTTPS bytes/SHA256/gitblob including default HTML; branch lease/compare/head checks recorded separately.'),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(commit=commit,files=len(checks),all_content_equal=True)))
if __name__=='__main__':{'prepare':prepare,'stage':stage,'verify':lambda:verify(sys.argv[2])}[sys.argv[1]]()
