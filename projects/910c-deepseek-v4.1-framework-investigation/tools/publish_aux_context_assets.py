#!/usr/bin/env python3
"""Prepare exact blobs and verify content; ref movement uses the connector."""
from pathlib import Path
from hashlib import sha1,sha256
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from urllib.parse import quote
import base64,json,re,subprocess,sys
from update_aux_context_maps import R,REPORT,TITLE,BODY,NEXT,PATTERN,banner,frontier
PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
PAYLOAD=Path('/tmp/aux-context-publication-payload.json')
def blob(b):return sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def prepare():
    base=json.loads(Path('/tmp/aux-context-public-base-checks.json').read_text());prior=json.loads(Path('/tmp/target-output-publication-payload.json').read_text());receipt=json.loads((R/'target-output-publication.json').read_text());assert base['base_commit']==receipt['commit']=='4bac54dcf9e20084bc81d89ed7ff32a48277fa72' and receipt['all_content_equal']
    old={f['path']:f for f in prior['files']};files=[];checks=[]
    def add(path,s,origin):
        b=s.encode();files.append(dict(path=PREFIX+path,content=s,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),local_source=origin))
    for n in ['aux-context-dependencies.json','aux-context-registry-dependencies.json','aux-context-parent-primary.json','aux-context-controls.json','aux-context-price.json','aux-context-source-excerpts.json','aux-context-edge-dispositions.json','aux-context-review.json','aux-context-integrity.json','aux-context-frontier.json',REPORT]:add('handoff/evidence/'+n,(R/n).read_text(),str(R/n))
    for n in ['read_aux_context_sources','read_aux_context_registry','probe_aux_context_layout','analyze_aux_context_price','review_aux_context','update_aux_context_maps','publish_aux_context_assets']:add('tools/'+n+'.py',Path('tools',n+'.py').read_text(),'tools/'+n+'.py')
    for p in sorted((R/'aux-context-source').rglob('*.py')):add('handoff/evidence/'+str(p.relative_to(R)),p.read_text(),str(p))
    for f in base['files']:
        name=f['path'];item=old[PREFIX+name];b=item['content'].encode();assert blob(b)==f['sha']==item['git_blob_sha1'] and sha256(b).hexdigest()==item['sha256'];s=b.decode();checks.append(dict(path=name,sha=f['sha'],bytes=len(b),sha256=sha256(b).hexdigest()))
        if name.endswith('.html'):
            before=s;s,n=re.subn(PATTERN,lambda _:banner('../'+REPORT),s,count=1,flags=re.S);assert n==1 and re.sub(PATTERN,'',before,flags=re.S)==re.sub(PATTERN,'',s,flags=re.S)
        elif name.endswith('.md'):
            body=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY;i=s.index('\n')+1;assert '## 当前断点 · '+TITLE not in s;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+body+'\n'+s[i:]
            if name=='handoff/README.md':s=re.sub(r'^当前源码覆盖：.*$',lambda _:'当前源码覆盖：19当前身份+3旧精确SHA回调、39具名body/range、24条真实receiver派发（匹配原工作表17边）、15原Python/有限布局控制。三aux→A3 FLOAT main→三context cache→各层SMLA实读；共享准备/打包按完整旧分支和新增成本判断，小family沿原预算停止，大host/context保留。原122/34未变，每替换原语逐位一致，新增准入0，全链路未闭合。',s,count=1,flags=re.M)
        else:
            d=json.loads(s);d.update(status=frontier()['status'],aux_context_review=frontier(),next_action=NEXT,whole_chain_complete=False,completeness_established=False,new_framework_replacement_admissions=0)
            if name=='progress.json':d.update(active_loop='loop-047',overall_complete=False,latest_review='Run194757Z actual aux/context source receiver/price bounded PASSNA; Run202744Z Compressor input/publication consumers PLANNED',previous_goal_turn_classification='PROGRESS:19 current identities +3 exactSHA prior callbacks,39 named source walks,24 receiver routes/17 original edge intersections,15 original Python/finite layout controls;24 complete aux/main/threecache chains72panels. A3 main quantNone, repeated full RoPE proxy and physical cache view/coordinate/reader obligations traced. Whole matched native branch0.893095%window, paid combined copy/preparation sharing~0.23%, request credit0; exact stopped tiny/changedM family retained. Larger host/context and all original goals remain, no native or fullchain admission.')
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Fresh base connector blob SHA + previously full-verified public base '+base['base_commit'])
    name='handoff/evidence/aux-context-publication-manifest.json';manifest=dict(base_commit=base['base_commit'],base_tree_sha=receipt['tree'],paths=[x['path'] for x in files]+[PREFIX+name],base_checks=checks,HTML_only_own_aside=True,current_source_pins=19,prior_exact_SHA_callbacks=3,source_walks=39,receiver_routes=24,matched_original_edges=17,controls=15,old_bound_inputs=122,old_unbound_inputs=34,new_native_admissions=0,every_replacement_bitwise=True,whole_chain_complete=False)
    p=R/'aux-context-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(name,p.read_text(),str(p));assert len(files)==len({x['path'] for x in files});PAYLOAD.write_text(json.dumps(dict(base_commit=base['base_commit'],base_tree_sha=receipt['tree'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),max_bytes=max(x['bytes'] for x in files))))
def stage():
    d=json.loads(PAYLOAD.read_text())
    def one(f):
        b=f['content'].encode();assert sha256(b).hexdigest()==f['sha256'] and blob(b)==f['git_blob_sha1'];p=subprocess.run(['gh','api','repos/wio1997/Inference-Foundry/git/blobs','-X','POST','--input','-'],input=json.dumps(dict(content=base64.b64encode(b).decode(),encoding='base64')).encode(),capture_output=True,timeout=90)
        if p.returncode:raise RuntimeError('Blob stage failed: '+f['path'])
        h=json.loads(p.stdout)['sha'];assert h==f['git_blob_sha1'];return dict(path=f['path'],mode='100644',type='blob',sha=h)
    with ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(one,d['files']))
    (R/'aux-context-staged-blobs.json').write_text(json.dumps(dict(base_commit=d['base_commit'],base_tree_sha=d['base_tree_sha'],entries=entries,payload_SHA256=sha256(PAYLOAD.read_bytes()).hexdigest(),branch_mutation=False),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(files=len(entries),all_blob_SHAs_match=True,branch_mutation=False)))
def verify(commit):
    assert re.fullmatch('[0-9a-f]{40}',commit);d=json.loads(PAYLOAD.read_text())
    def one(f):
        u='https://raw.githubusercontent.com/wio1997/Inference-Foundry/'+commit+'/'+quote(f['path'],safe='/')
        with urlopen(Request(u,headers={'User-Agent':'Inference-Foundry-source-audit'}),timeout=45) as response:b=response.read()
        assert b==f['content'].encode() and sha256(b).hexdigest()==f['sha256'] and blob(b)==f['git_blob_sha1'];return dict(path=f['path'],bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),content_equal=True)
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(one,d['files']))
    (R/'aux-context-publication.json').write_text(json.dumps(dict(commit=commit,base_commit=d['base_commit'],files=checks,all_content_equal=True,payload_SHA256=sha256(PAYLOAD.read_bytes()).hexdigest(),qualification='Pinned full raw HTTPS bytes/SHA256/gitblob including default HTML; branch lease/compare/head checks recorded separately.'),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(commit=commit,files=len(checks),all_content_equal=True)))
if __name__=='__main__':{'prepare':prepare,'stage':stage,'verify':lambda:verify(sys.argv[2])}[sys.argv[1]]()
