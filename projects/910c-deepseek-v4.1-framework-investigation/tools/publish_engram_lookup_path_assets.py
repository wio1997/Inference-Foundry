#!/usr/bin/env python3
"""Prepare exact blobs and verify content; ref movement uses the connector."""
from pathlib import Path
from hashlib import sha1,sha256
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from urllib.parse import quote
import base64,json,re,subprocess,sys
from update_engram_lookup_path_maps import R,REPORT,TITLE,BODY,NEXT,PATTERN,banner,frontier
PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
PAYLOAD=Path('/tmp/engram-lookup-path-publication-payload.json')
def blob(b):return sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def prepare():
    base=json.loads(Path('/tmp/engram-lookup-path-public-base-checks.json').read_text());prior=json.loads(Path('/tmp/compressor-path-publication-payload.json').read_text());receipt=json.loads((R/'compressor-path-publication.json').read_text());assert base['base_commit']==receipt['commit']=='13727e548c829438faa22bf79c99b86a711d7557' and receipt['all_content_equal']
    old={f['path']:f for f in prior['files']};files=[];checks=[]
    def add(path,s,origin):
        s=s.replace('](Engram完整交付与融合判断.md)','](engram-execution-review.md)') if path=='handoff/evidence/'+REPORT else s; b=s.encode();files.append(dict(path=PREFIX+path,content=s,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),local_source=origin))
    for n in ['engram-lookup-path-dependencies.json','engram-hash-path-dependencies.json','engram-hash-kernel-dependencies.json','engram-lookup-path-parent-primary.json','engram-lookup-path-price.json','engram-lookup-path-source-excerpts.json','engram-lookup-path-recursive-intersections.json','engram-lookup-path-review.json','engram-lookup-path-integrity.json','engram-lookup-path-frontier.json','recursive-semantic-frontier.json',REPORT]:add('handoff/evidence/'+n,(R/n).read_text(),str(R/n))
    for n in ['read_engram_lookup_path_sources','read_engram_hash_path_sources','read_engram_hash_kernel_sources','analyze_engram_lookup_path_price','review_engram_lookup_path','build_recursive_semantic_frontier','update_engram_lookup_path_maps','publish_engram_lookup_path_assets']:add('tools/'+n+'.py',Path('tools',n+'.py').read_text(),'tools/'+n+'.py')
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
            if name=='handoff/README.md':s=re.sub(r'^当前源码覆盖：.*$',lambda _:'当前源码覆盖：17当前身份、43具名body/range、19条receiver路由（原工作表16边交集）。实际Engram V1 history/hash/UVA到双层残差主体补齐，旧机制去重；24完整双层交付native包络1.954322%旧代表窗，父段保留，小lookup沿旧预算停止。累计前沿138节点全文记录仅供证据，不作为合法路径完成率；每替换原语逐位一致，native新增0，全链路未闭合。',s,count=1,flags=re.M)

        else:
            d=json.loads(s);d.update(status=frontier()['status'],engram_lookup_path_review=frontier(),next_action=NEXT,whole_chain_complete=False,completeness_established=False,new_framework_replacement_admissions=0)
            if name=='progress.json':d.update(active_loop='loop-047',overall_complete=False,latest_review='Run210932Z actualEngram hash/UVA/state and wholeparent price bounded PASSNA; Run213303Z cumulative frontier andMoEtransform consumers PLANNED',previous_goal_turn_classification='PROGRESS:17current source identities,43named body/range walks,19receiver routes/16old edge intersections,15bodyworklist intersections.24complete two-layer branches/48Wkv-gate chains, native envelope1.954322%mirroredwindow, larger input/owner/HC/iteration retained; lookup0.031226%small child stopped under oldbudget. No newmechanism or numerical test claimed. Cumulative exactSHA fullbody evidence138/319nodes,181nodes/177ranges withoutwholebody receipt,35unresolved receiver edges have explicit semantic receipts; not allbranch/nativeclosure or requestgain.')

            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Fresh connector blob SHA + pinned full base '+base['base_commit'])
    name='handoff/evidence/engram-lookup-path-publication-manifest.json';manifest=dict(base_commit=base['base_commit'],base_tree_sha=receipt['tree'],paths=[x['path'] for x in files]+[PREFIX+name],base_checks=checks,HTML_only_own_aside=True,current_source_pins=17,source_walks=43,receiver_routes=19,matched_original_edges=16,new_numerical_tests=0,old_bound_inputs=122,old_unbound_inputs=34,new_native_admissions=0,every_replacement_bitwise=True,whole_chain_complete=False)
    p=R/'engram-lookup-path-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(name,p.read_text(),str(p));assert len(files)==len({x['path'] for x in files});PAYLOAD.write_text(json.dumps(dict(base_commit=base['base_commit'],base_tree_sha=receipt['tree'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),max_bytes=max(x['bytes'] for x in files))))

def stage():
    d=json.loads(PAYLOAD.read_text())
    def one(f):
        b=f['content'].encode();assert sha256(b).hexdigest()==f['sha256'] and blob(b)==f['git_blob_sha1'];p=subprocess.run(['gh','api','repos/wio1997/Inference-Foundry/git/blobs','-X','POST','--input','-'],input=json.dumps(dict(content=base64.b64encode(b).decode(),encoding='base64')).encode(),capture_output=True,timeout=90)
        if p.returncode:raise RuntimeError('Blob stage failed: '+f['path'])
        h=json.loads(p.stdout)['sha'];assert h==f['git_blob_sha1'];return dict(path=f['path'],mode='100644',type='blob',sha=h)
    with ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(one,d['files']))
    (R/'engram-lookup-path-staged-blobs.json').write_text(json.dumps(dict(base_commit=d['base_commit'],base_tree_sha=d['base_tree_sha'],entries=entries,payload_SHA256=sha256(PAYLOAD.read_bytes()).hexdigest(),branch_mutation=False),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(files=len(entries),all_blob_SHAs_match=True,branch_mutation=False)))
def verify(commit):
    assert re.fullmatch('[0-9a-f]{40}',commit);d=json.loads(PAYLOAD.read_text())
    def one(f):
        u='https://raw.githubusercontent.com/wio1997/Inference-Foundry/'+commit+'/'+quote(f['path'],safe='/')
        with urlopen(Request(u,headers={'User-Agent':'Inference-Foundry-source-audit'}),timeout=45) as response:b=response.read()
        assert b==f['content'].encode() and sha256(b).hexdigest()==f['sha256'] and blob(b)==f['git_blob_sha1'];return dict(path=f['path'],bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),content_equal=True)
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(one,d['files']))
    (R/'engram-lookup-path-publication.json').write_text(json.dumps(dict(commit=commit,base_commit=d['base_commit'],files=checks,all_content_equal=True,payload_SHA256=sha256(PAYLOAD.read_bytes()).hexdigest(),qualification='Pinned full raw HTTPS bytes/SHA256/gitblob including default HTML; branch lease/compare/head checks recorded separately.'),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(commit=commit,files=len(checks),all_content_equal=True)))
if __name__=='__main__':{'prepare':prepare,'stage':stage,'verify':lambda:verify(sys.argv[2])}[sys.argv[1]]()
