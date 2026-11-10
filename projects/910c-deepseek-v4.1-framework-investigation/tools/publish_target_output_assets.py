#!/usr/bin/env python3
"""Stage exact immutable Git blobs; branch movement stays with the connector."""
from pathlib import Path
from hashlib import sha1,sha256
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from urllib.parse import quote
import base64,json,re,subprocess,sys
from update_target_output_maps import R,REPORT,TITLE,BODY,NEXT,PATTERN,banner,frontier

PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
PAYLOAD=Path('/tmp/target-output-publication-payload.json')
def blob(b):return sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def prepare():
    base=json.loads(Path('/tmp/target-output-public-base-checks.json').read_text());prior=json.loads(Path('/tmp/draft-output-publication-payload.json').read_text());receipt=json.loads((R/'draft-output-publication.json').read_text())
    assert base['base_commit']==receipt['commit']=='0613a54e995533969c577deed7a95c24a1276f6c' and receipt['all_content_equal'] and receipt['all_connector_shas_match']
    old={f['path']:f for f in prior['files']};files=[];checks=[]
    def add(path,text,origin):
        b=text.encode();files.append(dict(path=PREFIX+path,content=text,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),local_source=origin))
    names=['target-output-dependencies.json','target-output-aux-dependencies.json','target-output-parent-primary.json','target-output-controls.json','target-output-source-excerpts.json','target-output-review.json','target-output-price.json','target-output-frontier.json','recursive-source-worklist.json','recursive-source-disposition.json','target-output-integrity.json',REPORT]
    for n in names:add('handoff/evidence/'+n,(R/n).read_text(),str(R/n))
    for n in ['read_target_output_sources','read_target_output_aux','build_recursive_source_worklist','probe_target_output_protocol','analyze_target_output_price','review_target_output','update_target_output_maps','publish_target_output_assets']:add('tools/'+n+'.py',Path('tools',n+'.py').read_text(),'tools/'+n+'.py')
    for p in sorted((R/'target-output-source').rglob('*.py')):add('handoff/evidence/'+str(p.relative_to(R)),p.read_text(),str(p))
    for item in base['files']:
        name=item['path'];f=old[PREFIX+name];b=f['content'].encode();assert blob(b)==item['sha']==f['git_blob_sha1'] and sha256(b).hexdigest()==f['sha256'];s=b.decode()
        checks.append(dict(path=name,sha=item['sha'],bytes=len(b),sha256=sha256(b).hexdigest()))
        if name.endswith('.html'):
            before=s;s,n=re.subn(PATTERN,lambda _:banner('../'+REPORT),s,count=1,flags=re.S);assert n==1 and re.sub(PATTERN,'',before,flags=re.S)==re.sub(PATTERN,'',s,flags=re.S)
        elif name.endswith('.md'):
            body=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY
            i=s.index('\n')+1;assert '## 当前断点 · '+TITLE not in s;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+body+'\n'+s[i:]
            if name=='handoff/README.md':s=re.sub(r'^当前源码覆盖：.*$',lambda _:'当前源码覆盖：27当前源、64具名body走读、12原Python协议/有限控制；target bonus/拒绝全score读者到CPU/RPC/Engine下一轮分列。319body/2739callsite为保守递归工作表，含字段receiver/import/闭包/native未闭合边，不能当扫完。原122源码约束/34未定输入不变；旧固定投影小方案按原预算停止，大owner/Markov候选保留。每替换原语逐位一致，新增准入0。',s,count=1,flags=re.M)
        else:
            d=json.loads(s);d.update(status=frontier()['status'],target_output_review=frontier(),next_action=NEXT,whole_chain_complete=False,completeness_established=False,new_framework_replacement_admissions=0)
            if name=='progress.json':d.update(active_loop='loop-047',overall_complete=False,latest_review='Run191838Z target complete-output/recursive-source frontier bounded PASSNA; next Run194757Z PLANNED',previous_goal_turn_classification='PROGRESS:27 current identities,64 named source body walks,12 original Python protocols/finite controls. Complete score/raw/NaN/RNG/ID packet/async D2H/RPC/Engine/scheduler consumers traced.24 old matched full head/reject/seed chains; paid conditional score-only net~0.30%local window, request credit0, stopped tiny fixed-projection scopes retained. Explicit319body2739callsite conservative frontier separates34 unknown inputs from unreviewed source/receiver/native legs; not full completion or native admission.')
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Fresh connector base blob SHA + previously verified full base bytes '+base['base_commit'])
    name='handoff/evidence/target-output-publication-manifest.json'
    manifest=dict(base_commit=base['base_commit'],base_tree_sha=receipt['tree'],paths=[f['path'] for f in files]+[PREFIX+name],base_checks=checks,HTML_only_own_aside=True,source_pins=27,protocol_controls=12,named_source_body_walks=64,recursive_nodes=319,recursive_calls=2739,old_bound_inputs=122,old_unbound_inputs=34,new_admissions=0,every_replacement_bitwise=True,whole_chain_complete=False)
    p=R/'target-output-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(name,p.read_text(),str(p));assert len(files)==len({f['path'] for f in files})
    PAYLOAD.write_text(json.dumps(dict(base_commit=base['base_commit'],base_tree_sha=receipt['tree'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),max_bytes=max(f['bytes'] for f in files))))
def stage():
    d=json.loads(PAYLOAD.read_text())
    def one(f):
        b=f['content'].encode();assert sha256(b).hexdigest()==f['sha256'] and blob(b)==f['git_blob_sha1']
        p=subprocess.run(['gh','api','repos/wio1997/Inference-Foundry/git/blobs','-X','POST','--input','-'],input=json.dumps(dict(content=base64.b64encode(b).decode(),encoding='base64')).encode(),capture_output=True,timeout=90)
        if p.returncode:raise RuntimeError('Blob stage failed: '+f['path'])
        h=json.loads(p.stdout)['sha'];assert h==f['git_blob_sha1'];return dict(path=f['path'],mode='100644',type='blob',sha=h)
    with ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(one,d['files']))
    (R/'target-output-staged-blobs.json').write_text(json.dumps(dict(base_commit=d['base_commit'],base_tree_sha=d['base_tree_sha'],entries=entries,payload_SHA256=sha256(PAYLOAD.read_bytes()).hexdigest(),branch_mutation=False),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(files=len(entries),all_blob_SHAs_match=True,branch_mutation=False)))
def verify(commit):
    assert re.fullmatch('[0-9a-f]{40}',commit);d=json.loads(PAYLOAD.read_text())
    def one(f):
        url='https://raw.githubusercontent.com/wio1997/Inference-Foundry/'+commit+'/'+quote(f['path'],safe='/')
        with urlopen(Request(url,headers={'User-Agent':'Inference-Foundry-source-audit'}),timeout=45) as response:b=response.read()
        assert b==f['content'].encode() and sha256(b).hexdigest()==f['sha256'] and blob(b)==f['git_blob_sha1'];return dict(path=f['path'],bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),content_equal=True)
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(one,d['files']))
    out=dict(commit=commit,base_commit=d['base_commit'],files=checks,all_content_equal=True,payload_SHA256=sha256(PAYLOAD.read_bytes()).hexdigest(),qualification='Pinned raw HTTPS full bytes/SHA256/gitblob including large HTML. Connector SHA/ref lease and final head recorded separately.')
    (R/'target-output-publication.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(commit=commit,files=len(checks),all_content_equal=True)))
if __name__=='__main__':{'prepare':prepare,'stage':stage,'verify':lambda:verify(sys.argv[2])}[sys.argv[1]]()
