#!/usr/bin/env python3
"""Prepare exact blobs and verify content; ref movement uses the connector."""
from pathlib import Path
from hashlib import sha1,sha256
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from urllib.parse import quote
import base64,json,re,subprocess,sys
from update_moe_transform_path_maps import R,REPORT,TITLE,BODY,NEXT,PATTERN,banner,frontier
PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
PAYLOAD=Path('/tmp/moe-transform-path-publication-payload.json')
def blob(b):return sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def prepare():
    base=json.loads(Path('/tmp/moe-transform-path-public-base-checks.json').read_text());prior=json.loads(Path('/tmp/engram-lookup-path-publication-payload.json').read_text());receipt=json.loads((R/'engram-lookup-path-publication.json').read_text());assert base['base_commit']==receipt['commit']=='89e9f426774ff0bdfd1bebc3aac56ad70070ce5e' and receipt['all_content_equal']
    old={f['path']:f for f in prior['files']};files=[];checks=[]
    def add(path,s,origin):
        b=s.encode();files.append(dict(path=PREFIX+path,content=s,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),local_source=origin))
    for n in ['moe-transform-path-dependencies.json','moe-transform-binding-dependencies.json','moe-transform-path-parent-primary.json','moe-transform-path-controls.json','moe-transform-path-price.json','moe-transform-path-source-excerpts.json','moe-transform-path-recursive-intersections.json','moe-transform-path-review.json','moe-transform-path-integrity.json','moe-transform-path-frontier.json','moe-transform-path-cumulative-frontier.json',REPORT]:add('handoff/evidence/'+n,(R/n).read_text(),str(R/n))
    for n in ['read_moe_transform_path_sources','read_moe_transform_binding_sources','probe_moe_transform_path_controls','review_moe_transform_path','build_recursive_semantic_frontier','update_moe_transform_path_maps','publish_moe_transform_path_assets']:add('tools/'+n+'.py',Path('tools',n+'.py').read_text(),'tools/'+n+'.py')
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
            if name=='handoff/README.md':s=re.sub(r'^当前源码覆盖：.*$',lambda _:'当前源码覆盖：15当前身份、46具名body/range、21条receiver路由（原工作表14边交集）、8原Python协议组。冻结SP_SEDP/None routed transform归位，runtime归约/层名cursor/最终Tensor返回保留；内部tuple直出会激活外层legacy scale/add与TP reduce。旧add-HC/sharedDFC大父段保留。累计148节点有全文证据、171缺全文记录，只是证据索引；每替换原语逐位一致，native新增0，全链路未闭合。',s,count=1,flags=re.M)

        else:
            d=json.loads(s);d.update(status=frontier()['status'],moe_transform_path_review=frontier(),next_action=NEXT,whole_chain_complete=False,completeness_established=False,new_framework_replacement_admissions=0)
            if name=='progress.json':d.update(active_loop='loop-047',overall_complete=False,latest_review='Run213303Z actualMoE transform/reduction/finalTensor consumers bounded PASSNA; Run215338Z scheduler cachefree/nextiteration helpers PLANNED',previous_goal_turn_classification='PROGRESS:15current identities,46named body/range walks,21receiver routes/14oldedges,8original Python protocol groups. FrozenSP_SEDP and routedtransformNone have no phantom latent/sharedTP work; runtimecomm hooks, cursor and finalTensor preserved. Originalouter tuple negative triggers legacy scale/add+TP. Reuse12exactSHA oldsources and complete MoE/addHC parent prices; no newnative/requestgain. Cumulative148bodyevidence nodes,171nodes/167ranges withoutwholebody,44unresolved receiver edges with explicitreceipt; earlier cumulativebytes preserved, no globalclosure.')

            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Fresh connector blob SHA + pinned full base '+base['base_commit'])
    name='handoff/evidence/moe-transform-path-publication-manifest.json';manifest=dict(base_commit=base['base_commit'],base_tree_sha=receipt['tree'],paths=[x['path'] for x in files]+[PREFIX+name],base_checks=checks,HTML_only_own_aside=True,current_source_pins=15,source_walks=46,receiver_routes=21,matched_original_edges=14,original_protocol_groups=8,old_bound_inputs=122,old_unbound_inputs=34,new_native_admissions=0,every_replacement_bitwise=True,whole_chain_complete=False)
    p=R/'moe-transform-path-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(name,p.read_text(),str(p));assert len(files)==len({x['path'] for x in files});PAYLOAD.write_text(json.dumps(dict(base_commit=base['base_commit'],base_tree_sha=receipt['tree'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),max_bytes=max(x['bytes'] for x in files))))

def stage():
    d=json.loads(PAYLOAD.read_text())
    def one(f):
        b=f['content'].encode();assert sha256(b).hexdigest()==f['sha256'] and blob(b)==f['git_blob_sha1'];p=subprocess.run(['gh','api','repos/wio1997/Inference-Foundry/git/blobs','-X','POST','--input','-'],input=json.dumps(dict(content=base64.b64encode(b).decode(),encoding='base64')).encode(),capture_output=True,timeout=90)
        if p.returncode:raise RuntimeError('Blob stage failed: '+f['path'])
        h=json.loads(p.stdout)['sha'];assert h==f['git_blob_sha1'];return dict(path=f['path'],mode='100644',type='blob',sha=h)
    with ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(one,d['files']))
    (R/'moe-transform-path-staged-blobs.json').write_text(json.dumps(dict(base_commit=d['base_commit'],base_tree_sha=d['base_tree_sha'],entries=entries,payload_SHA256=sha256(PAYLOAD.read_bytes()).hexdigest(),branch_mutation=False),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(files=len(entries),all_blob_SHAs_match=True,branch_mutation=False)))
def verify(commit):
    assert re.fullmatch('[0-9a-f]{40}',commit);d=json.loads(PAYLOAD.read_text())
    def one(f):
        u='https://raw.githubusercontent.com/wio1997/Inference-Foundry/'+commit+'/'+quote(f['path'],safe='/')
        with urlopen(Request(u,headers={'User-Agent':'Inference-Foundry-source-audit'}),timeout=45) as response:b=response.read()
        assert b==f['content'].encode() and sha256(b).hexdigest()==f['sha256'] and blob(b)==f['git_blob_sha1'];return dict(path=f['path'],bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),content_equal=True)
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(one,d['files']))
    (R/'moe-transform-path-publication.json').write_text(json.dumps(dict(commit=commit,base_commit=d['base_commit'],files=checks,all_content_equal=True,payload_SHA256=sha256(PAYLOAD.read_bytes()).hexdigest(),qualification='Pinned full raw HTTPS bytes/SHA256/gitblob including default HTML; branch lease/compare/head checks recorded separately.'),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(commit=commit,files=len(checks),all_content_equal=True)))
if __name__=='__main__':{'prepare':prepare,'stage':stage,'verify':lambda:verify(sys.argv[2])}[sys.argv[1]]()
