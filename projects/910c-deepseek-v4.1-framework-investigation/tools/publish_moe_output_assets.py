#!/usr/bin/env python3
"""Prepare/stage immutable blobs or verify a pinned public commit; never move refs."""
from pathlib import Path
from hashlib import sha1,sha256
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from urllib.parse import quote
import base64,json,re,subprocess,sys
from update_moe_output_maps import ROOT,REPORT,TITLE,BODY,NEXT,PATTERN,banner,frontier
PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
PAYLOAD=Path('/tmp/moe-output-publication-payload.json')
def blob(b):return sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def prepare():
    base=json.loads(Path('/tmp/moe-output-public-base-checks.json').read_text())
    prior=json.loads(Path('/tmp/row-delivery-publication-payload.json').read_text())
    receipt=json.loads((ROOT/'row-delivery-publication.json').read_text());assert receipt['all_content_equal'] and base['base_commit']==receipt['commit']=='6fb584ec9c46ca1393d20f6190688f23f63ed8e3'
    files=[];oldfiles={f['path']:f for f in prior['files']};checks=[]
    def add(path,content,origin):
        b=content.encode();files.append(dict(path=PREFIX+path,content=content,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),local_source=origin))
    names=['moe-output-dependencies.json','moe-output-config.json','moe-output-parent-primary.json','moe-output-controls.json','moe-output-source-excerpts.json','moe-output-review.json','moe-output-price.json','moe-output-original-C8-tasks.json','moe-output-frontier.json','moe-output-integrity.json',REPORT]
    for n in names:add('handoff/evidence/'+n,(ROOT/n).read_text(),str(ROOT/n))
    for n in ['read_moe_output_sources.py','read_moe_output_config.py','probe_moe_output_controls.py','analyze_moe_output_price.py','review_moe_output.py','update_moe_output_maps.py','publish_moe_output_assets.py']:add('tools/'+n,Path('tools',n).read_text(),'tools/'+n)
    for f in base['files']:
        name=f['path']
        if PREFIX+name in oldfiles:
            old=oldfiles[PREFIX+name];b=old['content'].encode();assert blob(b)==old['git_blob_sha1'] and sha256(b).hexdigest()==old['sha256']
        else:
            assert name=='modules/M07.md';b=Path('/tmp/moe-output-public-M07.md').read_bytes()
            if blob(b)!=f['sha'] and b.endswith(b'\n') and blob(b[:-1])==f['sha']:b=b[:-1]
        assert blob(b)==f['sha']
        checks.append(dict(path=name,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b)));s=b.decode()
        if name.endswith('.html'):
            before=s;s,n=re.subn(PATTERN,lambda _:banner('../'+REPORT),s,count=1,flags=re.S);assert n==1 and re.sub(PATTERN,'',before,flags=re.S)==re.sub(PATTERN,'',s,flags=re.S)
        elif name.endswith('.md'):
            body=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY
            i=s.index('\n')+1;assert '## 当前断点 · '+TITLE not in s;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+body+'\n'+s[i:]
            if name=='handoff/README.md':s=re.sub(r'^当前源码覆盖：.*$',lambda _:'当前源码覆盖：122受限源码绑定/34原动态位未变；18当前源、13原Python控制，1032代表卡完整交付及原C8全16卡2560 add-HC对。实际SEDP已复制、无shared TP通信；runner后置scale1。保留BF16中间舍入的add-HC值得native验证，完整Post endpoint/分cohort净价已记录。新准入0、全链路未闭合。',s,count=1,flags=re.M)
        else:
            d=json.loads(s);d.update(status=frontier()['status'],moe_output_review=frontier(),next_action=NEXT,whole_chain_complete=False,completeness_established=False,new_framework_replacement_admissions=0)
            if name=='progress.json':d.update(active_loop='loop-047',overall_complete=False,latest_review='Run182623Z bounded realSEDP/factory/wholeMoE output/oldprice PASSNA; Run184748Z PLANNED',previous_goal_turn_classification='PROGRESS:18 current source pins,13 original factory/layout/shared pipeline controls; frozen SEDPTrue, no sharedTP collective or postscale. CompletedHC endpoint fixes prior start-only5.746% to6.126% representative-window relaxation.1032 local deliveries and2560 originalC8 addHC pairs; original-rounding addHC candidate has concrete ABI/bit and net-price gates. Original122/34 unchanged; no native/framework admission or full-chain claim.')
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Fresh connector SHA + exact previously fully verified public base '+base['base_commit'])
    name='handoff/evidence/moe-output-publication-manifest.json';manifest=dict(base_commit=base['base_commit'],base_tree_sha=receipt['tree'],paths=[f['path'] for f in files]+[PREFIX+name],base_checks=checks,HTML_only_own_aside=True,source_pins=18,controls=13,bound=122,pending=34,new_admissions=0,every_replacement_bitwise=True,whole_chain_complete=False)
    p=ROOT/'moe-output-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(name,p.read_text(),str(p));assert len(files)==len({f['path'] for f in files})
    PAYLOAD.write_text(json.dumps(dict(base_commit=base['base_commit'],base_tree_sha=receipt['tree'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),max_bytes=max(f['bytes'] for f in files))))
def stage():
    d=json.loads(PAYLOAD.read_text())
    def one(f):
        b=f['content'].encode();assert sha256(b).hexdigest()==f['sha256'] and blob(b)==f['git_blob_sha1']
        r=subprocess.run(['gh','api','repos/wio1997/Inference-Foundry/git/blobs','-X','POST','--input','-'],input=json.dumps(dict(content=base64.b64encode(b).decode(),encoding='base64')).encode(),capture_output=True,timeout=90)
        if r.returncode:raise RuntimeError('Blob stage failed: '+f['path'])
        h=json.loads(r.stdout)['sha'];assert h==f['git_blob_sha1'];return dict(path=f['path'],mode='100644',type='blob',sha=h)
    with ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(one,d['files']))
    (ROOT/'moe-output-staged-blobs.json').write_text(json.dumps(dict(base_commit=d['base_commit'],base_tree_sha=d['base_tree_sha'],entries=entries,payload_SHA256=sha256(PAYLOAD.read_bytes()).hexdigest(),branch_mutation=False),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(files=len(entries),all_blob_SHAs_match=True,branch_mutation=False)))
def verify(commit):
    assert re.fullmatch('[0-9a-f]{40}',commit);d=json.loads(PAYLOAD.read_text())
    def one(f):
        url='https://raw.githubusercontent.com/wio1997/Inference-Foundry/'+commit+'/'+quote(f['path'],safe='/')
        with urlopen(Request(url,headers={'User-Agent':'Inference-Foundry-source-audit'}),timeout=45) as response:b=response.read()
        assert b==f['content'].encode() and len(b)==f['bytes'] and sha256(b).hexdigest()==f['sha256'] and blob(b)==f['git_blob_sha1']
        return dict(path=f['path'],bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),content_equal=True)
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(one,d['files']))
    out=dict(commit=commit,base_commit=d['base_commit'],files=checks,all_content_equal=True,payload_SHA256=sha256(PAYLOAD.read_bytes()).hexdigest(),qualification='Pinned commit full raw HTTPS bytes including large HTML. Connector SHA and final branch lease/head checks are recorded separately before receipt completion.')
    (ROOT/'moe-output-publication.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(commit=commit,files=len(checks),all_content_equal=True)))
if __name__=='__main__':
    {'prepare':prepare,'stage':stage,'verify':lambda:verify(sys.argv[2])}[sys.argv[1]]()
