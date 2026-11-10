#!/usr/bin/env python3
"""Reconcile V1 owner/packet findings against fresh public workbench state."""
from pathlib import Path
from hashlib import sha256,sha1
import json,re
from update_v1_input_maps import TITLE,BODY,NEXT,banner
ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
    base=json.loads(Path('/tmp/v1-input-public-base.json').read_text());files=[]
    def add(path,content,source):
        b=content.encode();files.append(dict(path=PREFIX+path,content=content,local_source=source,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
    raw=['v1-input-dependencies.json','v1-input-source-review.json','v1-qsl-owner-controls.json','v1-packet-dependencies.json','v1-packet-write-extent.json','v1-input-parent-price.json','v1-input-source-excerpts.json','v1-input-frontier.json','v1-input-integrity.json','v1-packet-metadata-directory.json','V1输入到原生packet的owner与完整字节判断.md','v1-packet-write-extent-cpu/driver.cpp','v1-packet-write-extent-cpu/run.log']
    for n in raw:add('handoff/evidence/'+n,(ROOT/n).read_text(),str(ROOT/n))
    tools=['read_v1_input_dependencies.py','review_v1_input_owners.py','probe_v1_qsl_owner_controls.py','read_v1_packet_boundary.py','probe_v1_packet_write_extent.py','price_v1_input_parent.py','update_v1_input_maps.py','verify_v1_input_evidence.py','prepare_v1_input_publication.py']
    for n in tools:add('tools/'+n,(Path('tools')/n).read_text(),'tools/'+n)
    front=json.loads((ROOT/'v1-input-frontier.json').read_text())
    own_pattern=r'<aside id="(?:whole-forward-default-entry|v1-input-default-entry)".*?</aside>'
    for name,s in base['files'].items():
        if name.endswith('.html'):
            old=s;new=banner('../V1输入到原生packet的owner与完整字节判断.md',True)
            s,n=re.subn(own_pattern,lambda _:new,s,count=1,flags=re.S)
            if not n:s,n=re.subn(r'(<body[^>]*>)',lambda m:m.group(1)+new,s,count=1)
            assert n==1
            strip=lambda v:re.sub(own_pattern,'',v,flags=re.S)
            assert strip(s)==strip(old);add(name,s,'Fresh base '+base['base']+'; only own status aside; current dataflow/timeline/JS preserved');continue
        if name.endswith('.md'):
            body=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY
            i=s.find('\n')+1;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+body+'\n'+s[i:]
            if name=='handoff/README.md':
                s=re.sub(r'^当前源码覆盖：.*$',lambda _: '当前源码覆盖：本段16Python/11native当前source和1prior exactSHA引用；原156位累计70受限源码绑定、86未闭合。新框架准入0；继续V1预处理真实MM/embedding和Engram/DSpark坐标读者。详见本文最新断点与限定源行段，历史数保留原时间和资格。',s,count=1,flags=re.M)
                a=s.find('**2026-11-10 当前资格**')
                if a>=0:
                    z=s.index('\n\n',a);s=s[:a]+'**2026-11-10 当前资格**：每个替换原语逐位一致。此前受限CPU原型1保留，本轮原语/框架准入0。原156位70受限源码约束、86待追，完整递归调用/消费者/运行人口未闭合；位数不是算子或已扫全证明。Run124045Z PASS/NA只覆盖V1caller/显式stub/CPU serializer写边界，Run133903Z仍PLANNED。canonical incomingQSL、完整packet496/640B外围初始化、全部输入租约/双端退役与真实净价仍待确认；311.050ms是含保留工作的gross诊断价，非实测可省。'+s[z:]
        else:
            d=json.loads(s);d.update(status=front['status'],next_action=NEXT,v1_input_owner_packet_review=front,whole_chain_complete=False,completeness_established=False,new_framework_replacement_admissions=0)
            if name=='progress.json':d.update(active_loop='loop-047',overall_complete=False,latest_review='Run124045Z bounded V1 source/owner/serializer PASSNA;70sourcebindings86unclosed; nextRun133903Z PLANNED',previous_goal_turn_classification='PROGRESS:27currentpins,21additionalsourcebindings,actualQSLreaders/5packetkeys/physicalC2owners,32symbolic+3negative+16Engramstub+12hostserializerextent controls, matched existing gross/paid sensitivity; no newNPU/framework admission.')
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Fresh base '+base['base']+' reconciled; all prior concurrent evidence and history retained')
    dest='handoff/evidence/v1-input-publication-manifest.json'
    manifest=dict(base_commit=base['base'],paths=[f['path'] for f in files]+[PREFIX+dest],current_source_pins=27,prior_exact_SHA_references=1,source_bindings=70,unclosed_sites=86,new_native_admissions=0,whole_chain_complete=False,every_replacement_bitwise=True,large_base_verification=base['large_file_http_verification'],qualification='Exact raw controls and original156site ledger published; no full runtime/native bits or full scan claim. Large default viewer statusaside only; full postpublish rawHTTPS readback required.')
    p=ROOT/'v1-input-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(dest,p.read_text(),str(p))
    assert len(files)==36 and len(set(f['path'] for f in files))==36
    p=Path('/tmp/v1-input-publication-payload.json');p.write_text(json.dumps(dict(base_commit=base['base'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),max_file_bytes=max(f['bytes'] for f in files))))
if __name__=='__main__':main()
