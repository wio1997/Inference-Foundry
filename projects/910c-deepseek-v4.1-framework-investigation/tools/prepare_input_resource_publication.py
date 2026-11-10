#!/usr/bin/env python3
"""Reconcile this source delta against pinned workbench documents and full HTML."""
from pathlib import Path
from hashlib import sha1,sha256
import base64,json,re,zlib
from update_input_resource_maps import TITLE,REPORT,BODY,NEXT,PATTERN,banner
ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
    base=json.loads(Path('/tmp/input-resource-public-base-manifest.json').read_text());files=[]
    def add(path,content,source):
        b=content.encode();files.append(dict(path=PREFIX+path,content=content,local_source=source,bytes=len(b),sha256=sha256(b).hexdigest(),git_blob_sha1=sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
    evidence=['input-resource-dependencies.json','input-group-dispatch-dependencies.json','input-resource-callsite-inventory.json','input-resource-source-plan.json','input-resource-group-source-excerpts.json','input-retirement-controls.json','input-retirement-source-excerpts.json','input-frozen-flags.json','draft-expansion-native-labels.json','input-kernel-windows-receipt.json','input-kernel-island-price.json','input-resource-source-review.json','input-resource-consumer-source-excerpts.json','input-resource-frontier.json','input-resource-integrity.json','input-resource-grouping-a1-invalid.json',REPORT]
    for n in evidence:add('handoff/evidence/'+n,(ROOT/n).read_text(),str(ROOT/n))
    tools=['read_input_resource_dependencies.py','read_input_resource_callers.py','read_input_group_dispatch.py','probe_input_retirement_contracts.py','probe_input_resource_grouping.py','read_draft_expansion_task_labels.py','read_input_kernel_windows.py','read_input_frozen_flags.py','price_input_kernel_islands.py','review_input_resource_consumers.py','update_input_resource_maps.py','verify_input_resource_evidence.py','prepare_input_resource_publication.py','stage_input_resource_blobs.py','verify_input_resource_publication.py']
    for n in tools:add('tools/'+n,(Path('tools')/n).read_text(),'tools/'+n)
    source_index=[]
    for folder in ['input-resource-source','input-group-source','input-flag-source']:
        for p in sorted((ROOT/folder).rglob('*.py')):
            dest='handoff/evidence/'+str(p.relative_to(ROOT));add(dest,p.read_text(),str(p));source_index.append(dict(path=PREFIX+dest,sha256=sha256(p.read_bytes()).hexdigest()))
    raw=ROOT/'input-kernel-windows-raw.json.zlib';b=raw.read_bytes();decoded=zlib.decompress(b);d=json.loads(decoded);assert len(d)==16 and sum(len(x['windows']) for x in d)==3696
    encoded=base64.b64encode(b).decode();dest='handoff/evidence/input-kernel-windows-raw.zlib.base64';add(dest,'\n'.join(encoded[i:i+120] for i in range(0,len(encoded),120))+'\n',str(raw))
    native=dict(path=PREFIX+dest,format='Base64 ASCII -> zlib -> complete original JSON array',compressed_SHA256=sha256(b).hexdigest(),decompressed_SHA256=sha256(decoded).hexdigest(),compressed_bytes=len(b),decompressed_bytes=len(decoded),ranks=16,all_rank_windows=3696,active_rank_windows=3624,contains_all_matching_native_tasks_including_outside_qualified_windows=True,reconstruction='base64.b64decode(text) then zlib.decompress(bytes); compressed bytes match the original reader receipt exactly. No native row omitted or synthesized.')
    add('handoff/evidence/input-kernel-windows-public-manifest.json',json.dumps(native,ensure_ascii=False,indent=2)+'\n','Full original compressed/native record bytes')
    front=json.loads((ROOT/'input-resource-frontier.json').read_text());base_checks=[]
    for name,info in base['files'].items():
        b=(Path('/tmp/input-resource-public-base')/name).read_bytes();blob=lambda x:sha1(b'blob '+str(len(x)).encode()+b'\0'+x).hexdigest()
        if blob(b)!=info['sha'] and b.endswith(b'\n') and blob(b[:-1])==info['sha']:b=b[:-1]
        assert blob(b)==info['sha'],name;base_checks.append(dict(path=name,sha256=sha256(b).hexdigest(),git_blob_sha1=blob(b),bytes=len(b)));s=b.decode()
        if name.endswith('.html'):
            old=s;s,n=re.subn(PATTERN,lambda _:banner('../'+REPORT),s,count=1,flags=re.S);assert n==1 and re.sub(PATTERN,'',s,flags=re.S)==re.sub(PATTERN,'',old,flags=re.S)
        elif name.endswith('.md'):
            body=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY;i=s.index('\n')+1;assert '## 当前断点 · '+TITLE not in s;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+body+'\n'+s[i:]
            if name=='handoff/README.md':
                s=re.sub(r'^当前源码覆盖：.*$',lambda _:'当前源码覆盖：本段30当前源码身份，原156位90受限绑定/66待追；13KV组/26runner builder/1draft源计划与3624旧活跃窗一次扩展相符。A3默认Engram同步；19原Python退役/CPUguard控制不当native准入。756文件只是特定词盘点，不是审阅覆盖。全链路仍未闭合。',s,count=1,flags=re.M)
                a=s.find('**2026-11-10 当前资格**')
                if a>=0:
                    z=s.index('\n\n',a);s=s[:a]+'**2026-11-10 当前资格**：每替换原语逐位一致。Run150110Z有界PASS/NA，Run155707Z PLANNED；90源码约束/66待追，原语/框架准入0，旧CPU租约原型1保留。固定native两岛全免仅原C8 .365136%TTFT/.524144%整批，停止该小边界独立开发；不是大input/hostframe上界，不迁移六普通。更大frame保留actual/FIA/draft视图、全部owner/lease/native4096B与真实净价缺口。'+s[z:]
        else:
            j=json.loads(s);j.update(status=front['status'],next_action=NEXT,input_resource_consumer_review=front,whole_chain_complete=False,completeness_established=False,new_framework_replacement_admissions=0)
            if name=='progress.json':j.update(active_loop='loop-047',overall_complete=False,latest_review='Run150110Z bounded source/resource/native-window PASSNA;90bound66pending; Run155707Z PLANNED',previous_goal_turn_classification='PROGRESS:30current source pins; independent54resource/13KV/26runner/1draft plan;19cleanup/CPUguard source controls;3624old native windows each1expansion; A3overlapfalse. Stop absent repeats/existingwait relocation and small C8 native-only scopes, retain larger input-frame. No newprimitive/framework admission.')
            s=json.dumps(j,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Fresh pinned public base '+base['base_commit']+'; preserve other chats history/dataflow/timeline/token evidence')
    dest='handoff/evidence/input-resource-publication-manifest.json';manifest=dict(base_commit=base['base_commit'],base_tree_sha=base['base_tree_sha'],paths=[f['path'] for f in files]+[PREFIX+dest],base_checks=base_checks,new_source_bodies=source_index,current_source_pins=30,source_bindings=90,pending_original_sites=66,full_native_records=native,HTML_only_own_status_aside=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,every_replacement_bitwise=True,qualification='Current source scalar/stub and old native task/clock evidence only. All other public work preserved. Full postpublication byte readback required, including large HTML and exact base64 native reconstruction.')
    p=ROOT/'input-resource-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add(dest,p.read_text(),str(p));assert len({f['path'] for f in files})==len(files)
    out=Path('/tmp/input-resource-publication-payload.json');out.write_text(json.dumps(dict(base_commit=base['base_commit'],base_tree_sha=base['base_tree_sha'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),new_source_bodies=len(source_index),payload_bytes=out.stat().st_size,max_file_bytes=max(f['bytes'] for f in files))))
if __name__=='__main__':main()
