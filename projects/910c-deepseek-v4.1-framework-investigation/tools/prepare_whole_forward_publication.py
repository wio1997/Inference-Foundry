#!/usr/bin/env python3
# coding: utf-8
"""Reconcile dispatch/branch evidence and the default-view status onto fresh GitHub state."""
from pathlib import Path
from hashlib import sha256,sha1
from datetime import datetime,timezone
import json,re
from update_whole_forward_maps import TITLE,SUMMARY,NEXT,BODY,default_banner
ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
    base=json.loads(Path('/tmp/whole-forward-public-base.json').read_text());files=[]
    def add(path,content,source):
        b=content.encode();files.append(dict(path=PREFIX+path,content=content,local_source=source,bytes=len(b),content_sha256=sha256(b).hexdigest(),git_blob_sha1=sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
    def raw(name):p=ROOT/name;add('handoff/evidence/'+name,p.read_text(),str(p))
    for n in ['全模型派发继承与合法分支覆盖判断.md','whole-forward-dependencies.json','whole-forward-binding-dependencies.json','whole-forward-frozen-config.json','whole-forward-dispatch.json','whole-forward-coverage-delta.json','whole-forward-config-shape-controls.json','whole-forward-binding-source-excerpts.json','whole-forward-frontier.json']:raw(n)
    for n in ['read_whole_forward_dependencies.py','read_whole_forward_binding_entry.py','review_whole_forward_dispatch.py','probe_whole_forward_config_shapes.py','extract_whole_forward_binding_evidence.py','update_whole_forward_maps.py','prepare_whole_forward_publication.py']:
        p=Path('tools')/n;add('tools/'+n,p.read_text(),str(p))
    front=json.loads((ROOT/'whole-forward-frontier.json').read_text());stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    for name,s in base['files'].items():
        if name.endswith('.html'):
            assert len(s.encode())>1000000;banner=default_banner('../全模型派发继承与合法分支覆盖判断.md')
            if 'id="whole-forward-default-entry"' in s:s,n=re.subn(r'<aside id="whole-forward-default-entry".*?</aside>',lambda _:banner,s,count=1,flags=re.S)
            else:s,n=re.subn(r'(<body[^>]*>)',lambda m:m.group(1)+banner,s,count=1)
            assert n==1;add(name,s,'Freshpublicbase '+base['base']+' largeHTML verified with connectorblobSHA and pinnedrawHTTPS; statusaside only; all timeline/data/JS preserved');continue
        if name.endswith('.md'):
            section=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY;i=s.find('\n')+1;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+section+'\n'+s[i:]
            if name=='handoff/README.md':
                marker='## 4. 优化地图与研究阶段';assert marker in s;s=s.replace(marker,marker+'\n\n当前源码覆盖：40源注册/C3/别名/逐层角色已绑定；156原静态位中49受限源码约束、107输入/控制/消费者未闭合。新增框架准入0，全链路未扫全。默认时间线入口已加当前状态，旧数据/全局图/恢复历史保留。[派发与分支判断](evidence/全模型派发继承与合法分支覆盖判断.md)。\n',1)
                a=s.index('**2026-11-10 当前资格**');z=s.index('\n\n',a);s=s[:a]+'**2026-11-10 当前资格**：每替换原语逐位一致。受限CPU原型1保留，新增框架替换0，完整路径搜索未完成。40当前source/16方法解析与PythonC3、原标量40层角色和private draft128/3/3已核；156静态位中49有受限源码/角色约束，107仍须追输入/控制与消费者，不计为算子或已完成路径。配置stub两形态不代表真实Transformers/对象字段；纯文本或arch-info False不能证明vision-bias不存在。rawMTPbank分配并写但固定DSpark数值getter不消费，旧同预算小方案不重开；大共同frame/owner条件QSL/输入租约与双端退役仍待确认。loop047 Run114154Z PASS/NA仅此界，Run124045Z仍PLANNED。'+s[z:]
        else:
            d=json.loads(s);d.update(whole_forward_dispatch_review=front,status=front['status'],next_action=NEXT,whole_chain_complete=False,new_framework_replacement_admissions=0,updated_at_utc=stamp,current_analysis_base_commit='b360ca7c56f773068e91e2da7f4965b79ed3de1f')
            if name=='progress.json':d.update(overall_complete=False,active_loop='loop-047',active_module='Selectedtarget/draft/Engram dispatch and107remaining V1input/control/consumer sites',previous_goal_turn_classification='PROGRESS:40currentpins/frozenconfig/16MROaliasresolutions/PythonC3/40scalarroles/privateDSpark;156siteledger49sourcebindings107unclosed;two explicitconfig-shape controls;noNPUorframeworkadmission;defaultviewstatusreconciled.',latest_loop047_completed_review='Run114154Z boundeddispatch/oldrangePASSNA; two source configrepresentationcontrols separate; Run124045Z PLANNED remaining input/metadata/readers')
            else:d['completeness_established']=False
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Freshpublicbase '+base['base']+' reconciled; previousprofile/timeline/coverage/restoration and research preserved')
    dest='handoff/evidence/whole-forward-publication-manifest.json';manifest=dict(base_commit=base['base'],paths=[f['path'] for f in files]+[PREFIX+dest],whole_chain_complete=False,new_framework_replacement_admissions=0,new_primitive_admissions=0,isolated_CPU_prototypes_passed=1,every_replacement_bitwise=True,qualification='All40sourcepins/16methodMRO+aliases/40roles/156sites and bothconfigcontrols full.49sourcebindings are not execution/native coverage;107unclosed. Default >1MB HTML statusaside only, base connector returnedemptycontents: pinnedrawHTTPgitblobSHAverified; complete postpublish rawHTTPreadback required. No proposed kernel/framework patch applied; newTorch/NPU/model/profile/service/remotewrites0.',large_base_verification=base['large_file_http_verification'])
    p=ROOT/'whole-forward-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');raw(p.name);assert len(files)==30 and len({f['path'] for f in files})==30
    p=Path('/tmp/whole-forward-publication-payload.json');p.write_text(json.dumps(dict(base_commit=base['base'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),max_file_bytes=max(f['bytes'] for f in files),path=str(p))))
if __name__=='__main__':main()
