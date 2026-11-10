#!/usr/bin/env python3
"""Preserve foreign worktree bytes while updating this source delivery frontier."""
from pathlib import Path
from hashlib import sha256
import json,re
from review_moe_transform_path import R,REPORT,TITLE,SUMMARY,DETAIL,NEXT
LINKS='[本段判断](evidence/'+REPORT+') · [实际源与分支](evidence/moe-transform-path-review.json) · [逐边派发](evidence/moe-transform-path-recursive-intersections.json) · [累计源码前沿](evidence/moe-transform-path-cumulative-frontier.json) · [完整分支与付成本价](evidence/moe-transform-path-price.json)。'
BODY=SUMMARY+'\n\n原scope与整段价保留：没有从无操作helper认领延迟；runtime comm、层名cursor、最终Tensor返回和BF16 combine边界均为大融合合同。累计全文证据只作索引，不作为合法分支完成率。\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
PATTERN=r'<aside id="(?:engram-lookup-path-default-entry|moe-transform-path-default-entry)"[^>]*>.*?</aside>'
def banner(link,default=True):
    ident='moe-transform-path-default-entry' if default else 'moe-transform-path-followup'
    return '<aside id="'+ident+'" style="padding:16px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前：MoE实际变换、归约与最终Tensor返回消费者</strong><p>15当前源码、46具名body/range、21条receiver路由、8原Python协议组。冻结SP_SEDP+None routed transform，没有latent投影/额外shared TP通信；fake alias与traced bool不能代替runtime归约。内部tuple直交外层会触发legacy scale/add和TP reduce。原add→HC与shared/DFC大父段保留，native新增0。累计148节点有全文证据、171节点仍缺全文记录，不是合法路径完成率。下一段scheduler缓存归还与跨迭代消费者。</p><a href="'+link+'">介绍、到了哪里、结果与具体缺口</a></aside>'

def frontier():
    d=json.loads((R/'moe-transform-path-review.json').read_text());return dict(status=d['status'],current_source_pins=15,source_walks=46,receiver_routes=21,matched_original_edges=14,original_protocol_groups=8,original_bound_inputs=122,original_unbound_inputs=34,classification=d['classification'],whole_chain_complete=False,whole_recursive_callgraph_closed=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,next_action=NEXT,evidence=[REPORT,'moe-transform-path-review.json','moe-transform-path-recursive-intersections.json','moe-transform-path-cumulative-frontier.json','moe-transform-path-price.json'])
def insert(path,body):
    p=Path(path);s=p.read_text();assert '## '+TITLE not in s;i=s.index('\n')+1;p.write_text(s[:i]+'\n## '+TITLE+'\n\n'+body+'\n'+s[i:])
def main():
    names=['HANDOFF.md','171-推理数据流与已查节点.html','171-框架Profile热点.html','171-融合算子全盘排查进度图.html'];(R/'moe-transform-path-shared-before.json').write_text(json.dumps({n:sha256(Path(n).read_bytes()).hexdigest() for n in names},ensure_ascii=False,indent=2)+'\n')
    f=frontier();(R/'moe-transform-path-frontier.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n');new=Path('/tmp/moe-transform-path-next-run.txt').read_text().strip().split('/')[-1];footer='Run213303Z有界PASS/正确性NA，'+new+' PLANNED。公开同步回执待补；loop047活动，无live任务等待。';Path('/tmp/moe-transform-path-handoff-footer.txt').write_text(footer)
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\n'+footer+'\n')
    for m in ['M00','M06','M07']:insert('audit/modules/'+m+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(R/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT+'\n');insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    for p in [R/'source-frontier.json',Path('audit/first-pass.json')]:
        d=json.loads(p.read_text());d.update(status=f['status'],moe_transform_path_review=f,next_action=NEXT,whole_chain_complete=False,completeness_established=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name,default in [('171-推理数据流与已查节点.html',False),('171-框架Profile热点.html',True),('171-融合算子全盘排查进度图.html',False)]:
        p=Path(name);old=p.read_text();pat=PATTERN if default else r'<aside id="(?:engram-lookup-path-followup|moe-transform-path-followup)"[^>]*>.*?</aside>'
        s,n=re.subn(pat,lambda _:banner('audit/execution-search/'+REPORT,default),old,count=1,flags=re.S);assert n==1 and re.sub(pat,'',s,flags=re.S)==re.sub(pat,'',old,flags=re.S);p.write_text(s)
    task=Path('tasks/execution-derived-fusion-search');loop=task/'loops/loop-047'
    for p in [task/'index.json',loop/'state.json']:
        d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for p in [task/'status.md',loop/'status.md']:p.write_text(re.sub(r'^- 下一步：.*$',lambda _:'- 下一步：'+NEXT,p.read_text(),flags=re.M))
    for p in [loop/'journal.md',loop/'report.md']:p.write_text(p.read_text()+'\n\n## Run213303Z · MoE变换归约与Tensor返回消费者（有界PASS/NA）\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一Run '+new+' PLANNED。\n')
    print(json.dumps(dict(current_sources=15,receiver_routes=21,protocol_groups=8,next_run=new,whole_chain_complete=False)))
if __name__=='__main__':main()
