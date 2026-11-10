#!/usr/bin/env python3
"""Preserve foreign worktree bytes while updating this source delivery frontier."""
from pathlib import Path
from hashlib import sha256
import json,re
from review_engram_lookup_path import R,REPORT,TITLE,SUMMARY,DETAIL,NEXT
LINKS='[本段判断](evidence/'+REPORT+') · [实际源与分支](evidence/engram-lookup-path-review.json) · [逐边派发](evidence/engram-lookup-path-recursive-intersections.json) · [累计源码前沿](evidence/recursive-semantic-frontier.json) · [完整分支与付成本价](evidence/engram-lookup-path-price.json)。'
BODY=SUMMARY+'\n\n实际V1先发布slot history再hash，shared DP绕过每步DP查表通信，UVA lookup已含反量化与BF16 cast；先TP heads再SP行，原Wkv/gate/残差全部读者保留。旧exact-SHA合同与失败配方去重，0新增数值测试。累计前沿只验证全文/调用证据，未将其变成全部分支或native准入。\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
PATTERN=r'<aside id="(?:compressor-path-default-entry|engram-lookup-path-default-entry)"[^>]*>.*?</aside>'
def banner(link,default=True):
    ident='engram-lookup-path-default-entry' if default else 'engram-lookup-path-followup'
    return '<aside id="'+ident+'" style="padding:16px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前：实际Engram hash/history→共享UVA→TP/SP→双层残差</strong><p>17当前源、43具名body/range、19条receiver路由；24完整双层交付组合价，父段native包络1.954322%旧代表窗保留。lookup0.031226%小family沿原预算停止。该机制已有旧记录，本轮补实际主体与覆盖/价格，native新增0。累计前沿138条节点有全文证据、181条缺全文记录；这些不是合法路径完成率。下一处MoE routed/shared transform→combine。</p><a href="'+link+'">介绍、到了哪里、结果与具体缺口</a></aside>'

def frontier():
    d=json.loads((R/'engram-lookup-path-review.json').read_text());return dict(status=d['status'],current_source_pins=17,source_walks=43,receiver_routes=19,matched_original_edges=16,new_numerical_tests=0,original_bound_inputs=122,original_unbound_inputs=34,classification=d['classification'],whole_chain_complete=False,whole_recursive_callgraph_closed=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,next_action=NEXT,evidence=[REPORT,'engram-lookup-path-review.json','engram-lookup-path-recursive-intersections.json','recursive-semantic-frontier.json','engram-lookup-path-price.json'])
def insert(path,body):
    p=Path(path);s=p.read_text();assert '## '+TITLE not in s;i=s.index('\n')+1;p.write_text(s[:i]+'\n## '+TITLE+'\n\n'+body+'\n'+s[i:])
def main():
    names=['HANDOFF.md','171-推理数据流与已查节点.html','171-框架Profile热点.html','171-融合算子全盘排查进度图.html'];(R/'engram-lookup-path-shared-before.json').write_text(json.dumps({n:sha256(Path(n).read_bytes()).hexdigest() for n in names},ensure_ascii=False,indent=2)+'\n')
    f=frontier();(R/'engram-lookup-path-frontier.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n');new=Path('/tmp/engram-lookup-path-next-run.txt').read_text().strip().split('/')[-1];footer='Run210932Z有界PASS/正确性NA，'+new+' PLANNED。公开同步回执待补；loop047活动，无live任务等待。';Path('/tmp/engram-lookup-path-handoff-footer.txt').write_text(footer)
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\n'+footer+'\n')
    for m in ['M00','M01','M11']:insert('audit/modules/'+m+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(R/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT+'\n');insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    for p in [R/'source-frontier.json',Path('audit/first-pass.json')]:
        d=json.loads(p.read_text());d.update(status=f['status'],engram_lookup_path_review=f,next_action=NEXT,whole_chain_complete=False,completeness_established=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name,default in [('171-推理数据流与已查节点.html',False),('171-框架Profile热点.html',True),('171-融合算子全盘排查进度图.html',False)]:
        p=Path(name);old=p.read_text();pat=PATTERN if default else r'<aside id="(?:compressor-path-followup|engram-lookup-path-followup)"[^>]*>.*?</aside>'
        s,n=re.subn(pat,lambda _:banner('audit/execution-search/'+REPORT,default),old,count=1,flags=re.S);assert n==1 and re.sub(pat,'',s,flags=re.S)==re.sub(pat,'',old,flags=re.S);p.write_text(s)
    task=Path('tasks/execution-derived-fusion-search');loop=task/'loops/loop-047'
    for p in [task/'index.json',loop/'state.json']:
        d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for p in [task/'status.md',loop/'status.md']:p.write_text(re.sub(r'^- 下一步：.*$',lambda _:'- 下一步：'+NEXT,p.read_text(),flags=re.M))
    for p in [loop/'journal.md',loop/'report.md']:p.write_text(p.read_text()+'\n\n## Run210932Z · Engram实际hash/UVA主体与累计覆盖（有界PASS/NA）\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一Run '+new+' PLANNED。\n')
    print(json.dumps(dict(current_sources=17,receiver_routes=19,new_tests=0,next_run=new,whole_chain_complete=False)))
if __name__=='__main__':main()
