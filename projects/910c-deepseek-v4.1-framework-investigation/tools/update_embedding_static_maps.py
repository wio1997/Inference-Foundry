#!/usr/bin/env python3
"""Write one bounded source/delivery delta, preserving other maps and cohorts."""
from pathlib import Path
from hashlib import sha256
import json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-10 实际embedding入口与首层静态段：77源码约束，79位待追'
NEXT='loop047 Run150110Z（PLANNED）：继续79位及递归helper，从selected资源组构造走Engram V1 slot-cache hash输入、DSpark公共/每组输出到全部异步末读和最早storage写；含空/synthetic DP、FULL与错误退役。优先大input-frame交付，保留DP/TP与physical first-owner次序；先复用旧记录核人口/净价，每替换原语逐位一致，无Torch/NPU/模型/新profile/请求/服务或远端变更。'
SUMMARY='介绍：独立沿V1预处理→OOT注册→STANDARD TP8/FLOAT查表→持久input→首HC/QKV及DSpark公共/物理地址交付，再与旧范围相交。到了哪里：23当前Python源+1prior HF精确SHA，29原方法受限控制；新增7处，原156位77绑定/79待追；原C8 3624活跃rank窗的Qb/KVnorm末端全资格。结果：eager用外部embeds，FULL捕获rawIDs/None并执行内部lookup；SP mask入口等待与Engram producer仍属大交付。新原语/框架准入0，全链路未闭合，每替换原语逐位一致。'
DETAIL='待确认、值得深入：大input/Engram/graph-entry交付原C8活跃请求gross2.356753%TTFT/5.058344%整批，含必要工作不能全删；实删且暴露25%、新增0.1ms/波，条件净0.555673%/1.142482%，50%为1.144861%/2.407068%。未知比例默认0，非已兑现。首HC到pre-RoPE单独gross0.710251%/0.456940%，不再含空query动态Oproj等待；静态cache仍缺native精确位/shape-key/热行成本/六普通整段价。453活跃DP组不等于454物理组，synthetic8rank未价；不迁移并发padding行比例或另cohort诊断。\n\nDSpark公共IDs/positions/sampleindices不依赖物理组，slot依赖group table/size/DCP；5query为anchor+4noise，cad属性重绑不是旧Tensor退役。实际draft组人口/次序、全部末读、producer/consumer双退役和packet完整bytes继续待确认。已确认可实现：本轮0，旧封闭CPU租约原型1资格保留。已停止：旧lookup/copy/SP-only同预算小方案继续停；不以纯文本排除MM、不以runtime embeds推断FULL内部lookup跳过、不按5×实际请求认shape、不以BF16尾位容差放行。较大parent保留。'
LINKS='[本段判断](evidence/词嵌入实际派发与首层静态计算整段判断.md) · [77/79逐位账及29控制](evidence/embedding-static-source-review.json) · [原C8整段价](evidence/first-static-parent-price.json) · [源行段](evidence/embedding-static-source-excerpts.json)。'
BODY=SUMMARY+'\n\n'+DETAIL+'\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
ASIDE_PATTERN=r'<aside id="(?:whole-forward-default-entry|v1-input-default-entry|embedding-static-default-entry)"[^>]*>.*?</aside>'
def banner(link,default=True):
    ident='embedding-static-default-entry' if default else 'embedding-static-followup'
    return '<aside id="'+ident+'" style="padding:16px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前源码搜索：77受限绑定 / 79位待追，全链路未闭合</strong><p>实际TP8/FLOAT embedding与eager/FULL入口已分开。大input/Engram交付仍值得深入；原C8实删暴露25%且新增0.1ms/波的条件整批净价1.14%，非实测。453活跃DP组与454物理组分开。新原语/框架准入0，每替换原语逐位一致。</p><a href="'+link+'">整段判断、资格与下一断点</a></aside>'
def insert(path,body):
    p=Path(path);s=p.read_text();h='## '+TITLE;assert h not in s
    i=s.index('\n')+1;p.write_text(s[:i]+'\n'+h+'\n\n'+body+'\n'+s[i:])
def frontier():
    review=json.loads((ROOT/'embedding-static-source-review.json').read_text());price=json.loads((ROOT/'first-static-parent-price.json').read_text())
    return dict(status='ACTIVE_LOOP047_EMBEDDING_STATIC_ENTRY_77_SOURCE_BINDINGS_79_PENDING',classification='待确认，继续全链路',current_source_pins=23,prior_exact_SHA_references=1,original_static_sites=156,scoped_source_bindings=77,unclosed_input_control_sites=79,original_method_stub_controls=29,whole_callgraph_closed=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,isolated_CPU_prototypes_passed=1,
        candidates=[dict(status='待确认',scope='large input-frame / Engram / graph-entry delivery',stage='source dispatch and matched active-request old-C8 paid sensitivity',gaps=['actual producer release and complete lastread/earliest overwrite','DP/TP/physical group order and FULL/error retirement','ordinary-load whole-parent price and net costs'],gross_original_C8_percent={m:v['signed_percent'] for m,v in price['gross']['joint_first_static'].items()}),dict(status='待确认',scope='shape/lane/rank-keyed firstHC/preRoPE static cache or precomputation',stage='complete selected preRoPE source boundary and3624 active-rank exact task endpoints',gaps=['native primitive bits','hot-row memory/maintenance/population','synthetic DP and sixordinary whole-static prices']),dict(status='待确认',scope='DSpark common coordinates once plus physical group mapping',stage='original kernel source dependency graph',gaps=['actual group multiplicity and order','all consumers and leases','complete native integer/tail/error output and paid net'])],
        stopped=['old independent lookup/copy/SP-only same-budget child','puretext implies noMM/captured internal lookup skipped fromruntimeembeds','all5query noise / cad rebinding means storage retirement / BF16 last-bit tolerance'],
        qualification='Counts concern original156static sites only, not operators/full source paths/native population.29 scalar/object-stub source controls only.453 active-request DPgroups of454 physical groups priced; no synthetic-group attribution, C8percent transfer or summing othercohort padding/diagnostic. No newTorch/NPU/model/profile/request/service or remote writes.',next_action=NEXT,evidence=['词嵌入实际派发与首层静态计算整段判断.md','embedding-static-source-review.json','embedding-static-source-excerpts.json','first-static-parent-price.json','first-static-math-receipt.json'],whole_chain_complete=False)
def main():
    front=frontier();(ROOT/'embedding-static-frontier.json').write_text(json.dumps(front,ensure_ascii=False,indent=2)+'\n')
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nRun133903Z PASS/正确性NA仅本段源码/旧数据资格；Run150110Z PLANNED，无live任务等待。loop047保持EVALUATING。公开同步回执待补。\n')
    for n in ['M00','M01','M02','M03','M10','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT+'\n');insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    for p in [ROOT/'source-frontier.json',Path('audit/first-pass.json')]:
        d=json.loads(p.read_text());d.update(embedding_static_entry_review=front,status=front['status'],next_action=NEXT,whole_chain_complete=False,completeness_established=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for f,default in [('171-推理数据流与已查节点.html',False),('171-融合算子全盘排查进度图.html',False),('171-框架Profile热点.html',True)]:
        p=Path(f);s=p.read_text();old=s;pattern=ASIDE_PATTERN if default else r'<aside id="(?:v1-input-followup|whole-forward-followup|embedding-static-followup)"[^>]*>.*?</aside>'
        s,n=re.subn(pattern,lambda _:banner('audit/execution-search/词嵌入实际派发与首层静态计算整段判断.md',default),s,count=1,flags=re.S);assert n==1
        assert re.sub(pattern,'',old,flags=re.S)==re.sub(pattern if default else pattern,'',s,flags=re.S);p.write_text(s)
    for p in [Path('tasks/execution-derived-fusion-search/index.json'),Path('tasks/execution-derived-fusion-search/loops/loop-047/state.json')]:
        d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for p in [Path('tasks/execution-derived-fusion-search/loops/loop-047/status.md')]:
        s=p.read_text();s=re.sub(r'^- 下一步：.*$',lambda _:'- 下一步：'+NEXT,s,flags=re.M);p.write_text(s)
    p=Path('tasks/execution-derived-fusion-search/loops/loop-047/report.md');p.write_text(p.read_text()+'\n\n## Run133903Z · 实际embedding/首层静态边界（有界PASS/NA）\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一Run150110Z仍PLANNED，继续未闭合物理组/全部末读/原79位与递归helper；全链路未完成。\n')
    print(json.dumps(dict(source_bindings=77,pending=79,whole_chain_complete=False)))
if __name__=='__main__':main()
