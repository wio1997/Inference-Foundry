#!/usr/bin/env python3
"""Publish bounded V1 input evidence without erasing broader search gaps."""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import ast,json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-10 V1输入实际读者与packet边界：70源码约束，86位待追'
NEXT='loop047 Run133903Z（PLANNED）：沿继承的V1 _preprocess绑定supports_mm_inputs/配置convertor/embedding真实实现→持久input→首HC消费者；并追globalQSL/物理block-slot经Engram和DSpark per-group视图的全部末读/最早覆写。继续86位及递归helper，不以具名缺口当扫完；已有匹配profile优先，每替换原语逐位一致，无Torch/NPU/模型/新profile/服务动作。'
SUMMARY='介绍：从V1实际caller的默认参数、PP/async、Engram准备字典和QSL输入→packet交付继续扫。到了哪里：16当前Python依赖/11当前原生source；新增21处受限源码绑定，原156位现70绑定/86仍待追；原Engram方法16个stub控制、原base32个owner控制及3个反例、原GenMetadata12个本机C++写范围控制。结果：canonical输出QSL不代表所有闭包输入QSL已共享；5个packet key首owner与每物理组C2地址映射首owner不同。显式重绑incomingQSL的大parent值得深入，尚无原语/框架新准入，未扫全。'
DETAIL='待确认：较大CPU/scalar/canonicalQSL/device长度frame，保留原物理映射/packet owner并核全部读者、输入不可变租约、producer_done+consumer_done及最早写前acquire；逐原语完整bytes、空query/尾部/错误/FULL和真实净价仍缺。原source packet struct只含整数，Gen分别写3600/3456B而wrapper各分配4096B，496/640B外部初始化未闭合，不能默许忽略padding或当NPU实测。MM预处理即使纯文本也可能先embed再copy，配置与实现待下一段绑定。\n\n已有同serve SHA诊断：CP/base/helper互斥self-Free合计311.050016ms/卡/11周期，占独立1025.503ms窗30.3315%，含必要首producer/物理工作，不是全可删。假设实删10%/转移50%/新增0.1ms每frame，条件净1.314ms/frame、镜像配平C8轻量整批1.8924%；默认未知比例0，非实测，不迁移原不同prompt C8 TTFT或六普通，不加旧device-only/CPU原型收益。\n\n已确认可实现：此前受限CPU原型1保留，本轮融合替换0。已停止：删除首全局QSL H2D（中间slot mapping/logits已读），只因batch命中就删各privateQSL而不重绑incoming（SMLA/QLI/C2反例）。这不停止按owner共享及更大frame；86位不是86个算子，递归调用图/全部消费者仍未闭合。'
LINKS='[本段判断](evidence/V1输入到原生packet的owner与完整字节判断.md) · [70/86逐位账](evidence/v1-input-source-review.json) · [owner控制](evidence/v1-qsl-owner-controls.json) · [packet写边界](evidence/v1-packet-write-extent.json) · [整段条件价](evidence/v1-input-parent-price.json) · [限定源行段](evidence/v1-input-source-excerpts.json)。'
BODY=SUMMARY+'\n\n'+DETAIL+'\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
def insert(path,body):
    p=Path(path);s=p.read_text();h='## '+TITLE
    if h in s:
        a=s.index(h);z=s.find('\n## ',a+len(h));z=len(s) if z<0 else z+1;s=s[:a]+h+'\n\n'+body.rstrip()+'\n\n'+s[z:]
    else:
        i=s.find('\n')+1;s=s[:i]+'\n'+h+'\n\n'+body.rstrip()+'\n\n'+s[i:]
    p.write_text(s)
def banner(link,default=False):
    id='v1-input-default-entry' if default else 'v1-input-followup'
    return '<aside id="'+id+'" style="padding:16px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前源码搜索：70受限绑定 / 86位待追，全链路未闭合</strong><p>V1实际输入与5类packet首owner已追；canonical incoming QSL仍缺完整原语字节与双端退役。新框架准入0，311.050ms是含保留工作的旧诊断整段价，非已省时。</p><a href="'+link+'">本段判断、字节边界与下一断点</a></aside>'
def main():
    review=json.loads((ROOT/'v1-input-source-review.json').read_text());price=json.loads((ROOT/'v1-input-parent-price.json').read_text());pack=json.loads((ROOT/'v1-packet-write-extent.json').read_text());assert review['scoped_source_bindings']==70 and review['unresolved_input_or_control_sites']==86
    excerpt_sources=review['sources']+review['prior_exact_SHA_references']
    packet=json.loads((ROOT/'v1-packet-dependencies.json').read_text())
    excerpt_sources += [dict(runtime_path=s['runtime_path'],local_path=s['local_path'],sha256=s['runtime_sha256']) for s in packet['sources']]
    ranges={
      '/worker/model_runner_v1.py':[[501,507],[708,709],[750,764],[1060,1107],[1435,1462],[1505,1626],[1635,1679],[2240,2277],[2475,2509],[2551,2617],[2634,2678],[3169,3270],[3454,3467],[3521,3615],[3653,3907],[6128,6139]],
      '/v1/worker/gpu_model_runner.py':[[532,560],[568,576],[716,718],[1090,1101],[1712,1734],[3502,3655],[3840,3853]],
      '/models/deepseek_v41/model.py':[[1098,1252],[1271,1317],[1332,1373],[1392,1608],[1696,1724]],
      '/models/deepseek_v41/vl_model.py':[[45,75],[179,250],[261,275]],
      '/attention/context_parallel/dsa_v41_cp.py':[[48,167]],
      '/attention/dsa_v41.py':[[879,927],[945,1079],[1161,1279],[1287,1349],[1390,1425]],
      '/worker/device_metadata.py':[[49,64],[74,155]],
      '/config/parallel.py':[[122,129],[222,225],[353,355]],'/config/model.py':[[875,918],[1515,1517]],'/vllm/envs.py':[[1575,1583]],'/config/speculative.py':[[1941,1942]]}
    indexes=[];excerpts=[]
    for s in excerpt_sources:
        raw=Path(s['local_path']).read_bytes();assert sha256(raw).hexdigest()==s['sha256'];L=raw.decode().splitlines();rr=[]
        for suffix,rs in ranges.items():
            if s['runtime_path'].endswith(suffix):rr+=rs
        if '/csrc/' in s['runtime_path']:
            name=Path(s['runtime_path']).name
            if name in ['sparse_flash_mla_kernel_metadata.h','quant_lightning_indexer_v2_metadata.h'] or '/op_api/' in s['runtime_path']:rr=[[1,len(L)]]
            elif name.endswith('_torch_adpt.h'):rr=[[18,min(90,len(L))]]
            elif name.endswith('_aicpu.cpp'):
                text=raw.decode();cls='SparseFlashMlaMetadataCpuKernel' if 'sparse_flash_mla_metadata' in name else 'QuantLightningIndexerV2MetadataCpuKernel';a=text.index('bool '+cls+'::GenMetadata(');b=text.index('\nnamespace {',a);rr=[[text[:a].count('\n')+1,text[:b].count('\n')]]
        indexes.append(dict(runtime_path=s['runtime_path'],sha256=s['sha256'],binding_read_ranges=rr,qualification='Listed binding/serializer/output-API ranges only; empty ranges identity-only, never native or full-body coverage.'))
        for a,z in rr:
            assert 1<=a<=z<=len(L),(s['runtime_path'],a,z,len(L));excerpts.append(dict(runtime_path=s['runtime_path'],sha256=s['sha256'],first_line=a,last_line=z,lines=L[a-1:z]))
    (ROOT/'v1-input-source-excerpts.json').write_text(json.dumps(dict(sources=indexes,excerpts=excerpts,new_primitive_admissions=0,whole_chain_complete=False),ensure_ascii=False,indent=2)+'\n')
    front=dict(status='ACTIVE_LOOP047_V1_INPUT_OWNER_PACKET_BOUNDARY_70_SOURCE_BINDINGS_86_UNCLOSED',classification='待确认，继续全链路',current_Python_source_pins=16,current_native_source_pins=11,prior_exact_SHA_references=1,original_static_sites=156,scoped_source_bindings=70,unclosed_input_control_sites=86,engram_handoff_stub_controls=16,base_owner_symbolic_controls=32,incoming_QSL_negative_controls=3,host_serializer_extent_controls=12,whole_callgraph_closed=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,isolated_CPU_prototypes_passed=1,worth_deepening=review['candidate'],stopped_scopes=review['stopped'],gross_parent_Free_ms=price['mean_three_function_self_Free_ms'],gross_parent_is_net_saving=False,remaining=review['remaining']+['Strict complete native packet bytes remain unproved: serializer-only controls expose496/640B outside struct, no inference that runtime leaves them uninitialized.'],next_action=NEXT,evidence=['v1-input-source-review.json','v1-qsl-owner-controls.json','v1-packet-write-extent.json','v1-input-parent-price.json','v1-input-source-excerpts.json'],whole_chain_complete=False)
    (ROOT/'v1-input-frontier.json').write_text(json.dumps(front,ensure_ascii=False,indent=2)+'\n')
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nloop047仍EVALUATING。Run124045Z已PASS/正确性NA，仅本段源码/显式stub与CPU serializer控制；Run133903Z仍PLANNED，无live任务等待。公开同步回执待补。')
    for n in ['M00','M01','M03','M08','M09','M10','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT);insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    insert('ENVIRONMENT.md','适用冻结910C/CANN9.2/DP2TP8EP16PP1，2026-11-10核实171hostname与同owned容器/image。只读16份Python依赖（11旧SHA匹配/5新body）与11份native依赖（8匹配/3新body）。SMLA L2路径初猜缺op_host已由目录tar清单纠正，重取完成；不是环境版本变化。27身份不等于27份完整审阅。仅本机stdlib和clang++执行显式控制，无Torch/Transformers/NPU/模型导入、远端写入/请求/profile/服务动作。running非health/loaded binary证明；完整native packet仍待确认。')
    for p in [ROOT/'source-frontier.json',Path('audit/first-pass.json')]:
        d=json.loads(p.read_text());d['v1_input_owner_packet_review']=front;d['status']=front['status'];d['next_action']=NEXT;d['whole_chain_complete']=False;d['completeness_established']=False;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for f,default in [('171-推理数据流与已查节点.html',False),('171-融合算子全盘排查进度图.html',False),('171-框架Profile热点.html',True)]:
        p=Path(f);s=p.read_text();b=banner('audit/execution-search/V1输入到原生packet的owner与完整字节判断.md',default)
        pattern=r'<aside id="(?:whole-forward-default-entry|v1-input-default-entry)".*?</aside>' if default else r'<aside id="(?:whole-forward-followup|v1-input-followup)".*?</aside>'
        s,n=re.subn(pattern,lambda _:b,s,count=1,flags=re.S)
        if not n:s,n=re.subn(r'(<body[^>]*>)',lambda m:m.group(1)+b,s,count=1)
        assert n==1;p.write_text(s)
    for p in [Path('tasks/execution-derived-fusion-search/index.json'),Path('tasks/execution-derived-fusion-search/loops/loop-047/state.json')]:
        d=json.loads(p.read_text());d['next_action']=NEXT
        if p.name=='state.json':d['status']='EVALUATING'
        p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    p=Path('tasks/execution-derived-fusion-search/loops/loop-047/status.md');s=p.read_text();s=re.sub(r'^- 状态：.*$','- 状态：EVALUATING',s,flags=re.M);s=re.sub(r'^- 下一步：.*$',lambda _: '- 下一步：'+NEXT,s,flags=re.M);p.write_text(s)
    print(json.dumps(dict(source_pins=27,prior_exact_SHA_refs=1,bounded_excerpts=len(excerpts),sites=156,source_bindings=70,unclosed=86,new_framework_admissions=0)))
if __name__=='__main__':main()
