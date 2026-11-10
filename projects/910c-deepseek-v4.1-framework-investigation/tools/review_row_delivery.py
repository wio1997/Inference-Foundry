#!/usr/bin/env python3
"""Record real row provenance, two mask contracts and qualified old prices."""
from pathlib import Path
from hashlib import sha256
from statistics import mean
import json

ROOT=Path('audit/execution-search')
REPORT='输入尾部与MC2有效行及MoE完整输出判断.md'
TITLE='2026-11-11 输入尾部到原生DFC：两套mask不同，padding省算不可重复计价'
SUMMARY='介绍：沿target/draft输入尾部追实际MoE有效行交付。到了哪里：30当前源身份，38原Python前缀/尾部/MC2行控制；从actual前缀到SP切片、量化payload和原生DFC mask写者。结果：target旧尾部可保留，DSpark query有真实用途；DFC已有独立有效行处理。原156位仍122受限绑定/34动态待追，全链路未闭合，新原语/框架准入0，每替换原语逐位一致。'
DETAIL='已停止：按position=0/ID=0或图shape认整行恒零；按5个DSpark query只保留anchor；把router未读is_padding推成DFC没有有效行mask，再重复计算“新增跳过padding expert”的收益；盲删空DP整卡执行。target只写actual ID/embedding前缀，clamp只改负数，positions尾部清零不能清除非负旧ID/旧embedding。DSpark eager固定，query尾部显式填parallel drafting ID/position0/slot-1，与SP新补零行不同。\n\n待确认：大row交付保留，但拆开已存在的mc2_mask与model is_padding。target实际count、draft实际query count经forward context生成mc2_mask，MC2按本卡原SP行再补DP长度，W8A8 payload传到DFC x_active_mask。当前DFC源核在init routing前把无效行expertIdx storage改为expertNum哨兵；不是-1。原生变异、expert counts、完整routed输出、shared输出/归约/scale/add、HC残差/系数、aux/MTP和下一轮租约都属于替换边界义务。torch schema仅声明out/count可变，源码kernel却写expertIdx；包装层到loaded op实际alias仍须核实，不能据此宣称已观测caller Tensor变异。\n\n旧EP0/8 localM3窗的DFC全免均16.758165%、norm→Post全免均22.504527%；它们包含已有mask和必要数学，不是padding行可省比例。保留DFC必需时，parent span减DFC的极乐观包络均5.746362%（两代表卡，global24/480calls），仍含必要norm/router/shared/HC和等待，不能全计为候选收益。新row请求收益信用默认0；六普通及原C8请求曝光/净价待绑，不迁移不同cohort。\n\n继续深入的边界：真实AscendMoERunner的shared/routed reduction→scale→foreach_add→HC Post整体交付，保持每一处BF16/FP32舍入和事件/collective顺序；不先假定把多个数学式合并就逐位相同。把已有mask可省之外的具体工作与完整原语输出闭合后，才给出开发准入。大metadata/gid12和Engram owner/input候选保留。本轮仅源码、行tag与旧数据，无新Torch/NPU/模型/profile/请求/服务/远端写入。'
LINKS='[本段判断](evidence/'+REPORT+') · [buffer与真实消费者](evidence/row-delivery-review.json) · [38原源码控制](evidence/row-delivery-controls.json) · [分cohort价格](evidence/row-delivery-price.json)。'
NEXT='loop047 Run182623Z PLANNED：继续M06真实AscendMoERunner.forward/_forward_impl→shared_experts各执行/等待/归约→scale/foreach_add→HC Post的完整输出；保留原生DFC有效行、expertIdx潜在变异/count与全部EP参与。先走较大交付再用匹配旧profile拆必要工作/新增开销/请求曝光，若整段乐观低才剪同预算子项。然后转下一未扫递归消费者；34动态原位/全callee仍未闭合，不重复122冻结约束或已停小配方。每替换原语逐位一致，无新Torch/NPU/模型/profile/请求/服务或远端变更。'

def write(name, value): (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def main():
    sources=json.loads((ROOT/'row-delivery-dependencies.json').read_text())['sources']
    assert len(sources)==30
    for s in sources: assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
    controls=json.loads((ROOT/'row-delivery-controls.json').read_text());assert controls['controls']==38
    previous=json.loads((ROOT/'model-gate-source-review.json').read_text());assert previous['scoped_source_bindings']==122 and previous['unresolved_input_or_control_sites']==34
    # Original source-first whole row parent predates these historical envelopes.
    primary=ROOT/'model-pad-row-parent-primary.json'
    old=ROOT/'dfc-input-delivery-price.json';profile=json.loads(old.read_text());groups=[]
    for p in profile['profiles']:
        for g in p['groups']:
            assert g['parent_sum_us']>=g['DFC_sum_us']
            rest=g['parent_sum_us']-g['DFC_sum_us']
            groups.append(dict(ep=p['ep'],global_rows=g['global_rows'],local_M=g['local_M'],calls=g['calls'],window_us=p['window_us'],DFC_us=g['DFC_sum_us'],parent_us=g['parent_sum_us'],nonDFC_span_optimistic_us=rest,nonDFC_span_optimistic_pct=rest/p['window_us']*100))
    prices=[]
    for T in [20,24]:
        rows=[g for g in groups if g['global_rows']==T];assert len(rows)==2 and {g['ep'] for g in rows}=={0,8}
        vals=[]
        for eta,fee in [(0,5),(.25,5),(1,0)]:
            nets=[g['nonDFC_span_optimistic_us']*eta-g['calls']*fee for g in rows]
            vals.append(dict(fraction_of_nonDFC_containing_span_removed=eta,added_nonoverlap_us_each=fee,mean_net_us=mean(nets),mean_conditional_window_pct=mean(n/g['window_us']*100 for n,g in zip(nets,rows))))
        prices.append(dict(global_rows=T,representative_ep=[0,8],calls_per_rank=sorted({g['calls'] for g in rows}),nonDFC_optimistic_window_pct_mean=mean(g['nonDFC_span_optimistic_pct'] for g in rows),scenarios=vals))
    price=dict(status='OLD_TWO_RANK_ENVELOPES_AFTER_EXISTING_MASK_SOURCE_REVIEW',groups=groups,summaries=prices,request_credit=0,
               qualification='Only two old EP0/8 localM3 shape windows. Parent-minus-DFC is a relaxed maximum if required DFC time remains on that same span; not measured scale/add/HC/shared removable cost. Includes required work and waits. Not all16 ranks, full request, originalC8/sixordinary transfer or actual padded-row population. eta is time removed, never padded-row fraction. No repeated credit for already-existing mask handling.',
               inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [primary,old]],whole_chain_complete=False,new_primitive_admissions=0,new_framework_replacement_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
    write('row-delivery-price.json',price)
    domains=[
      dict(domain='target actual prefix',writer='NPUModelRunner._prepare_inputs:1306-1609 + inherited _prepare_input_ids:1790-1924',value='current CPU gathered IDs / previous sampled IDs / previous draft scatter; corrected GPU position',consumer='GPUModelRunner._preprocess and target full/eager model input',gap='Full capture entry/read-address and all error/lease retirement remain; source cases are not observed frame population.'),
      dict(domain='target FIA/graph/DP tail',writer='actual-only CpuGpuBuffer.copy_to_gpu:139-143; sanitize placeholder:1737-1752; _preprocess:3513-3656',value='nonnegative old IDs or old external embedding may remain; position tail zero; graph internal lookup differs from eager external embedding',consumer='target model hidden/HC, Engram/hash selected reads, attention padded metadata',gap='No claim every tail reaches same consumer; graph/eager chosen bytes, history and retained outputs need exact binding.'),
      dict(domain='DSpark real query',writer='AscendDSparkProposer:226-357 / copy_and_expand kernel:126-202',value='K=5 source default sample_from_anchor=True: query0 bonus +4 parallel IDs, five positions and sample indices; alternate False uses K+1 queries',consumer='_run_merged_draft:1476-1534 -> full draft layers -> gathered selected states/LMHead',gap='Do not discard4noise as padding. Alternate config legal source leg, frozen currentHF selection inherits previous model/source evidence.'),
      dict(domain='draft DP padding',writer='_propose:1093-1114,1223-1227; _pad_draft_buffers:2653-2669',value='parallel draft ID / position0 / query slot-1; whole context suffix slot-1; no hidden/external-embed writer in that method',consumer='DeepseekV41DSparkModel.forward raw ID embedding, local HC; mc2 active mask uses pre-padding num_tokens',gap='Different from target old tail and SP modulo zeros; preserve cached physical views.'),
      dict(domain='SP modulo padding',writer='existing source-pinned sp_shard + target:1541-1551 / draft:251-261',value='only extra rows beyond model shape are newly zero-filled; clone behavior when aligned preserved',consumer='HC, normalization/router, shared/routed; mc2 mask separately excludes inactive actual domain',gap='Zero mathematical input alone is no native bit proof; downstream coefficients, outputs, counts and leases remain.'),
      dict(domain='MC2 valid rows',writer='set_ascend_forward_context:222-232; MC2.prepare:272-314',value='True before actual count, False suffix; slice original SP shard then DP-pad false',consumer='routed_experts:717-750 -> W8A8.apply:218-234 -> payload.mc2_mask -> fused DFC:590-608 / CANN MegaMoe alternative:514-551',gap='model is_padding and mc2_mask are separate contracts. Original controls prove geometry only. Native caller alias, selected loaded op and packet/bits remain.'),
      dict(domain='native DFC mask mutation',writer='dispatch_ffn_combine_kernel.hpp:360-399,799-807',value='inactive expertIdx storage overwritten with expertPerRank*EP before init routing, followed by sync',consumer='routing/group-list/EP GMM and combine; expert_token_nums output',gap='Python schema declares only out/count mutable. Adapter/API forward input; runtime op executor/format transformations and caller-visible alias must be verified, not inferred. Other native internal routing/unpermute branches not fully audited here.'),
      dict(domain='retained whole MoE/HC/aux',writer='AscendMoERunner.forward:215-246; _forward_impl:485-540; decoder:879-920; target:1556-1607',value='shared/routed selected reductions+scale+foreach_add -> HC Post; residual/pre_mix and aux checkpoints retained',consumer='next layer, aux DSpark context, MTP buffer and final collapse/norm, next iteration consumers',gap='Still must follow inherited reduction/scale hooks, all shared streams and quant/reduction boundaries. Cannot zero whole row or skip an empty EP rank with remote work.')]
    result=dict(status='ROW_WRITERS_AND_NATIVE_MASK_REVIEWED_CONTINUE_MOE_WHOLE_OUTPUT',sources=sources,row_domains=domains,controls=38,source_bindings=122,pending_original_sites=34,new_original_bindings=0,
                classification=dict(待确认=['MoE shared/routed reduction-scale-add-HC Post parent, complete output/side effects and matched residual price.','Larger row/caching/owner parent after existing active mask, needing full shape/history/bit keys and population.','Existing metadata/gid12 and Engram input-owner parent.'],已确认可实现=[],已停止=['Blind zero/position0/ID0/static-shape padding recipes.','Delete four real DSpark queries.','Count already-existing DFC active mask as a new skip-padding implementation.','Skip full empty-DP EP rank or alter expert IDs to -1.']),
                whole_callgraph_closed=False,whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],
                harness_corrections=['Initial row stub omitted self.positions for the original preprocess statement; added alias.','Initial MC2 method extraction omitted QuantType.NONE default; added opaque enum default. Both were harness errors before any evidence result; no native execution/admission.'],
                inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [ROOT/'row-delivery-dependencies.json',ROOT/'row-delivery-controls.json',ROOT/'row-delivery-price.json',ROOT/'model-gate-source-review.json',primary]],reviewer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    write('row-delivery-review.json',result)
    body=SUMMARY+'\n\n'+DETAIL+'\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
    table='\n| 行域 | 实际交付 | 已追消费者 / 缺口 |\n|---|---|---|\n'+''.join('| '+r['domain']+' | '+r['value']+' | '+r['consumer']+'；'+r['gap']+' |\n' for r in domains)
    (ROOT/REPORT).write_text('# 输入尾部、MC2有效行与MoE完整输出\n\n'+body.replace('(evidence/','(')+table+'\n30身份不等于30全文审阅；38控制均为整数/布尔/opaque行tag，不是native逐位对拍。原122/34账未新增绑定，递归callee覆盖另列，不伪造全链路完成。CANN MegaMoe替代leg及本地DFC核分列，旧DFC任务证明历史人口，不证明当前loaded op。\n')
    print(json.dumps(dict(sources=30,controls=38,bound=122,pending=34,price=prices,new_admissions=0),ensure_ascii=False))

if __name__=='__main__': main()
