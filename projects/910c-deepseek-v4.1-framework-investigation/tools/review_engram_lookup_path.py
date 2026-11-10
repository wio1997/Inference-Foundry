#!/usr/bin/env python3
"""Reconcile actual hash/UVA bodies with exact-SHA prior semantics and value."""
from pathlib import Path
from hashlib import sha256
import ast,json
R=Path('audit/execution-search')
TITLE='2026-11-11 Engram实际hash与UVA内核：补齐覆盖，完整父段继续保留'
REPORT='Engram实际hash与UVA查表到双层残差的覆盖判断.md'
SUMMARY='介绍：从实际prepare_engram调用追V1 history写入、两层hash、共享UVA查表/反量化、TP heads交付，到SP行上的Wkv/rotated gate与后继HC。到了哪里：17当前源码身份；补齐实际适配器、上游history和Triton hash/lookup主体，并与exact-SHA旧合同去重；旧两代表卡24完整双层交付定价。结果：整段native包络1.954322%代表窗，不能停止整个父段；lookup本体0.031226%、私有预零0.008845%，同预算小方案不重测。该父段与机制已有旧记录，本轮是覆盖/组合价补全，不冒充新发现。新增原生准入0，全链路未闭合。'
DETAIL='''实际调用与状态：DeepseekV41Model.prepare_engram一次hash/gather供层1/14，再直接table.embed_gathered；Embedding.forward不是这处服务入口。shared-memory=true令table.dp_size=1，绕过每步DP hash gather/rowexchange；TP8每rank3个head，仍需完整head-order交付再SP切行。V1 adapter继承NgramHashState.ensure_cache：未绑KV返回false并丢history，同一live KV对象保留，换对象重置；request-index一次供两层，独立_write_hash_cache先写mapped/dead token，再hash读取。当前chunk>known lookback>slot fallback，known dead会阻断且不回退；阻断对后续shift保持。层/head multipliers、prime和offset不同，不能共用两层hash结果。合hash/lookup或结果cache仍须保存所有未来history读者和原INT64溢出/XOR/modulo次序；普通program顺序不能替代旧kernel完成屏障。

UVA与反量化的完整边界：当前共享映射每rank各自CANN注册、构造INT64 chunk指针；leader只在全员attach/失败汇总及barrier后unlink。codes/scales是独立映射，关闭成功后才撤对应alias，unregister失败保留可重试owner。指针表不是每token建立。lookup将ID先转INT64，按全局vocab区间判断归属，减vocab_start后按2^22行chunk定位；INT8/FP32 group32读数、乘法和BF16舍入已在一个内核。invalid ID安全读row0，最终select为+0；不能改成0乘可能NaN的结果。选中pad_heads=local_heads，全部本地head每次写完，局部out的zeros可作为冗余候选；通用wrapper仅写本地head前缀，不能删pad尾/固定buffer尾零化或初始化表零。两表、两层Wkv及其结果均不同，已有DP共享/反量化/ID widening不得重复记收益。

数值与消费者：BF16 rows先TP heads收齐，再SP按原行/填充规则交原M3、K6144、N25600的BF16 FLOAT Wkv，随后原FP32 block32旋转/RMS、左结合dot、signbit/sqrt/sigmoid、mask、乘V加hidden、最终BF16四路残差。原Wkv BF16边界和最后残差仍有HcPost读者。直接K分片再求和、擦掉舍入、mask false直通、stock AIVAGMM或改变M/N/tiling的失败配方不复跑。原数值panel/N-owner复用B与直接SP-owner去掉全局bank是不同布局，需要的新交换与生存期不能同时免费。本轮匹配旧hash/lookup/state合同SHA并复用其有限检查；没有新数值验收。旧M3/M15常量/hidden widening复用资格仍只限其fixture，不能扩大到全父段。

完整价格：旧配平C8 EP0/8、global24/local3中24完整双层分支，含48个Wkv+37gate链。整段native均1.954322%窗口，原Wkv0.717463%，其余1.236860%；mask/hash/history0.200640%，lookup0.031226%，private zeros0.008845%，hash slice0.009031%，TP AG+transpose0.557507%。同起止到层14 HC跨度19.915867%，夹着13层必要target工作，不能当融合收益。TP交付native也含arrival wait，不是可删纯传输。若保留原Wkv、只省其余50%并另付20us/完整双层交付，条件净5.357–5.724ms/0.569763–0.615753%窗口；付50us为0.531473–0.577029%。这些是敏感性情景，不能外推请求或六普通。

旧负载另列：原C8 lookup全部免的固定模型P90 TTFT条件信用最高0.118045%、makespan0.046259%；旧Wkv/gate/copy全部免最高1.115388% TTFT，非全Engram上界。原owner-only六普通tuned约0.427–0.929%，default0.323–0.726%；联合input+gate+external tuned0.968–2.073%含其它父段，不相加。查表/compact scale/预零/hash小helper、旧独立gate与readiness不变tuple cache、原样overlap/staged prefix均沿其原停止预算；当前1.954322%大包络并不替代这些旧子范围，也不足以证明任何新实现会达到净收益。

归档去重：旧loop012的Engram完整交付报告已记录同一大父段、history/invalid/padding/alias合同、容量路线和native位反例；本轮补实际函数/内核与原递归worklist的精确交集及代表窗组合价。待确认仍是较大input/owner/HC/iteration与NUMA/capacity：原vendor M/N/K tile和Fixpipe逐位路径、全部新packet/collective/pack成本、全16rank ready与资源竞争、persistent bank最后读者/失败退役、六普通和请求曝光。INT8保留E8M0的17.16665GiB物理共享host容量路线仍按旧CPU穷举资格和NPU/成对加载/映射缺口保留；纯lookup延迟小预算停止。未知请求信用为0。'''
NEXT='loop047下一Run：先把旧源全文范围、当前实际receiver与aux/Compressor/Engram侧表合成累计递归覆盖前沿，保留每项的精确SHA/分支/消费者/历史价及未闭合原因，避免重复扫已有主体。再从未归位的AscendMoERunner.apply_routed_input_transform、_maybe_reduce_*、shared/routed transform与最终combine实际消费者继续，绑定冻结SEDP/DFC与合法替代分支；按完整输出交付审共享、数学/量化/布局和megakernel，再用已有同负载价。不要用319节点、34输入谓词或具名缺口声称全链路闭合；每替换原语逐位一致。本源码Run无新Torch/NPU/模型/profile/请求/服务/远端写入。'
WALKS={
 'models/deepseek_v41/engram/embedding.py':{'AscendParallelEngramEmbedding':['__init__','_get_shard_info','_allocate_weights','close_host_offload','embed_gathered','wait_lookup','forward','lookup'],None:['_torch_lookup']},
 'models/deepseek_v41/engram/hash_state.py':{'AscendEngramSlotCache':['__init__','kv_cache'],'AscendNgramHashState':['forward'],None:['engram_dead_mask','create_engram_hash_state']},
 'models/deepseek_v41/engram/npu.py':{None:['gather_dequantize_engram_int8','gather_dequantize_host_uva'],'SharedUvaBuffer':['__init__','_unlink','close']},
 'models/deepseek_v41/engram/parallel.py':{None:['resolve_dp_shared_memory','engram_gathered_num_tokens','gather_engram_hashes','exchange_engram_rows']},
 'models/deepseek_v41/engram/layer.py':{'AscendEngram':['__init__','forward']},
 'models/deepseek_v41/engram/common.py':{None:['engram_gate']},
 'models/deepseek_v41/model.py':{'DeepseekV41Model':['prepare_engram','prepare_engram_inputs','_can_overlap_engram_preparation'],'DeepseekV41DecoderLayer':['forward']},
 'ops/triton/engram_lookup.py':{None:['_decode_e8m0','_engram_int8_gather_dequant_kernel','_engram_host_uva_gather_dequant_kernel']},
 'ops/triton/engram_hash.py':{None:['_engram_req_index_kernel','_hash_ids_kernel']},
 'common/engram.py':{'NgramHashState':['__init__','ensure_cache','dummy_hashes'],'EngramLayout':['__init__','from_config'],None:['_write_hash_cache_kernel']},
}
def main():
    deps=['engram-lookup-path-dependencies.json','engram-hash-path-dependencies.json','engram-hash-kernel-dependencies.json'];pins={s['runtime_path']:s for n in deps for s in json.loads((R/n).read_text())['sources']};assert len(pins)==17
    for pin in pins.values():assert sha256(Path(pin['local_path']).read_bytes()).hexdigest()==pin['runtime_sha256']
    excerpts=[]
    for suffix,owners in WALKS.items():
        pin=next(p for p in pins.values() if p['runtime_path'].endswith('/'+suffix));raw=Path(pin['local_path']).read_text();ls=raw.splitlines();tree=ast.parse(raw)
        for owner,names in owners.items():
            parent=tree if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner)
            for name in names:
                f=next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name==name);excerpts.append(dict(runtime_path=pin['runtime_path'],sha256=pin['runtime_sha256'],owner=owner,name=name,first=f.lineno,last=f.end_lineno,lines=ls[f.lineno-1:f.end_lineno],qualification='Bounded source semantics walked; exact-SHA old overlap explicitly reused. Not all native implementations/populations or a new mechanism.'))
    pin=next(p for p in pins.values() if p['runtime_path'].endswith('/models/deepseek_v41/model.py'));ls=Path(pin['local_path']).read_text().splitlines()
    for name,lo,hi in [('table_and_hash_construction',1017,1084),('actual_Engram_residual_consumers',1522,1592)]:excerpts.append(dict(runtime_path=pin['runtime_path'],sha256=pin['runtime_sha256'],owner='DeepseekV41Model',name=name,first=lo,last=hi,lines=ls[lo-1:hi],qualification='Explicit partial range; never claims entire constructor/forward.'))
    (R/'engram-lookup-path-source-excerpts.json').write_text(json.dumps(dict(excerpts=excerpts,whole_chain_complete=False),ensure_ascii=False,indent=2)+'\n')
    work=json.loads((R/'recursive-source-worklist.json').read_text());nodes={n['id']:n for n in work['nodes']};intersections=[]
    for n in work['nodes']:
        matches=[e for e in excerpts if e['runtime_path']==n['runtime_path'] and e['sha256']==n['sha256'] and e['first']<=n['first'] and e['last']>=n['last']]
        if matches:intersections.append(dict(node=n['id'],first=n['first'],last=n['last'],sha256=n['sha256'],source_excerpts=[e['name'] for e in matches],body_state='BOUNDED_SOURCE_WALK_WITH_EXACT_SHA_HISTORY_REUSE',all_recursive_callees_closed=False))
    rules={'self.embed_gathered':'Embedding.forward helper exists; actual model calls table.embed_gathered directly per layer','table.embed_gathered':'Actual prepare_engram per-table caller; selected sharedDP1 thenTP8 AG and fullorderedheads','table.lookup':'Direct persistent-buffer shortcut onlyDP1/TP1; frozenTP8 does not select this shortcut','gather_engram_hashes':'One model call covers both layer slots; sharedDP true returns samehash tensor','self.lookup':'Explicit owned local heads→hostUVA/dequant native; every localprefix row written','gather_dequantize_host_uva':'Selected sharedHost codes/scales mapping, original INT64 chunk pointers and BF16 result','gather_dequantize_engram_int8':'Device-table alternative exists, outside selected CPUoffload frozen allocation','_torch_lookup':'CPU reference fallback outside selected registered hostUVA path','exchange_engram_rows':'Only table.dp_size>1; currentsharedDP1 excludes perstepDPexchange','tensor_model_parallel_all_gather':'Actual TP8 head gather before SProw slice; group/generation/orderedheads retained','self.wkv':'AscendReplicatedLinear→original FLOAT BF16 unquantized GEMM, no changedM/N or owner flag admission','engram_gate':'Selected rotated original FP32 operation order + BF16 residual, HcPost still reads residual','hash_state':'AscendNgramHashState.forward, requestindex→historywrite→hash before layer selection','hash_state.ensure_cache':'Inherited NgramHashState.ensure_cache, exact objectowner/reset semantics','create_engram_hash_state':'Ascend adapter wraps firstEngramlayerSWA, blocksize from actual layer','_write_hash_cache_kernel':'Upstream mapped/dead token state publication remains required even result cache hit','self._can_overlap_engram_preparation':'Frozen A3 defaultFalse from exactSHA old config/serve; alternate enabled protocol not claimed admitted'}
    rules.update({'AscendParallelEngramEmbedding':'Actual constructor binds each layer.embed_tokens to Ascend table; native storage/quant dtype comes from selected checkpoint andhardware gate','layer.engram':'Actual per-layer Engram.forward writes original BF16 HC residual before Decoder.forward/HcPre; downstream HcPost remains a reader'})
    routes=[]
    for e in excerpts:
        tree=ast.parse(Path(pins[e['runtime_path']]['local_path']).read_text())
        for c in ast.walk(tree):
            if not isinstance(c,ast.Call) or not e['first']<=c.lineno<=e['last']:continue
            name=ast.unparse(c.func)
            if name not in rules:continue
            ident=(e['runtime_path'],c.lineno,name)
            if any((r['runtime_path'],r['line'],r['callee'])==ident for r in routes):continue
            old=[dict(from_node=x['from_node'],line=x['line'],callee=x['callee']) for x in work['edges'] if x['line']==c.lineno and x['callee']==name and nodes[x['from_node']]['runtime_path']==e['runtime_path']]
            routes.append(dict(runtime_path=e['runtime_path'],sha256=e['sha256'],line=c.lineno,callee=name,disposition=rules[name],old_worklist_edges=old,all_recursive_callees_closed=False))
    outedges=dict(nodes=intersections,node_count=len(intersections),routes=routes,route_count=len(routes),matched_old_edges=sum(len(x['old_worklist_edges']) for x in routes),worklist_sha256=sha256((R/'recursive-source-worklist.json').read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Receiver/body index including frozen-inactive alternatives; not a global branch or native closure proof, not necessarily new source coverage relative to loop012.')
    (R/'engram-lookup-path-recursive-intersections.json').write_text(json.dumps(outedges,ensure_ascii=False,indent=2)+'\n')
    classification={'已停止':['沿旧预算lookup/scale广播/私有预零/hash小helper、独立gate与固定readiness tuple cache，不重测','原样overlap/staged prefix、mask passthrough、已失败changedM/stockAIVAGMM与直接错用E8M0 decoder配方'],'待确认':['完整input/owner/HC/iteration联合交付：原vendor数值tile、全rank资源ready、全部packet/pack/collective成本、bank末读与失败退役、六普通/请求曝光','INT8+原E8M0容量路线及NUMA/capacity用途，沿旧NPU位/配对加载/映射生命周期缺口','既有metadata/gid12、add-HC/sharedDFC、Markov/hostcontext原具体缺口'],'已确认可实现':['本轮新增0；旧常量/同次hidden widening有限native fixture资格独立保留']}
    out=dict(status='ENGRAM_ACTUAL_HASH_UVA_BODY_COVERAGE_AND_WHOLE_PARENT_PRICE_RECONCILED',current_source_pins=17,primary_source_pins=13,separate_extension_source_pins=4,named_body_or_range_walks=len(excerpts),body_worklist_intersections=len(intersections),receiver_routes=len(routes),matched_old_edges=outedges['matched_old_edges'],new_numerical_tests=0,new_mechanisms_claimed=0,history='Same parent and mechanisms already in Engram完整交付与融合判断.md / loop012 and later owner/storage records. Current body/receiver intersections and combined representative profile price fill conservative worklist omissions; no historical artifact overwritten.',classification=classification,original_bound_inputs=122,original_unbound_inputs=34,next_action=NEXT,whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],remaining=['No all-input/native tiling/resource/physical-lifetime admission for new owner or megakernel; larger parent not pruned by lookup price.','Full recursive frontier must combine exactSHA prior body receipts with actual selected receiver/branch data, rather than repeatedly choosing an old-covered node from stale no-excerpt list.','Source contracts distinguish current shared/rotatedV1 from nonsharedDP/device/MXFP8/unrotated/slotlessV2 alternatives. Actual source branch coverage and runtime population are separate.'],inputs=[dict(path=str(R/n),sha256=sha256((R/n).read_bytes()).hexdigest()) for n in deps+['engram-lookup-path-parent-primary.json','engram-lookup-path-price.json','engram-lookup-path-source-excerpts.json','engram-lookup-path-recursive-intersections.json','engram-execution-frontier.json','engram-joint-entry-review.json','engram-row-owner-source-review.json']],reviewer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (R/'engram-lookup-path-review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');(R/REPORT).write_text('# '+TITLE+'\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n旧完整合同与请求/六普通价：[Engram完整交付与融合判断](Engram完整交付与融合判断.md)。\n\n下一条待扫：'+NEXT+'\n')
    print(json.dumps({k:out[k] for k in ['current_source_pins','named_body_or_range_walks','body_worklist_intersections','receiver_routes','matched_old_edges','new_mechanisms_claimed','whole_chain_complete']}))
if __name__=='__main__':main()
