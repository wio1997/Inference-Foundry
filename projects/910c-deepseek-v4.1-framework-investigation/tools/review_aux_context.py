#!/usr/bin/env python3
"""Reconcile actual aux/context receiver edges and stop only priced children."""
from pathlib import Path
from hashlib import sha256
import ast,json
R=Path('audit/execution-search')
TITLE='2026-11-11 三份aux到context cache：解析真实receiver，共享准备按原预算停止'
REPORT='三份aux到DSpark_context_cache的字段派发与共享判断.md'
SUMMARY='介绍：从target层37/38/39入口的HC均值，追三次SP交付、选行/concat、main投影/RMS和三个draft context cache，到各层SMLA实读。到了哪里：19当前源码身份、3份旧精确SHA回调源、15原Python/有限布局控制；24完整aux/main/三cache链、72panel。结果：A3 main投影不量化，RoPE有重复准备；同窗独立aux打包/准备共享沿原预算停止，更大的host/context共同producer保留。原122约束/34未定输入不变，native准入0，全链路未闭合。'
DETAIL='''真实字段派发：文本wrapper的language_model_cls→AscendDeepseekV41LLMForCausalLM，model_cls→DeepseekV41Model；PP1拥有40层，DSpark wrapper→DeepseekV41DSparkModel→三个DSparkDecoder。后者继承原Decoder.forward，attention_cls指向DSparkSWAAttention，原attention前向别名和CP实现保留。具名receiver与原工作表匹配边、额外展开body另列，不把“字段名字相同”当相同计算。

aux是真实三个不同时间的输入：模型在进入zero-based37/38/39之前取HC4均值、SP gather并裁full_num_tokens；不是三个相同Tensor，也不是层输出。runner的无spec metadata前缀分支先裁num_scheduled，验证分支按token_indices选行；随后按配置顺序concat，保留重复行、原顺序、discard/rejected与DP/SP尾部关系。DFlash继承的build_model_inputs_first_pass把固定context隐藏/位置buffer和各层slot交给三层cache producer，不能把B×5 query当作全部context行。

main数学边界：DSpark main_proj传入的quant_config仅在packed-cache设备且checkpoint标为fp8时非None；A3 packed=None，所以这里是原BF16/FLOAT unquantized GEMM，不能按模型W8A8标签省quant。main_proj前缀不命中特殊column op，ColumnParallelLinear保留K15360、TP8列640，feature gather回5120后原RMS；原return_bias=False。Replicated wkv经CustomReplicatedOp调用各自quant method，无这条feature TP gather。wkv的全局quant_config仍要按各prefix描述和实际native任务判断，不能从main不量化外推所有wkv。RMS还有bias_loaded分支，保持原norm/add边界；不把三份aux拆K后相加、改变M/分片权重或穿过RMS作代数折叠当作逐位等价。旧changed-M有4/2位差反例，原样配方不重跑。

实际少算点：每层_project_shared_kv都调用_apply_dsv4_rope；其get_cos_and_sin_dsa(context_positions)默认use_cache=False、layer_names=None，遍历全部已注册配置并构造新cos/sin，再只取本层名字。当前匹配链每层有4个表Index及其4个IndexCheck，三个panel共12次表Index；保留原第一份完整proxy、后两层共享准备，可删后8次及对应检查。它不要求三层config相同，但要求同一position/registry/full-cache代次、只读cos/sin与覆盖三消费者的租约。若再按layer_names限制配置，必须保留各层真实config，不能猜所有名字共用一套频率。15控制执行了原getter/proxy、不同config和缺层错误，并给出position变代次的否定例；没有原生数值验收资格。原consumer内cast、RotaryMul、KV/RMS数学和写入仍保留，不能把准备共享说成RoPE本体全免。

能少搬什么：先保持三个均值的BF16结果不动，考虑本卡aux按每row的三个H段打包，单次SP gather成全局[N,3H]，再选行；其值与原三次gather后选行/concat可做坐标对应。总aux逻辑payload不减少，仍要付pack、等体积collective和选行成本。三个legacy aux若以strided view暴露，还须闭合全部reader/别名、graph固定地址与native stride支持；只证明数值排列不足以准入。full_N裁切必须先于可能的负索引，TP8/N11与重复索引有限tag例保留了这点。

追到真实cache读者：_context_slot_mapping_buffers按_layer_group_idx重建，gid相同可引用同一slot buffer，三个cache Tensor仍是不同层的物理view。_store_standard_swa_kv在slot=None/empty时只停止store；上层仍已算过该层Wkv/norm/rope。A3一维linear slot先clamp/除模并把负数两列置-1；二维坐标保留。scatter_cache_sk把values转cache.dtype并contiguous，原SK算子保留cache实际page stride和[-1,-1]跳行，不能按shape假设平面连续。随后各DSpark层经显式dsa_v41_forward side-effect边界、query cache更新和metadata ATTENTION wait，把同一swa_cache_layer.kv_cache[0]作为SMLA ori_kv。packed A5、long-compressed reader不属于这个冻结draft角色。跨层/跨流须证明物理byte-span、cache/query读写次序及producer/consumer退役；不是前缀/gid不同就可并发。

按整段定价再剪枝：旧配平C8 EP0/8、M24/local3的完整aux均值+三AG+选行/main+三cache native分支，均0.893095%该窗；从最早aux到cache末写的墙钟跨度达5.374410%，其中夹着必需target末三层、head/rejection与metadata，不能当融合预算。aux AG/选行/concat全免仅0.158914%，context全部90任务0.583920%；三次全RoPE getter0.243367%，保留第一份而共享后两份的native信用0.161580%。新增pack10us+AG30us+选行25us、共享控制5us/call，二者不相交的条件净2.119–2.193ms/0.225392–0.235898%该窗；较贵110+10us仅0.161575–0.171357%，210+20us只0.021179–0.029368%，aux packing单项已转负。均为付成本情景，非实测新kernel收益。

历史/请求分开：原C8全16卡112周期中，全部90 context任务全免最多0.866501%周期，沿用原stage-confined row/RoPE/slot停止决定。固定所有此前迭代和请求开始、只推进当前carrier的旧context全免，TP/latest-DP最大个体1.759380%不代表P90；重算该模型P90只有2.175160ms/0.146108%，全16时钟宽包络为5.134580ms/0.344896%。它们都不是全迭代或客户端收益上界。旧逐波传播的更大host/context父段仍有P90全免23.118418ms/1.549643%、整批271.438628ms/1.460072%的宽包络，含必需工作且与metadata/frame重叠，不与本轮比例相加。六普通的完整新父段/净曝光仍缺，新增请求信用0；1%只是旧独立优先预算的引用，不是新用户硬阈值。

结果分档：已停止——相同预算下独立aux打包、context RoPE/slot/row小方案及既有changed-M反例配方，不做原样native重验。待确认——更大的aux/host/context共同producer、保持原数值tile的owner或megakernel，以及既有metadata/gid12/add-HC/sharedDFC/Engram/Markov候选；要扣必需数学、通信、query/cache写入及全部租约，再看六普通和请求曝光。已确认可实现——本轮0，原15协议/坐标控制不是逐位准入；旧封闭CPU原型资格独立保留。'''
NEXT='loop047下一Run：转DeepseekV41Compressor.forward及其实际设备实现，沿输入缓存/压缩KV、index-K发布到跨层SMLA实际消费者，先绑定冻结role与receiver，再按exact-SHA旧证据去重。覆盖合法ratio/short-input/append/reset/缓存复用分支和全部可见输出；先审整段数学/搬运/等待共享，再匹配原C8/六普通，低整段预算才结束同预算子项。保留大metadata/gid12、add-HC/sharedDFC、Engram owner、Markov/host-context具体缺口；每替换原语逐位一致，不重开本轮停止的小配方。本源码Run无新Torch/NPU/模型/profile/请求/服务/远端写入。'

WALKS={
 'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py':{None:['init_attention_projections'],'DeepseekV41Model':['forward'],'DeepseekV41DecoderLayer':['forward','hc_collapse']},
 'vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py':{None:['_apply_dsv4_rope'],'DeepseekV41DSparkModel':['__init__','combine_hidden_states','_project_shared_kv','_store_standard_swa_kv','precompute_and_store_context_kv','get_draft_kv_cache_layer_names','forward'],'DSparkDeepseekV41ForCausalLM':['combine_hidden_states','precompute_and_store_context_kv']},
 'vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py':{'AscendDflashProposer':['build_model_inputs_first_pass']},
 'vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py':{'AscendDSparkProposer':['set_inputs_first_pass']},
 'vllm-ascend/vllm_ascend/worker/model_runner_v1.py':{'NPUModelRunner':['propose_draft_token_ids']},
 'vllm-ascend/vllm_ascend/ops/rope_dsv4.py':{None:['get_cos_and_sin_dsa'],'RopeDataProxy':['__getitem__'],'ComplexExpRotaryEmbedding':['forward']},
 'vllm-ascend/vllm_ascend/models/common/ops/sequence_parallel.py':{None:['_custom_collective','sp_all_gather']},
 'vllm-ascend/vllm_ascend/ops/linear.py':{'AscendUnquantizedLinearMethod':['apply'],'AscendLinearBase':['__init__'],'AscendColumnParallelLinear':['forward'],'AscendReplicatedLinear':['forward']},
 'vllm-ascend/vllm_ascend/ops/linear_op.py':{None:['_get_column_parallel_op','get_parallel_op','get_replicated_op'],'CustomReplicatedOp':['apply_impl'],'CustomLinearOp':['apply']},
 'vllm/vllm/model_executor/layers/linear.py':{'ColumnParallelLinear':['forward']},
 'vllm-ascend/vllm_ascend/ops/layernorm.py':{'AscendRMSNorm':['forward_oot']},
}
def main():
    pins={s['runtime_path']:dict(s,acquisition=n) for n in ['aux-context-dependencies.json','aux-context-registry-dependencies.json'] for s in json.loads((R/n).read_text())['sources']}
    prior={}
    for file in ['whole-forward-dependencies.json','whole-forward-binding-dependencies.json','draft-output-dependencies.json']:
        prior.update({s['runtime_path']:dict(s,acquisition=file) for s in json.loads((R/file).read_text())['sources']})
    prior_names=['vllm-ascend/vllm_ascend/utils.py','vllm-ascend/vllm_ascend/attention/dsa_v41.py','vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py']
    refs={k:prior['/vllm-workspace/'+k] for k in prior_names};allpins=dict(pins);allpins.update({v['runtime_path']:v for v in refs.values()})
    for s in allpins.values():assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
    excerpts=[];walks=dict(WALKS);walks.update({'vllm-ascend/vllm_ascend/attention/dsa_v41.py':{None:['scatter_cache_sk','dsa_v41_forward'],'AscendDSAV41Impl':['_preprocess','_forward_attention']},'vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py':{'AscendDSAV41CPImpl':['_prepare_inputs_and_caches']}})
    for path,owners in walks.items():
        pin=allpins['/vllm-workspace/'+path];raw=Path(pin['local_path']).read_text();tree=ast.parse(raw);ls=raw.splitlines()
        for owner,names in owners.items():
            parent=tree if owner is None else next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name==owner)
            for name in names:
                f=next(n for n in parent.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name);excerpts.append(dict(runtime_path=pin['runtime_path'],sha256=pin['runtime_sha256'],owner=owner,name=name,first=f.lineno,last=f.end_lineno,lines=ls[f.lineno-1:f.end_lineno],source_acquisition=pin['acquisition'],fresh_this_run=pin['runtime_path'] in pins,qualification='Named body/source alternatives only; nested native/ownership/population closure is separate.'))
    utils=refs['vllm-ascend/vllm_ascend/utils.py'];ls=Path(utils['local_path']).read_text().splitlines();excerpts.append(dict(runtime_path=utils['runtime_path'],sha256=utils['runtime_sha256'],owner=None,name='OOT_type_registration',first=830,last=952,lines=ls[829:952],source_acquisition=utils['acquisition'],fresh_this_run=False,qualification='Prior exact source SHA reference: actual RMS/Linear OOT type map, not ops/register_custom_ops native function registration or a loaded-object dump.'))
    (R/'aux-context-source-excerpts.json').write_text(json.dumps(dict(excerpts=excerpts,current_disk_identities=19,prior_exact_SHA_sources=3,whole_chain_complete=False),ensure_ascii=False,indent=2)+'\n')
    routes=[
      ('models/deepseek_v41/vl_model.py',269,'self.language_model','language_model_cls=AscendDeepseekV41LLMForCausalLM; ctor95–98'),
      ('models/deepseek_v41/model.py',1715,'self.model','model_cls=DeepseekV41Model; ctor1626; alternate1723 same receiver'),
      ('models/deepseek_v41/model.py',1592,'layer','PP1 target40 Decoder layers, frozen build_layer_plan and make_layers'),
      ('models/deepseek_v41/dspark.py',409,'self.model','DeepseekV41DSparkModel ctor358'),
      ('models/deepseek_v41/dspark.py',265,'layer','3 DSparkDecoder layers; inherited Decoder.forward with DSpark attention_cls'),
      ('models/deepseek_v41/model.py',896,'self.input_layernorm','RMSNorm ctor818 -> AscendRMSNorm via utils OOT map'),
      ('models/deepseek_v41/model.py',899,'self.self_attn','attention_cls target/DSpark receiver; explicit dsa_v41_forward custom op'),
      ('models/deepseek_v41/model.py',913,'self.mlp','DeepseekV41MoE ctor811; existing shared/DFC proof retained, not closed by this edge'),
      ('models/deepseek_v41/model.py',1561,'hidden_states.mean','Original native HC4 mean, keep BF16 result/order'),
      ('models/deepseek_v41/model.py',1563,'sp_all_gather','SP custom_collective if supplied else TP allgather(dim0), crop full_N'),
      ('worker/model_runner_v1.py',2098,'self.drafter._propose','Actual AscendDSpark C3 -> AscendSpecDecodeBaseProposer._propose'),
      ('spec_decode/llm_base_proposer.py',1068,'cast(_HiddenStateDrafter, model).combine_hidden_states','Original unwrap guard -> DSpark wrapper446 -> model281'),
      ('models/deepseek_v41/dspark.py',282,'self.main_proj','AscendColumnParallelLinear -> no special main_proj column op -> original Column.forward; A3 quantNone'),
      ('models/deepseek_v41/dspark.py',282,'self.main_norm','AscendRMSNorm.forward_oot, preserve bias_loaded conditional add'),
      ('spec_decode/dflash_proposer.py',315,'self.model.precompute_and_store_context_kv','DSpark wrapper449 -> model295 with persistent context/position/slot slices'),
      ('models/deepseek_v41/dspark.py',308,'self._project_shared_kv','Three layers by ModuleDict iteration; separate weights/norms'),
      ('models/deepseek_v41/dspark.py',290,'attn.wkv','AscendReplicatedLinear -> CustomReplicatedOp -> original prefix quant_method; no feature AG'),
      ('models/deepseek_v41/dspark.py',290,'attn.kv_norm','AscendRMSNorm; original width512/native rounding retained'),
      ('models/deepseek_v41/dspark.py',292,'_apply_dsv4_rope','Imported local helper default inverseFalse; key from per-layer rotary_emb.layername'),
      ('models/deepseek_v41/dspark.py',62,'get_cos_and_sin_dsa','Original use_cacheFalse/layer_namesNone; full registry proxy; state/positions generation required for sharing'),
      ('models/deepseek_v41/dspark.py',68,'rotary_emb','ComplexExpRotaryEmbedding.forward -> native npu_rotary_mul(interleave) -> copy_ into original x view'),
      ('models/deepseek_v41/dspark.py',309,'self._store_standard_swa_kv','A3 standard store, preserve None/empty and 1D/2D coordinate branches'),
      ('models/deepseek_v41/dspark.py',246,'scatter_cache_sk','dsa_v41.scatter_cache_sk227: real cache stride, dtype+contiguous updates and native SK'),
      ('attention/dsa_v41.py',657,'torch.ops._C_ascend.npu_sparse_flash_mla','same per-layer swa_cache_layer.kv_cache[0] as ori_kv; metadata ATTENTION wait and query cache writes precede read'),
    ]
    work=json.loads((R/'recursive-source-worklist.json').read_text());bindings=[]
    for path,line,callee,target in routes:
        runtime='/vllm-workspace/vllm-ascend/vllm_ascend/'+path;pin=allpins[runtime]
        matches=[dict(from_node=e['from_node'],line=e['line'],callee=e['callee']) for e in work['edges'] if e['line']==line and e['callee']==callee and any(n['id']==e['from_node'] and n['runtime_path']==runtime for n in work['nodes'])]
        bindings.append(dict(runtime_path=runtime,sha256=pin['runtime_sha256'],line=line,callee=callee,selected_source_binding=target,original_worklist_edges=matches,state='SOURCE_RECEIVER_OR_PRIMITIVE_CONTRACT_IDENTIFIED_NOT_NATIVE_ADMISSION',all_recursive_callees_closed=False))
    edges=dict(status='AUX_CONTEXT_SELECTED_RECEIVER_AND_NATIVE_BOUNDARY_DISPOSITIONS',bindings=bindings,matched_original_edges=sum(len(x['original_worklist_edges']) for x in bindings),new_followup_bodies=[dict(path=e['runtime_path'],owner=e['owner'],name=e['name'],first=e['first'],last=e['last']) for e in excerpts if e.get('name') in ['combine_hidden_states','_project_shared_kv','_store_standard_swa_kv','precompute_and_store_context_kv','get_cos_and_sin_dsa','__getitem__','scatter_cache_sk']],old_recursive_worklist_sha256=sha256((R/'recursive-source-worklist.json').read_bytes()).hexdigest(),qualification='Sidecar semantic dispositions, not rewriting all319 old AST nodes as reviewed. Native kernels, field quant method/loaded state, cache byte ownership and branch populations keep concrete gaps.',whole_chain_complete=False)
    (R/'aux-context-edge-dispositions.json').write_text(json.dumps(edges,ensure_ascii=False,indent=2)+'\n')
    conditions=[('SP aux output','source selectedTrue under frozen SP; non-SP leg exists but not current selected allocation'),('aux role predicate','37/38/39 entry only; other37 layers do not produce aux'),('spec metadata None/present','Both prefix slicing and verification selector/concat legs read; preserve padded/default and no-copy cardinalities'),('main packed/quant gate','A3 packedNone fixes main quantNone; A5 fp8 other branch outside frozen hardware'),('context zero/None','empty states or overall slotsNone earlyreturn; empty positions return before first projection'),('per-layer slotNone/empty','Projection already ran, only store returns; do not invent current avoided projection cost'),('slot ndim1/2','Linear slots clamp/divmod/mask to paired -1; prepared2D kept'),('RoPE helper flags','Actual context false-cache/no-layer-filter; registry mutation, missing layers and alternate use_cache legs retained/fallback'),('norm bias_loaded','Runtime weight state, keep original conditional add'),('cache writer/reader','A3 SK handles physical strides and cache dtype; draft role has no compressed-source bank; preserve per-layer query writes/SMLA metadata wait')]
    classification={'已停止':['同旧独立优先预算下aux packing/独立context RoPE、slot、row小配方；未知代价不能将低整段预算变成值得重测','既有changed-M提前投影及split-K/穿RMS/擦掉舍入的无准入方案'],'待确认':['较大aux/host/context共同producer、原tile owner或megakernel；物理cache/view/collective与错误/capture/租约及六普通/请求净价缺口','既有metadata/gid12、add-HC/sharedDFC、Engram owner和完整Markov候选'],'已确认可实现':['本轮0；15原Python/有限tag控制不构成native位准入，旧CPU原型资格独立']}
    out=dict(status='AUX_CONTEXT_ACTUAL_RECEIVERS_AND_COMPLETE_NATIVE_BRANCH_PRICED_SMALL_CHILDREN_STOPPED',current_source_pins=19,prior_exact_SHA_sources=[dict(path=s['runtime_path'],sha256=s['runtime_sha256'],acquisition=s['acquisition']) for s in refs.values()],original_or_finite_controls=15,named_body_or_range_walks=len(excerpts),source_receiver_routes=len(bindings),matched_original_worklist_edges=edges['matched_original_edges'],source_branch_dispositions=[dict(predicate=a,disposition=b,observed_population='NOT_INFERRED_FROM_SOURCE') for a,b in conditions],classification=classification,remaining=['New packed/strided aux consumer acceptance, fixed graph addresses and alias/retirement have no native admission; small family stopped so no probe now.','Larger shared host/context producer must preserve old main M/tile/RMS, all3 per-layer KV/cache/query outputs and stable RoPE registry/position generation; actual loaded quant/bias/physical byte ownership and paid costs are separate.','Original context current-carrier versus all-wave request propagation and6ordinary complete-parent exposure differ; request credit0.','Recursive edge sidecar closes these receiver identifications only; continue Compressor/input-state/long-cache/index publication consumers rather than counts or named candidate gaps as completion.'],previous_goal_turn='PROGRESS: complete target output source/protocol/price and conservative recursive worklist were persisted and published4bac54dc; current turn follows that worklist.',scope_first_manifest_preserved=True,registry_note='ops/register_custom_ops is native function registration; actual high-level OOT class map is prior exactSHA utils830–952. No old manifest overwritten or loaded-object inspection claimed.',next_action=NEXT,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],inputs=[dict(path=str(R/n),sha256=sha256((R/n).read_bytes()).hexdigest()) for n in ['aux-context-dependencies.json','aux-context-registry-dependencies.json','aux-context-parent-primary.json','aux-context-controls.json','aux-context-price.json','aux-context-source-excerpts.json','aux-context-edge-dispositions.json','aux-config-call-chain.json','draft-output-dependencies.json']],reviewer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (R/'aux-context-review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');(R/REPORT).write_text('# '+TITLE+'\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一条待扫：'+NEXT+'\n')
    print(json.dumps(dict(current_sources=19,prior_callback_sources=3,controls=15,source_walk_bodies=len(excerpts),receiver_routes=len(bindings),matched_old_edges=edges['matched_original_edges'],new_native_admissions=0,whole_chain_complete=False)))
if __name__=='__main__':main()
