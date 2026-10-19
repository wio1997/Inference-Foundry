#!/usr/bin/env python3
"""Mechanical source anchors; notes are frozen by the controller, no runtime inference."""
import datetime,hashlib,json,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[1];S=ROOT/'sources/installed';OUT=ROOT/'runs/L001-R005-path-table';OUT.mkdir(parents=True,exist_ok=False)
P='python/sglang/srt/'
spec=[
('M001','managers/tokenizer_manager.py','_tokenize_texts|_batch_tokenize_and_process','前端CPU入口存在；模板与tokenizer及动态批处理配置待本任务运行合同确认。'),
('M002','managers/detokenizer_manager.py','handle_batch_token_id_out|_grouped_batch_decode','反分词/IPC入口存在；背压和网络消费者需服务时间线。'),
('M003','utils/numa_utils.py','numa_bind_to_node|_read_pci_numa_node','NUMA工具存在；NPU设备映射、绑核和线程设置尚未冻结。'),
('M004','disaggregation/decode.py','get_next_disagg_decode_batch_to_run|event_loop_overlap_disagg_decode','专用D组批循环存在；实际调用次数未测。'),
('M005','managers/schedule_policy.py','_check_prefill_budget|preempt_to_schedule','预算与抢占入口存在；P/D资源和推测预留分别核验。'),
('M006','disaggregation/prefill.py','process_prefill_chunk|_send_kv_chunk','P续块与发送入口存在；chunk与传输粒度不能混为一个开关。'),
('M007','managers/scheduler.py','_dispatch_event_loop_once|mix_with_running','专用PD分派到独立P/D循环；P/D文件无mix_with_running或is_mixed_chunk引用。仅混批候选在专用PD下不适用，packed/ragged输入单独保留。'),
('M008','managers/schedule_policy.py','SchedulePolicy|_compute_prefix_matches','缓存感知策略存在；实际policy、缓存类型与D角色收益未知。'),
('M009','mem_cache/memory_pool.py','ReqToTokenPool|MLATokenToKVPool','页映射与MLA池存在；实际Ascend池/dtype及复制或分片方式待确认。'),
('M010','mem_cache/unified_cache/unified_tree_core.py','match_prefix|_match_prefix_helper','Unified树匹配入口存在；是否实际选择Unified而非旧Radix待确认。'),
('M011','mem_cache/unified_cache/unified_tree_core.py','evict_device_leaf|_update_evictable_leaf_sets','释放/淘汰入口存在；索引clone不等于KV载荷Copy-on-Write。'),
('M012','mem_cache/unified_cache/unified_tree_core.py','component_evictable_size|build_hicache_transfers','组件路径存在；模型状态类型、cache实现和组别待确认。'),
('M013','managers/cache_controller.py','HiCacheController|write|load','分层控制器存在；当前主线是否启用及NPU后端支持需独立探针。'),
('M014','managers/cache_controller.py','start_loading|_l2_load_transfers|write_storage','预取/回写入口存在；布局、event和后端能力不能由接口存在推断。'),
('M015','disaggregation/decode.py','event_loop_overlap_disagg_decode','D overlap循环存在；MTP保留，按真实batch回退与时间线诊断。'),
('M016','managers/overlap_utils.py','publish|stash|resolve_seq_lens_cpu|_resolve_spec_extras','真实新长度和bonus等字段仍需发布；普通decode删长度副本不适用于MTP。'),
('M017','managers/overlap_utils.py','decide_needs_cpu_seq_lens|resolve_seq_lens_cpu','CPU长度镜像已有按backend门控；缺失声明保守保留，TBO/ngram强制需要。'),
('M018','managers/overlap_utils.py','resolve_forward_inputs|resolve_mixed_spec_tails','延后H2D入口存在；私有D2H流分配明确CUDA-only，不能直接当Ascend实现。'),
('M019','hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py','alt_stream|q_event|record_stream','NPU Q投影副流/事件分支存在；模型和实际shape命中未知。'),
('M020','batch_overlap/two_batch_overlap.py','TboForwardBatchPreparer|MaybeTboDeepEPDispatcher','TBO入口存在；Ascend实际通信后端和spec组合待探针，不能套CUDA文档。'),
('M021','hardware_backend/npu/graph_runner/npu_graph_runner.py','NPUGraphRunner|_capture_graph','NPU graph runner存在；draft/target图覆盖及桶填充需运行计数。'),
('M022','hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py','fused_split_qk_norm|_disable_npu_fused_split_qk_norm','现有融合与opt-out门控存在；实际模型映射未闭合，不能强开融合或宣称数值等价。'),
('M023','layers/quantization/modelslim/schemes/modelslim_w8a8_int8.py','class |def ','W8A8方案文件存在；目录名C8不证明全部KV是INT8，量化描述与池需交叉确认。'),
('M024','hardware_backend/npu/attention/mla_preprocess.py','class |def ','NPU MLA预处理文件存在；调用者、KV布局及实际shape待记录。'),
('M025','hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py','auto-skip topk|skip_topk|topk_indices','DSA短序列跳选属于已有能力；HiSparse shared-index prefetch安装文档明确要求no speculative decoding，仅该子候选排除。'),
('M026','hardware_backend/npu/moe/fuseep.py','class |def ','NPU FuseEP文件存在；路由/重排/计算分段与实际专家分布未测。'),
('M027','hardware_backend/npu/graph_runner/npu_graph_runner.py','_create_device_graph|_capture_graph','后端和图路径存在；选择合法性、冷启动和形状分派需安装配置确认。'),
('M028','layers/moe/token_dispatcher/deepep.py','class |dispatch|combine','DeepEP源码存在不表示Ascend主线使用；应与NPU FuseEP分派分开核验。'),
('M029','disaggregation/prefill.py','get_num_tp_groups|num_kv_heads|attn_tp_size','PD并行分组相关代码存在；TP/EP/DP/PP/CP和跨节点拓扑仍UNKNOWN。'),
('M030','eplb/eplb_manager.py','rebalance|_check_rebalance_needed','重平衡入口存在；实际后端支持、冗余与最慢rank数据未测。'),
('M031','speculative/eagle_worker_v2.py','EAGLEWorkerV2|verify|draft','V2推测worker和NPU graph导入存在；具体MTP参数、接受长度分布及总成本仍未知。'),
('M032','speculative/spec_info.py','supports_grammar_overlap','实际算法能力判定比decode旧注释更细；EAGLE族等可声明支持，host-draft类按batch同步。CPU谓词不证明Ascend FSM/collective正确。'),
('M033','managers/detokenizer_manager.py','DecodeStatus|_decode_batch_token_id_output','增量反分词存在；SSE合并和结束回收需端点/取消测试。'),
('M034','docs/docs/advanced_features/sgl_model_gateway.mdx','cache_aware|cache-aware|cache aware','仅取得固定安装文档；实际gateway安装版本和副本数未知。单副本不做路由A/B，副本饱和不排除路由价值。'),
('M035','disaggregation/ascend/conn.py','AscendKVManager|send_kvcache|send_kvcache_dcp','Ascend连接器和DCP分支存在；复用PD-CHUNK索引，实际backend与发送时间线待证。'),
('M036','docs/docs/advanced_features/lora.mdx','speculative|Speculative|multimodal|LoRA','仅取得固定文档；会话/LoRA/多模态逐子功能核验，不默认开启非文本工作量。')]
catalog=json.loads((ROOT.parent/'repo-export/tuning-brainstorm/sglang-pd-spec-36/mechanisms.json').read_text())
rows=[]
for mid,path,pattern,note in spec:
 if not path.startswith('docs/'):path=P+path
 f=S/path;anchors=[]
 if f.exists():
  b=f.read_bytes();lines=b.decode().splitlines()
  for i,l in enumerate(lines,1):
   if re.search(pattern,l):anchors.append({'line':i,'text':l.strip()[:220]})
  h=hashlib.sha256(b).hexdigest()
 else:h=None
 rows.append({'id':mid,'title':catalog[int(mid[1:])-1]['title'],'installed_path':path,'sha256':h,'anchors':anchors[:12],'static_status':'ENTRY_LOCATED' if anchors else 'ANCHOR_GAP','runtime_path_confirmed':False,'performance_result':'NOT_RUN','controller_note':note})
assert len(rows)==36 and len({r['id'] for r in rows})==36
v={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'installed_commit':'ce0a8fea5f4245839d174b2b8c55692be1810a7f','reference_commit':'57db5089c207ae4ce189cc092a89a2027c8aea29','scope':'Static installed-tree anchors only. No task PD runtime, numerical or performance support is inferred.','rows':rows}
(OUT/'path-support.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
md=['# 36点安装路径与支持性审计','',v['scope'],'','观察安装提交：`'+v['installed_commit']+'`。本任务运行路径全部仍待实例化与计数；ENTRY_LOCATED不是PATH_CONFIRMED。','', '| ID | 机制 | 安装路径入口 | 当前边界 |','| --- | --- | --- | --- |']
for r in rows:
 line=r['anchors'][0]['line'] if r['anchors'] else None
 entry='`'+r['installed_path']+((':'+str(line)) if line else '')+'`'
 md.append('| '+r['id']+' | '+r['title']+' | '+entry+' | '+r['controller_note']+' |')
(OUT/'path-support.md').write_text('\n'.join(md)+'\n')
(OUT/'receipt.json').write_text(json.dumps({'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'exit_code':0,'count':36,'anchor_gaps':[r['id'] for r in rows if not r['anchors']],'artifacts':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['path-support.json','path-support.md']}},indent=2)+'\n')
print(json.dumps({'count':36,'anchor_gaps':[r['id'] for r in rows if not r['anchors']]}))
