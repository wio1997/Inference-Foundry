from pathlib import Path
import ast,json,hashlib
P=Path(__file__).resolve().parent;old=P.parent/'idle-attribution-171'
rows=json.loads((old/'source-map.json').read_text())+json.loads((old/'additional-source-map.json').read_text())
files={r['runtime_path']:(old/r['bundle_path'],'audit/idle-attribution-171/'+r['bundle_path']) for r in rows}
for r in json.loads((P/'source-map.json').read_text()):
 if not r.get('missing'):files[r['path']]=(P/'source'/r['name'],'audit/framework-timing-171/source/'+r['name'])
catalog=json.loads((P/'full-request-functions.json').read_text())['catalog']
meaning={
'NPUModelRunner._prepare_inputs':'组装本轮模型输入、位置与序列信息，并准备缓存slot等交付数据；结合源码分支和下游metadata判断可共享的大段。',
'NPUModelRunner._build_attention_metadata':'按实际attention/KV组建立本轮metadata并收集设备metadata任务；候选重点是兼容同代数据的共同构造与发布。',
'AscendDSAV41CPMetadataBuilder.build':'将全局输入映射到CP本地token/序列视图，再构建并交付局部与全局metadata；不能跨代或跨物理布局盲缓存。',
'AscendDSACPMetadataBuilder._build_local_token_metadata':'构造CP本地query范围、query起始与序列长度。重点检查相同输入在兼容读者间是否重复计算。',
'AscendDSAV41MetadataBuilder.build':'按缓存种类与压缩比构造slot、序列、压缩器和索引metadata，保留必要缓存组差异。',
'_ReplicatedCacheMetadataBuilder._build_global_metadata':'获取或构建全局metadata；源码已有部分共享，新增收益只能按增量计。',
'AscendDSAV41MetadataBuilder._publish_task':'发布设备metadata构造任务；函数体本身通常小，详细标记开销须剔除，优先并入整体发布设计。',
'GPUModelRunner._preprocess':'把准备好的输入、位置与中间状态交给模型forward，包括配置实际启用的其它输入分支。',
'NPUModelRunner._model_forward':'Host调用模型执行；设备异步图执行需用NPU trace计时，不能用此函数墙钟替代。',
'NPUModelRunner.sample_tokens':'采样、校验、记账与草稿链路的Host入口；父时间包含子工作及等待。',
'NPUModelRunner._sync_metadata_across_dp':'跨DP协调本轮执行规模及metadata；通信必要工作与Host等待分别计价。',
'Scheduler.schedule':'从待执行请求生成下一批次与token预算；与等待worker结果的future范围分开。',
'Scheduler.update_from_output':'将worker结果写回调度状态并形成输出；不能把包含它的整个step当调度计算。',
}
out={}
for name,r in catalog.items():
 if r['path'] not in files:continue
 path,href=files[r['path']];text=path.read_text();lines=text.splitlines();start=r['line'];end=r['end_line'];code='\n'.join(lines[start-1:end]);tree=ast.parse(text)
 node=next((n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.lineno==start),None)
 calls=[]
 if node:
  for n in ast.walk(node):
   if isinstance(n,ast.Call):
    try:label=ast.unparse(n.func)
    except Exception:continue
    if label not in calls:calls.append(label)
 short=name.split(':')[-1]
 out[name]={'source':r['path'],'line':start,'end_line':end,'source_url':href+'#L'+str(start),
   'meaning':meaning.get(short,ast.get_docstring(node) if node and ast.get_docstring(node) else '此入口的实际交付逻辑见完整冻结源码；已测耗时不代表该函数全部可删。'),
   'snippet':'\n'.join(lines[start-1:min(end,start+39)]),'snippet_is_complete':end-start<40,
   'static_calls':calls[:25],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(P/'source-details.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(len(out),'source-bound function entries')
