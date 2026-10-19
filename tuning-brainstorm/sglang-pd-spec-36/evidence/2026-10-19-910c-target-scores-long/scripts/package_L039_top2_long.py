"""Package native target-score and original long-case evidence with compact views."""
import copy,hashlib,json,pathlib,re,shutil
R=pathlib.Path(__file__).resolve().parents[1];M=R.parent/'repo-export';P=R/'publication-L039-top2-final';prefix='tuning-brainstorm/sglang-pd-spec-36';sub=prefix+'/evidence/2026-10-19-910c-target-scores-long';O=M/sub;O.mkdir(parents=True,exist_ok=False);changed=[];seen=set();views=[]
def transfer(n):
 if n in seen:return
 seen.add(n);p=O/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,p);changed.append(sub+'/'+n)
def write(n,value):
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value);changed.append(sub+'/'+n)
def long_view(tag):
 name=f'runs/{tag}-long-state/remote-result.json';raw=(R/name).read_bytes();obj=json.loads(raw);v=copy.deepcopy(obj);w=v['nodes']['46']['data']['long_worker'];c=w['result'];omitted=0
 for role,response in c['batch_responses'].items():
  response['body_summary']=[{'id':x['meta_info']['id'],'prompt_tokens':x['meta_info']['prompt_tokens'],'completion_tokens':x['meta_info']['completion_tokens']} for x in response['body']];del response['body']
 for response in c['requests'].values():
  body=response['body'];ids=body.pop('prompt_token_ids');omitted+=len(ids);body['prompt_token_ids_fingerprint']={'count':len(ids),'sha256_json_compact':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'validated_exact_in_remote_worker':True}
 v['public_view_provenance']={'original_local_path':str(R/name),'original_sha256':hashlib.sha256(raw).hexdigest(),'original_bytes':len(raw),'changes':'Replace repeated echoed prompt ID arrays by count/hash and remove duplicate batched bodies. Keep all per-request outputs/meta/logprobs, every derived target-score row, timing, checks, worker/parent metadata and model logs. No timed sample or error filtered.','echoed_prompt_ID_count_replaced':omitted,'raw_worker_result_remote':w['handle']['client_root']+'/result.json','raw_worker_result_sha256':w['result_sha256']}
 out=f'runs/{tag}-long-state/public-view.json';write(out,json.dumps(v,ensure_ascii=False,indent=2)+'\n');views.append({'view':out,**v['public_view_provenance']})
for n in range(84,91):
 tag=f'L039-R{n:03}';transfer('specs/'+tag+'.json')
 for name,digest in json.loads((R/f'specs/{tag}.json').read_text())['inputs'].items():
  assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,(tag,name)
  transfer(name)
 for suffix in ['zcode','frozen-controller']:transfer(f'runs/{tag}-{suffix}/receipt.json')
 for d in R.glob('runs/'+tag+'*'):
  if d.name.endswith(('-zcode','-frozen-controller')):continue
  for name in ['receipt.json','remote-result.json','stdout.json']:
   if (d/name).exists():
    if n in [88,90] and name=='remote-result.json':long_view(tag)
    else:transfer(str((d/name).relative_to(R)))
 for name in ['manifest.json','result.json']:
  paths=list((R/'tasks/sglang-pd-spec-36/loops/L039-910c-runtime-recovery/runs').glob('*/'+tag+'/'+name));assert len(paths)==1;transfer(str(paths[0].relative_to(R)))
for n in ['notes/L039-top2-long-review.json','notes/L039-group-memory-zcode-audit.json','notes/L039-model-handles-r090.json','specs/L039-serving-contract-v2.json','runtime-contract.json','scripts/audit_L039_group_memory.py','scripts/package_L039_top2_long.py']:transfer(n)
write('public-view-sources.json',json.dumps(views,ensure_ascii=False,indent=2)+'\n')
text='''# 910C：目标分数诊断与原三路32K长用例

**原3×32768输入、每条256输出的PD＋MTP用例已连续两次完成功能校验。** 原模型、两台910C各16逻辑NPU、TP16/EP16、BF16 KV、EAGLE三步/topk1/draft4及graph均保留，没有缩短输入/输出。当前合计2048个输出token通过与返回目标最大分数的一致性检查；这不等于独立模型精度金标准、KV逐字节验证或完整生命周期证明。性能基线仍未接受，整体E0、accepted_baseline=null。

## 目标分数诊断

R084读取实际EAGLE公共验证、logprob处理与调度控制源码，并确认两端ready200。确定性推理开关实际为false，sampling_backend为pytorch。源码在token选择之后调用compute_spec_logprobs，支持返回所选token及top2分数；没有据此更改模型、采样、MTP、graph或确定性开关。

R085/R086保留两路512→128及PDv2输入数组顺序，只额外返回logprob/top2，均通过原功能检查。两次共512个选中token都等于各自返回的目标最大分数，全部输出IDs、MTP计数、完整top2行也逐行重复。允许报告分数并列，并不要求所选ID出现在任意tie-break后的前两名列表里。

前序分叉位置33/40在R085这个输出版本中的top2差距为0.375/0.625个自然对数单位；不能把它们笼统称为微小舍入噪声，也没有另一版本同位置的分数，尚不能证明根因。当前只有相对于实际目标分数的MTP选择一致性证据，没有将文本相似当精度通过。

回查各Run同一P主机时间范围内的日志，发现HTTP batch2不保证实际Prefill微批形状：

| Run | 实际P微批日志 | 输出关系 |
| --- | --- | --- |
| R078 | 一批2序列/1024 token | 首次执行，第三组输出版本 |
| R079、R083 | 两批1序列/512 token | 两次完整IDs/接受直方图相同 |
| R081、R082 | 一批2序列/1024 token | 两次完整IDs/接受直方图相同，区别于上一组 |
| R085、R086 | 两批1序列/512 token | top2观察版本，复现R079/R083输出 |

这是调度形状与输出分组的相关证据，不是已证明的因果关系。后续控制批次形状或数值质量时要记录真实微批，而非仅看HTTP数组顺序。观察logprob也可能改变调度与成本，因此本轮不与无logprob请求比较速度。

## 原长输入完整执行，没有改成短用例

R087将三条原32768-token输入同时以batch3到达P/D，独占客户端PID及源文件SHA均保存。其数值客户端在R089中保持完全相同字节重复；只增加父进程监督与实际退出回执。R088/R090只读采集精确PID、结果文件、模型日志和状态，没有发送新推理或重放任务。

两轮均满足：每条返回prompt IDs与原32768输入逐个相同，P边界token与D首token相同，D完整256个合法词表ID且以length结束，MTP verify/正确草稿/proposed/接受直方图守恒。两轮共1536个输出token均与各自返回目标最大分数一致，三条输出IDs及全部MTP计数/直方图在重复间完全相同。

| 观测 | 客户端完整批窗口 | 三条verify次数 | 正确草稿数 | 生命周期观察 |
| --- | ---: | --- | --- | --- |
| R087启动 / R088读取 | 58.932 s | 92 / 91 / 100 | 163 / 164 / 156 | 完整结果、stderr空、原PID消失；非父进程无法取得实际退出码，记UNKNOWN |
| R089启动 / R090读取 | 55.094 s | 92 / 91 / 100 | 163 / 164 / 156 | 父进程wait确认客户端退出0；父/子PID均消失、stderr空 |

每轮开始和结束的两端空闲cache flush均200，结束后两端pure ready200。源码flush_cache仅在is_fully_idle通过后重置cache与请求/KV pool；这仍不替代取消时的发送drain、延迟ack、页/引用所有权测试。没有把第二轮新增退出监督算作数值或性能优化。

两个批窗口由同一910C46客户端monotonic时钟计算，包含P/D HTTP、Prefill、KV交接、排队、MTP生成和额外分数观察。没有跨主机时间戳相减。两轮没有优化A/B，58.932→55.094秒不是加速结论；当前也未报告稳态TTFT/TPOT或p95/p99。

## 三路到达与真实容量边界

max_running_requests仍为3。P/D KV容量实际为43904/43776 token。D长请求日志显示一个活跃上下文占32896/33024 token，其余预分配等待数2→1→0；npu graph为True。P按原4096 chunk推进三条完整32K输入。**通过的是三路到达及正常准入/排队，没有声称三份完整上下文同时驻留。**

R090服务报告D的target_verify/draft_decode/draft_extend图区间内存分别3.188/0.500/0.594 GB，共4.282 GB，P为原角色默认无图。普通decode graph批次列表为空，不代表EAGLE验证/草稿图未执行。该内存字段也不能直接解释成可全部消除的图节点缓冲，需要分离lazy通信分配与真实图工作区。

下一步先验证真实自然EOS、取消、重试、延迟发送/页引用回收，再冻结适合数值路径的质量判据与稳态A/A。随后按实际内存与关键路径证据选择单个配置/代码候选；不降低正确性，不关闭PD/MTP/graph，不以短输入替代原长用例。

- [主控审查、top2逐行重复、微批相关性与长用例范围](notes/L039-top2-long-review.json)
- [R085原始短请求及全部分数](runs/L039-R085-pd-functional/remote-result.json)、[R086重复](runs/L039-R086-pd-functional/remote-result.json)
- [R088长用例公开视图](runs/L039-R088-long-state/public-view.json)、[R090重复与实际退出回执](runs/L039-R090-long-state/public-view.json)
- [视图来源与原始SHA](public-view-sources.json)：只替换重复回显的prompt ID数组及重复batched body，保留所有输出、逐token分数、计数、时间、日志和错误；原始结果仍在任务本地与自有910C路径。receipt中的SHA属于原始数据，不是公开视图。
- [实际源码与运行配置](runs/L039-R084-logprob-contract/stdout.json)、[冻结原长用例](specs/L039-R087-pd-plan.json)、[退出监督版本](specs/L039-R089-pd-plan.json)
- [38条冻结命令各一次的会话元数据审计](notes/L039-group-memory-zcode-audit.json)，不公开执行器完整模型I/O。
- [此前首次真实PD及重复输出反例](../2026-10-19-910c-first-pd/README.md)
'''
write('README.md',text)
path=prefix+'/mechanisms.json';old=json.loads((P/'base'/path).read_text());new=json.loads(json.dumps(old));states={'M004':'真实HTTP批次未固定P微批；原三路32K到达/排队已观测，取消/重试待测','M005':'原三路32K完成，D一份上下文驻留与2→1→0等待已实测；回收生命周期未全测','M006':'原4096 chunk完成三路32768输入两次；未进行chunk性能A/B','M021':'真实长短请求使用EAGLE图，D三阶段图区间内存已读取；裁剪候选未执行','M028':'现有分组缓冲支持原长短PD+MTP功能用例；稳定性能基线未接受','M031':'2048个输出token与返回目标最大分数一致；长Case两次输出/接受直方图相同，精度金标准及完整生命周期未闭合','M035':'原三路32K真实PD功能交接及重复通过；KV逐字节、取消/drain与引用回收待测'}
for x in new:
 if x['id'] in states:
  x['status']=states[x['id']];x['target_score_and_long_PD']={'runs':[f'L039-R{n:03}' for n in range(84,91)],'short_observer_runs':2,'long_original_3x32768to256_runs':2,'observed_chosen_tokens_at_target_max':2048,'D_co_resident_long_contexts':1,'long_queued_arrivals':3,'independent_model_or_KV_byte_oracle':False,'full_lifecycle':False,'optimization_speedup':'NOT_MEASURED','accepted_baseline':None,'evidence':'evidence/2026-10-19-910c-target-scores-long/README.md'}
assert len(new)==36 and all(a==b for a,b in zip(old,new) if a['id'] not in states)
(M/path).write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n');changed.append(path)
for path in [prefix+'/README.md','tuning-brainstorm/README.md']:
 text=(P/'base'/path).read_text()
 if path==prefix+'/README.md':
  i=text.index('**2026-10-19原生进展：**');j=text.index('\n\n',i);text=text[:i]+'**2026-10-19原生进展：** 原长短PD＋MTP功能用例均已实测；三路32768→256连续两次完成，D实际一份上下文驻留、其余排队。2048个输出token与返回目标最大分数一致；短Case微批变化/输出分叉、完整生命周期及稳定性能基线仍待核验，整体E0。见[目标分数、原长用例与实际容量边界](evidence/2026-10-19-910c-target-scores-long/README.md)。'+text[j:]
  lines=text.splitlines()
  for i,line in enumerate(lines):
   for id,status in states.items():
    if line.startswith('| ['+id+']'):parts=line.split('|');parts[-3]=' '+status+' ';lines[i]='|'.join(parts)
  text='\n'.join(lines)+'\n'
 else:
  i=text.index('- [910C首次真实PD：');text=text[:i]+'- [910C原长用例：三路32K→256两次通过、目标分数一致性、实际排队与退出证据](sglang-pd-spec-36/evidence/2026-10-19-910c-target-scores-long/README.md)\n'+text[i:]
  text=text.replace('真实PD＋MTP短请求功能通过，输出重复性与基线待完成','原长短PD＋MTP功能通过，三路32K完成两次；完整生命周期与基线待完成')
 (M/path).write_text(text);changed.append(path)
for name in changed:
 data=(M/name).read_bytes();assert not re.search(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{20,}|\bghp_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}',data),name
(P/'changed-files.json').write_text(json.dumps(changed,indent=2)+'\n');print(json.dumps({'files':len(changed),'bytes':sum((M/f).stat().st_size for f in changed),'mechanisms_updated':list(states),'raw_large_results_preserved':True,'baseline':None}))
