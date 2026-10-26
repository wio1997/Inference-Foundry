"""Publish original real PD requests and the unresolved repeat-output counterexample."""
import hashlib,json,pathlib,re,shutil
R=pathlib.Path(__file__).resolve().parents[1];M=R.parent/'repo-export';P=R/'publication-L039-first-PD-final';prefix='tuning-brainstorm/sglang-pd-spec-36';sub=prefix+'/evidence/2026-10-19-910c-first-pd';O=M/sub;O.mkdir(parents=True,exist_ok=False);changed=[];seen=set()
def copy(n):
 if n in seen:return
 seen.add(n);p=O/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,p);changed.append(sub+'/'+n)
for n in range(77,84):
 tag=f'L039-R{n:03}';copy('specs/'+tag+'.json')
 for name,digest in json.loads((R/f'specs/{tag}.json').read_text())['inputs'].items():
  assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,(tag,name)
  copy(name)
 for suffix in ['zcode','frozen-controller']:copy(f'runs/{tag}-{suffix}/receipt.json')
 for d in R.glob('runs/'+tag+'*'):
  if d.name.endswith(('-zcode','-frozen-controller')):continue
  for name in ['receipt.json','remote-result.json','stdout.json']:
   if (d/name).exists():copy(str((d/name).relative_to(R)))
 for name in ['manifest.json','result.json']:
  found=list((R/'tasks/sglang-pd-spec-36/loops/L039-910c-runtime-recovery/runs').glob('*/'+tag+'/'+name));assert len(found)==1;copy(str(found[0].relative_to(R)))
for n in ['specs/L039-serving-contract-v1.json','notes/L039-first-pd-review.json','notes/L039-group-memory-zcode-audit.json','notes/L039-model-handles-r076.json','runtime-contract.json','scripts/audit_L039_group_memory.py','scripts/package_L039_first_pd.py']:copy(n)
review=json.loads((R/'notes/L039-first-pd-review.json').read_text())
text='''# 910C：首次真实PD＋MTP请求与重复输出反例

原GLM-5.3-W8A8C8、两台910C各16逻辑NPU、P/D分离和EAGLE三步/topk1/draft4保持不变。所有请求在910C46的标准库客户端发送至精确自有P067/D073；Mac只控制和整理。**5批共10条真实短请求的功能检查通过；贪心重复输出仍有差异，完整质量/性能基线未接受，整体E0、accepted_baseline=null。**

## 先核实真实协议，再冻结合同

R077确认两端仍ready200，从实际镜像读取9个PD路由、io_struct、采样和输出源码，均匹配ce0a8fea5f4245839d174b2b8c55692be1810a7f的HEAD。真实HTTP PD路由向P/D并发发送相同业务体，使用P的bootstrap_host/port和相同唯一room。P内部把max_new_tokens置1，首token随KV交接；不能用fake bootstrap或两个独立单引擎结果代替。

冻结原2×512输入IDs、每条128输出。显式temperature0/top_k1/top_p1、无penalty/stop、ignore_eos=true以固定输出长度，非stream、无logprob额外计算；这是长度控制的功能协议，尚未覆盖自然EOS。API原本就返回output_ids和MTP计数，所以无需为拿token IDs启用logprob。输入/模板/tokenizer哈希沿用原C8的R026重验结果。

PDv1：两对请求、共4条P/D HTTP POST在同一910C46线程barrier释放。每Run使用独立rid与room，不重试。前置两端空闲cache flush，完成后再验证pure ready及flush。失败时只abort本Run的精确rid，绝不abort_all或停止服务。

## 原生功能结果

R078/R079及后续PDv2的R081/R082/R083均通过：P/D HTTP200，返回prompt IDs与512个原输入逐个一致，P返回1个token、D完整128个合法词表ID，P/D首token相同，D以length结束，MTP verify非零、proposed=verify×3、正确草稿数有界，接受直方图的总数及加权和与原始计数一致。所有Run结束后两端ready与空闲cache flush均200。

R078日志实际记录P为2序列/1024个新token、cached0，D为npu graph True。该观察使用原Ascend传输后端和非fake私网bootstrap。它是功能交接证据，没有逐字节读回真实KV，也没有单凭首token一致证明所有层的数值质量。

| Run | 到达合同 | 同一客户端完整批窗口 | 每请求verify次数 | 128-token输出重复性 |
| --- | --- | ---: | --- | --- |
| R078 | PDv1，独立请求同时释放 | 22.519 s | 44 / 45 | 首次真实执行；保存完整输出 |
| R079 | 同PDv1重复 | 6.787 s | 49 / 42 | 相对R078从零基位置33/40分叉 |
| R081 | PDv2，固定顺序batch2 | 6.802 s | 44 / 41 | 新诊断到达合同，不与v1作优化比较 |
| R082 | 同PDv2重复 | 3.429 s | 44 / 41 | 两条完整IDs及全部MTP计数/直方图与R081相同 |
| R083 | 同PDv2第三次 | 4.264 s | 49 / 42 | 与R081/R082不同，但完整IDs和MTP计数恰好复现R079 |

这些秒数是同一910C46客户端monotonic窗口，包含发请求与等完整响应。没有跨主机时钟相减；没有从非stream窗口推断TTFT/TPOT。第一轮可能含首次执行开销；后续输出与MTP工作量也不同。**22.519→6.787秒不是本任务已实现的加速比，3.429秒也不是已接受的稳定基线。** 日志内的gen throughput可能包含长时间空闲，未用作本Case吞吐。

R078的两请求分别有83/83个正确草稿、132/135个proposed，接受直方图[9,8,6,21]和[5,15,7,18]。各Run完整计数在原始结果中，保留每请求的接受分布，不把高接受率直接叫作性能改善。首token与末尾长度截断需另核，不能简单把正确草稿数加verify次数当最终输出长度。

## 重复性诊断与未完成项

R080读取真实SamplingBatchInfo、EAGLE验证路径及安装的NPU greedy Triton kernel。top_k<=1使源码is_all_greedy为真，EAGLE目标路径使用argmax；这只是调用路径依据，没有观察每步实际logits，也不能据此宣布底层数值确定。预期的scheduler_control_mixin.py路径在镜像中不存在，记录exists=false，没有冒称已读取其分配器状态。

PDv2仍是两路512→128、同一采样/MTP/graph/模型/缓冲，只把每角色的两条独立POST改成固定输入顺序的一个batch2 POST，以排查到达/批次顺序因素。R081/R082一次相同尚不足以接受；第三次分叉证明固定输入数组顺序仍不足以保证输出一致。根因尚未定位，不将其武断归为随机采样、硬件误差或KV损坏。

下一步核实原NPU EAGLE的logprob返回支持、实际调度/graph/确定性设置，冻结目标top2分数的观察诊断，核对分叉位置的margin与token选择。保持PD、MTP、graph和原长度，不通过换模型或缩短输入制造通过。

原3×32768→256仍待容量/准入/排队与真实请求验证；自然EOS、取消、重试、独立页/引用回收计数及稳定A/A基线也未完成。空闲flush200只是当前请求已结束/可清缓存的检查，不等于完整资源生命周期证明。没有生产源码补丁或优化速度结论。

- [主控逐Run审查、重复输出差异与下一动作](notes/L039-first-pd-review.json)
- [首个真实PD原始结果、输出IDs、MTP直方图和日志](runs/L039-R078-pd-functional/remote-result.json)
- [PDv1重复反例](runs/L039-R079-pd-functional/remote-result.json)
- [PDv2第一次](runs/L039-R081-pd-functional/remote-result.json)、[第二次](runs/L039-R082-pd-functional/remote-result.json)、[第三次反例](runs/L039-R083-pd-functional/remote-result.json)
- [实际协议源码与模型元数据](runs/L039-R077-pd-contract/stdout.json)、[实际greedy路径及NPU kernel源码](runs/L039-R080-greedy-sources/stdout.json)
- [冻结PD-v1合同](specs/L039-serving-contract-v1.json)、[PD-v2诊断计划](specs/L039-R081-pd-plan.json)
- [精确执行会话元数据审计](notes/L039-group-memory-zcode-audit.json)，R053–R083各冻结命令一次，不公开执行器完整模型I/O。
- [前序双端就绪](../2026-10-19-910c-pd-ready/README.md)、[分组通信缓冲原生对照](../2026-10-19-910c-group-memory-bootstrap/README.md)
'''
(O/'README.md').write_text(text);changed.append(sub+'/README.md')
path=prefix+'/mechanisms.json';old=json.loads((P/'base'/path).read_text());new=json.loads(json.dumps(old));ids=['M004','M028','M031','M035']
statuses={'M004':'真实两路PD功能通过；固定batch仍有贪心输出差异，调度/取消及稳定基线未完成','M028':'分组缓冲下真实2x512→128 PD+MTP功能通过；输出重复性与性能基线未接受','M031':'真实MTP verify/正确草稿/接受直方图已采集；重复输出分叉，完整质量与生命周期待测','M035':'真实非fake PD短请求功能交接通过；KV逐字节核验及完整异步生命周期仍待测'}
for x in new:
 if x['id'] in ids:
  x['status']=statuses[x['id']];x['first_real_PD_requests']={'runs':['L039-R078','L039-R079','L039-R081','L039-R082','L039-R083'],'case':'2x512->128','functional_batches':5,'logical_requests':10,'graph_and_MTP_kept':True,'bitwise_KV_oracle':False,'full_quality_or_lifecycle':False,'repeat_output_stability':'NOT_ESTABLISHED','accepted_baseline':None,'evidence':'evidence/2026-10-19-910c-first-pd/README.md'}
assert len(new)==36 and all(a==b for a,b in zip(old,new) if a['id'] not in ids)
(M/path).write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n');changed.append(path)
for path in [prefix+'/README.md','tuning-brainstorm/README.md']:
 text=(P/'base'/path).read_text()
 if path==prefix+'/README.md':
  i=text.index('**2026-10-19原生进展：**');j=text.index('\n\n',i);text=text[:i]+'**2026-10-19原生进展：** 两台910C原GLM-5.3-W8A8C8、PD与MTP保留，5批共10条真实512→128短请求功能通过；贪心重复输出仍有分叉，固定输入batch顺序也不足以保证一致。长用例、完整生命周期与性能基线未完成，整体E0，见[真实请求、MTP计数和反例](evidence/2026-10-19-910c-first-pd/README.md)。'+text[j:]
  lines=text.splitlines()
  for i,line in enumerate(lines):
   for id,status in statuses.items():
    if line.startswith('| ['+id+']'):parts=line.split('|');parts[-3]=' '+status+' ';lines[i]='|'.join(parts)
  text='\n'.join(lines)+'\n'
 else:
  i=text.index('- [910C双端就绪：');text=text[:i]+'- [910C首次真实PD：10条短请求功能通过、MTP接受直方图与贪心重复输出反例](sglang-pd-spec-36/evidence/2026-10-19-910c-first-pd/README.md)\n'+text[i:]
  text=text.replace('双端API就绪，真实PD＋MTP请求与基线待完成','真实PD＋MTP短请求功能通过，输出重复性与基线待完成')
 (M/path).write_text(text);changed.append(path)
for name in changed:
 data=(M/name).read_bytes();assert not re.search(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{20,}|\bghp_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}',data),name
(P/'changed-files.json').write_text(json.dumps(changed,indent=2)+'\n');print(json.dumps({'files':len(changed),'bytes':sum((M/f).stat().st_size for f in changed),'mechanisms_updated':ids,'overall':'E0_SPEC_ONLY','baseline_accepted':False}))
