"""Package exact frozen native evidence; never include executor model I/O."""
import hashlib,json,pathlib,re,shutil
R=pathlib.Path(__file__).resolve().parents[1];M=R.parent/'repo-export';P=R/'publication-L039-exit-final'
prefix='tuning-brainstorm/sglang-pd-spec-36';sub=prefix+'/evidence/2026-10-19-910c-group-memory-bootstrap';O=M/sub
review=json.loads((R/'notes/L039-group-memory-review.json').read_text());O.mkdir(parents=True,exist_ok=False);changed=[];seen=set()
def copy(n):
 if n in seen:return
 seen.add(n);p=O/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,p);changed.append(sub+'/'+n)
def write(n,text):
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);changed.append(sub+'/'+n)
for n in range(53,75):
 tag=f'L039-R{n:03}';copy('specs/'+tag+'.json');s=json.loads((R/f'specs/{tag}.json').read_text())
 for name,digest in s['inputs'].items():
  assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,(tag,name)
  copy(name)
 for suffix in ['frozen-controller','zcode']:copy(f'runs/{tag}-{suffix}/receipt.json')
 for d in R.glob('runs/'+tag+'*'):
  if d.name.endswith(('-zcode','-frozen-controller')):continue
  for filename in ['receipt.json','remote-result.json','stdout.json']:
   if (d/filename).exists():copy(str((d/filename).relative_to(R)))
 for filename in ['manifest.json','result.json']:
  paths=list((R/'tasks/sglang-pd-spec-36/loops/L039-910c-runtime-recovery/runs').glob('*/'+tag+'/'+filename));assert len(paths)==1,(tag,filename)
  copy(str(paths[0].relative_to(R)))
for name in ['notes/L039-group-memory-review.json','notes/L039-group-memory-zcode-audit.json','notes/L039-model-handles-r073.json','runtime-contract.json','scripts/audit_L039_group_memory.py','scripts/package_L039_group_memory.py']:copy(name)
write('README.md','''# 910C：分组通信缓冲与原模型启动

全部计算与数值校验在本任务预留的910C节点执行。Mac只负责控制、静态审查、整理和提交。使用两台服务器：44为Prefill、46为Decode，各16个逻辑NPU。SSH入口是跳板。

原模型仍是封存的GLM-5.3-W8A8C8，每端193文件、186权重分片、764089835336字节；source/P/D全部文件SHA已在前序R035匹配。镜像config ID为c5cf2deb…，不是OCI仓库digest。安装SGLang提交ce0a8fea5f4245839d174b2b8c55692be1810a7f，3948个tracked Python文件匹配HEAD；无生产源码补丁。ModelSlim、BF16 KV、TP16/EP16、PD、MTP EAGLE三步/topk1/draft4、context33792、max_running3、mem_fraction0.90和原角色graph配置保留。

本检查点的最新观测R074：**44的原R067 Prefill纯ready接口200；46的新R073 Decode处于启动状态，尚未API就绪。** 完整PD请求、输出质量、生命周期与性能基线均未通过，整体E0、accepted_baseline=null。

## 进程退出诊断

R053去掉DeepEP，只运行16rank HCCL。所有worker完成、显式销毁且atexit记录通过后，docker top仍只剩torchrun父进程。达到限制后只停止该精确自有CID，退出137、OOMKilled=false。R055保持worker不变，改用不导入Torch的标准库父启动器，得到16个子进程及容器真实退出0。R056在创建设备计算容器前遇到EADDRINUSE，保留失败；不能据此断言是监听者还是TIME_WAIT。R057使用每Run端口，完成独立BF16全行golden及真实退出0。R062只改async_finish，亦通过。

这解决的是局部诊断启动器的退出问题，没有更换完整SGLang引擎的父进程实现，也没有用强制成功退出代替清理。

## HCCL窗口满足后仍有显存约束

R058把全局HCCL_BUFFSIZE设为2048后，原P/D均完成target和NextN权重加载，却在后续执行失败。R061完整自有日志明确记录HCCL尝试分配4296015872字节失败、ret24。EH0012 allocator lookup的标题不能独自证明根因。两个容器均自行退出0，语义仍是失败；观察器没有停止它们。

R063/R064读取实际安装源码和二进制，确认现有get_torch_distributed_pg_options：普通命名TP组返回None、继承全局HCCL_BUFFSIZE；default和含moe的组采用DEEPEP_HCCL_BUFFSIZE，未设置时才回退全局值。NPU创建独立moe_ep组，即使TP==EP。deep_ep_cpp根路径是包内同一二进制的符号链接，真实SHA为bf222f4e51b8fd5824380c34cbcc225441a316296f65bbf17a6e159c14a6ad58。

因此冻结同一代码、原安装helper和CPP、每rank实际分配并触碰55GiB压力的对照：

| Run | 普通TP / MoE缓冲MB | 原生结果 |
| --- | --- | --- |
| R065 A | 2048 / 2048 | 16rank TP精确求和通过；全部rank在低延迟dispatch出现明确EL0004、4GiB HCCL显存不足；golden未执行 |
| R066 B | 200 / 2048 | 全16rank独立BF16全行golden通过，16384有效行/201326592字节，16子进程及容器真实退出0 |

相同128输入token/rank、declared max256、hidden6144、256专家、topk8、BF16异步eager。TP初始化可用显存下降的中位值从4387776512降为513726464字节；这是该夹具中的资源占用证据，没有吞吐或延迟结论。

R067只采用镜像已有分组配置：HCCL_BUFFSIZE=200，DEEPEP_HCCL_BUFFSIZE=2048。**Prefill完整原模型及内部warmup通过，API就绪**。该内部请求不是外部驱动的真实P→D基线。Decode完成target verify图和draft decode图，随后draft extend图OOM。原始失败容器与日志均保留。

## Decode下一步：保持真实token容量

R069保存完整Decode日志，定位到draft extend图申请194MiB时NPU0只剩9.08MiB。此前target verify捕获4/8/12/16批次约增加5.18GiB，draft decode批次16再增加约0.50GiB。R070确认实际捕获max_bs16、draft width4，即64个token；不能从max_running_requests=3推断捕获批次3。实际SGLang dispatch默认上界128，容器没有该环境变量覆盖，已检索的生产源码没有运行时setter。

据此冻结新的支持性夹具：**64输入token/rank、声明上界128**，其余布局与55GiB压力相同，仅在46节点执行。它不是缩短服务请求；原有512→128、32768→256输入候选没有改变。

| Run | 普通TP / MoE缓冲MB | 原生结果 |
| --- | --- | --- |
| R071 A | 200 / 2048 | 全16rank BF16全行/数量golden与真实退出0通过 |
| R072 B | 200 / 1024 | 同上，8192有效行/100663296字节；dispatch后每rank比A多2144063488–2150739968字节可用显存，中位2147600384 |

R073随后只启动新的46 Decode：DEEPEP_HCCL_BUFFSIZE=1024，其余原模型参数与源码不变。44的R067精确CID保持运行，启动前只读ready200守卫通过。R074观测P仍就绪，D尚在启动。后续必须观察其完整加载、graph、内部warmup和API状态，再冻结真实PD输出/采样/到达/生命周期合同。

## 证据与边界

- [主控逐层审查、日志路径/SHA与后续动作](notes/L039-group-memory-review.json)
- [R065压力反例](runs/L039-R065-plain-exit/remote-result.json)、[R066单变量通过](runs/L039-R066-plain-exit/remote-result.json)
- [R071 Decode窗口A](runs/L039-R071-plain-exit/remote-result.json)、[R072 B](runs/L039-R072-plain-exit/remote-result.json)
- [R069图失败与实际源码](runs/L039-R069-decode-bounds/stdout.json)、[R070真实默认值及graph边界](runs/L039-R070-effective-bounds/stdout.json)
- [R074原服务与新Decode精确句柄/HTTP/资源观测](runs/L039-R074-full-model-state/stdout.json)
- [22条冻结命令各一次的元数据审计](notes/L039-group-memory-zcode-audit.json)，不公开执行器完整模型I/O；CLI0与内层失败分别保留。

每个golden在910C worker内用独立CPU路由枚举，比对每专家数量及所有有效6144维BF16行的SHA多重集合。未检查padding、combine、量化FFN全路径、真实KV或服务输出。压力夹具只近似显存占用，并未模拟模型语义；单组A/B资源结果不代替A/A噪声测量和多组性能对照。

下一步保持Prefill，观察精确D073句柄；就绪后先完成真实PD请求合同与输出/回收校验，再建A/A基线。此前[窗口不足诊断与失败](../2026-10-19-910c-hccl-window/README.md)保留，当前只更新M026/M028的狭义支持性认识，其余34项不改。
''')
path=prefix+'/mechanisms.json';old=json.loads((P/'base'/path).read_text());new=json.loads(json.dumps(old))
for x in new:
 if x['id']=='M028':
  x['status']='910C16rank分组HCCL压力对照通过；原Prefill已就绪，Decode图OOM后仅D改EP1024启动中；PD基线未建立'
  x['native_bootstrap_candidate']['group_memory_diagnostic']={'runs':[f'L039-R{n:03}' for n in range(53,75)],'evidence':'evidence/2026-10-19-910c-group-memory-bootstrap/README.md','P_ready':True,'D_ready':False,'full_PD_workload':False,'performance':'NOT_MEASURED','production_source_patch':False,'ordinary_TP_MB':200,'P_MoE_MB':2048,'D_MoE_MB':1024}
 if x['id']=='M026':
  x['status']='原Prefill的DeepEP完整warmup已通过；默认分发维度反例保留，完整PD性能基线仍未建立'
  x['native_startup_counterexample']['later_P_only_warmup']={'run':'L039-R067','observation':'L039-R068/R074','old_failure_preserved':True,'evidence':'evidence/2026-10-19-910c-group-memory-bootstrap/README.md','full_PD_quality_and_perf':False,'FuseEP_path_confirmed':False}
assert len(new)==36 and all(a==b for a,b in zip(old,new) if a['id'] not in ['M026','M028'])
(M/path).write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n');changed.append(path)
for path in [prefix+'/README.md','tuning-brainstorm/README.md']:
 text=(P/'base'/path).read_text()
 if path==prefix+'/README.md':
  i=text.index('**2026-10-19原生进展：**');j=text.index('\n\n',i);text=text[:i]+'**2026-10-19原生进展：** 两台910C各16NPU，原GLM-5.3-W8A8C8、PD与MTP保留。16rank退出与分组HCCL显存压力对照通过，Prefill原服务就绪，Decode图OOM后仅D调整通信缓冲启动中；完整PD输出/生命周期/性能基线仍为E0，见[原始证据、单变量对照和当前句柄](evidence/2026-10-19-910c-group-memory-bootstrap/README.md)。'+text[j:]
  lines=text.splitlines()
  for i,line in enumerate(lines):
   for x in new:
    if x['id'] in ['M026','M028'] and line.startswith('| ['+x['id']+']'):parts=line.split('|');parts[-3]=' '+x['status']+' ';lines[i]='|'.join(parts)
  text='\n'.join(lines)+'\n'
 else:
  marker='- [910C窗口诊断：';i=text.index(marker);text=text[:i]+'- [910C分组缓冲：16rank真实退出、55GiB压力反例/对照、原Prefill就绪与Decode图内存边界](sglang-pd-spec-36/evidence/2026-10-19-910c-group-memory-bootstrap/README.md)\n'+text[i:]
  text=text.replace('完整启动MoE维度及DeepEP两种模式tiling失败已记录，PD＋MTP基线待完成','旧MoE维度及tiling反例保留，后续分组缓冲对照通过、Prefill就绪、Decode启动中，PD＋MTP基线待完成')
 (M/path).write_text(text);changed.append(path)
for path in changed:
 data=(M/path).read_bytes();assert not re.search(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',data),path
 assert not re.search(rb'\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})',data),path
(P/'changed-files.json').write_text(json.dumps(changed,indent=2)+'\n')
print(json.dumps({'files':len(changed),'bytes':sum((M/p).stat().st_size for p in changed),'mechanisms_updated':['M026','M028'],'overall':'E0_SPEC_ONLY'}))
