"""Package verified shutdown and final resume handoff after the experiment checkpoint."""
import copy,hashlib,json,pathlib,re,shutil
R=pathlib.Path(__file__).resolve().parents[1];M=R.parent/'repo-export';P=R/'publication-L039-stopped-pause'
prefix='tuning-brainstorm/sglang-pd-spec-36';sub=prefix+'/evidence/2026-10-19-910c-stopped-pause';O=M/sub;O.mkdir(parents=True,exist_ok=False)
result=json.loads((R/'runs/L039-R093-shutdown/remote-result.json').read_text())
assert result['status']=='PASS_OWN_SERVICES_STOPPED_RESERVATIONS_RELEASED' and result['generation_requests']==0 and result['containers_removed']==0
assert len(result['stops'])==3 and all(x['exit_code']==0 for x in result['stops'])
for stop in result['stops']:
    for item in stop['data']['results']:
        assert not item['after']['Running'] and not item['original_owned_processes_still_active'] and item['container_retained'] and not item['model_files_modified']
assert len(result['lease_release'])==2 and all(x['UID_precondition'] and x['old_UID_absent'] and x['delete_exit_code']==0 for x in result['lease_release'].values())
changed=[];seen=set()
def transfer(name):
    if name in seen:return
    seen.add(name);p=O/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/name,p);changed.append(sub+'/'+name)
def write(name,text):
    p=O/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);changed.append(sub+'/'+name)
transfer('specs/L039-R093.json')
for name,digest in json.loads((R/'specs/L039-R093.json').read_text())['inputs'].items():
    assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest;transfer(name)
for name in ['runs/L039-R093-zcode/receipt.json','runs/L039-R093-frozen-controller/receipt.json','runs/L039-R093-shutdown/receipt.json','runs/L039-R093-shutdown/remote-result.json','notes/L039-stop-review.json','notes/L039-server-migration-handoff.json','notes/L039-group-memory-zcode-audit.json','scripts/package_L039_stopped_pause.py','scripts/verify_L039_stop_publication.py']:
    transfer(name)
for filename in ['manifest.json','result.json']:
    transfer('tasks/sglang-pd-spec-36/loops/L039-910c-runtime-recovery/runs/lifecycle/L039-R093/'+filename)
write('README.md','''# 本任务服务停止、NPU预留释放与暂停

用户明确要求完成当前R092、提交实验记录后，停止本任务的Prefill、Decode与共享存储，释放NPU预留并保留模型、容器及日志。实验已先提交为fb9f91a97582dc78f646abe56fbd0e3f1157fa07，随后才执行R093；没有再发送生成请求。

R093对三个完整CID、task/run标签及两个reservation Pod UID进行守卫。按Decode→Prefill→store顺序停止，停止前后完整自有日志写入各节点本任务R093目录并保存SHA；保留原容器及模型文件。核验原docker top进程的PID与启动tick，停止后均无同一进程仍活跃。容器退出码如实保留，停机中的非零引擎退出码不被改写为模型测试成功。

两个NPU预留Pod通过Kubernetes DeleteOptions UID precondition删除，随后确认旧UID消失；不会删除同名替代Pod。44与46各16逻辑NPU的本任务预留已释放。访问Pod、其他服务和模型磁盘未动；PriorityClass只保留无硬件占用的调度元数据。

本任务当前没有活跃模型服务、共享存储或实验客户端。按用户要求暂停，等待新服务器和明确恢复指令；目标仍未完成。原长短PD＋MTP功能、条件取消/重试和局部通信显存证据保留，稳定性能A/A、优化A/B、完整36机制原生覆盖与硬件前沿仍待继续。

- [独立停机复核与资源释放摘要](notes/L039-stop-review.json)。
- [完整停止响应、PID/start-tick与日志SHA](runs/L039-R093-shutdown/remote-result.json)。
- [最终迁移交接](notes/L039-server-migration-handoff.json)、[任务续跑入口](../../HANDOFF.md)。
- [先提交的EOS失败与取消/重试实验](../2026-10-19-910c-lifecycle-pause/README.md)。
- [冻结执行与退出回执](runs/L039-R093-frozen-controller/receipt.json)、[41条冻结命令元数据审计](notes/L039-group-memory-zcode-audit.json)。
''')
handoff='''# SGLang 36机制任务暂停交接

2026-10-19，用户要求完成当前实验、提交记录并暂停，之后提供新服务器继续。**目前无本任务活跃模型/共享存储服务；两端NPU预留已释放，原模型、容器和日志保留。不要自动重启或重放旧Run。**

## 从哪里继续

先读[最终停止与资源释放](evidence/2026-10-19-910c-stopped-pause/README.md)、[迁移合同](evidence/2026-10-19-910c-stopped-pause/notes/L039-server-migration-handoff.json)、[实验合同](CONTRACT.md)。最后数值实验为R092；R093仅作用户授权的停机，无新推理。下一个可用编号为L039-R094，也可按新服务器建立新Loop，必须先登记后执行，不能复用旧冻结命令。

新服务器到位后需要连接方式、可用设备与资源归属、模型/镜像位置。先只读盘点硬件、CANN/驱动/依赖和其他任务占用，再对原模型/tokenizer/安装源码/镜像做哈希校验、建立新位置合同。旧服务器数据不能直接作为新硬件性能基线。

## 保留的主线

原GLM-5.3-W8A8C8，193模型文件、186权重分片，共764089835336字节；SGLang固定ce0a8fea5f4245839d174b2b8c55692be1810a7f。原镜像config ID c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263（不是OCI仓库digest）。原模型全文件封存证据在R035，实际位置及校验入口见迁移合同。

Ascend、P/D分离、MTP EAGLE/NEXTN三步/topk1/draft4、TP16/EP16/DP1、BF16 KV。原context33792、max_running3、chunk4096、mem_fraction0.90；D保留验证与两类草稿图。普通HCCL缓冲200MB，P MoE/default2048MB、D1024MB。这些是旧环境已验证配置，换环境必须核对支持性，不能盲套。

## 已有证据与未完成工作

原2×512→128及3×32768→256均完成功能检查。三路32K两次输出相同，D实际一份长上下文驻留、其余排队。R091一个请求359 token自然EOS、另一请求正常4096上限，冻结整体失败保留。R092真实4096分块触发取消、两个survivor和同RID新room重试通过，无中间flush；768个完整输出与此前参考相同。累计2816个token与返回目标最大分数一致，这只是目标分数一致性，不能冒充独立BF16精度或物理KV oracle。

普通TP分组配置在压力夹具每rank节省约3.61GiB；D MoE窗口对照多约2GiB可用显存并支持完整启动。没有确认的端到端加速百分比。CPU描述符/分词候选有局部效果但未部署，无生产源码补丁。

恢复后先在新硬件重验原长短PD＋MTP及生命周期，冻结适合数值路径的质量判据和稳态A/A，再做单变量优化。图捕获上界、DeepEP声明容量/窗口、KV预算仍是待验证候选，不能组合猜测调参。36点独立原生实验/不适用证据、硬件可达上界和延迟/吞吐/容量前沿仍未完成；accepted_baseline=null，目标应恢复继续而不是标记完成。
'''
(M/prefix/'HANDOFF.md').write_text(handoff);changed.append(prefix+'/HANDOFF.md')
for path in [prefix+'/README.md','tuning-brainstorm/README.md']:
    text=(P/'base'/path).read_text()
    if path==prefix+'/README.md':
        i=text.index('**2026-10-19原生进展及暂停：**');end=text.index('\n\n',i)
        text=text[:i]+'**2026-10-19已按用户要求暂停：** 当前R092实验完成并已提交，R093随后停止本任务P/D及共享存储、释放两项NPU预留，模型/容器/日志保留。新服务器到位后从[暂停交接](HANDOFF.md)恢复；原长短PD＋MTP及一次分块取消/重试有条件通过，EOS整体失败保留。稳定性能基线、优化A/B与完整36机制目标尚未完成。见[停止核验](evidence/2026-10-19-910c-stopped-pause/README.md)。'+text[end:]
    else:
        i=text.index('- [910C生命周期与暂停交接：');text=text[:i]+'- [本任务已停止并暂停：两端NPU预留释放，模型/容器/日志保留，新服务器续跑入口](sglang-pd-spec-36/HANDOFF.md)\n'+text[i:]
    (M/path).write_text(text);changed.append(path)
for name in changed:
    data=(M/name).read_bytes();assert not re.search(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{20,}|\bghp_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}',data),name
(P/'changed-files.json').write_text(json.dumps(changed,indent=2)+'\n');print(json.dumps({'files':len(changed),'bytes':sum((M/f).stat().st_size for f in changed),'services_stopped':3,'leases_released':2}))
