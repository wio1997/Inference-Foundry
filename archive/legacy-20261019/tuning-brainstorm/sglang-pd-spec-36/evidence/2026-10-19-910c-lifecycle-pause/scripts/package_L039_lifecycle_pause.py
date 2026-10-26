"""Publish honest EOS failure, native cancellation/reuse, and user-requested pause handoff."""
import copy, hashlib, json, pathlib, re, shutil
R=pathlib.Path(__file__).resolve().parents[1];M=R.parent/'repo-export';P=R/'publication-L039-lifecycle-pause'
prefix='tuning-brainstorm/sglang-pd-spec-36';sub=prefix+'/evidence/2026-10-19-910c-lifecycle-pause';O=M/sub;O.mkdir(parents=True,exist_ok=False)
changed=[];seen=set();views=[]
def transfer(name):
    if name in seen:return
    seen.add(name);p=O/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/name,p);changed.append(sub+'/'+name)
def write(name,text):
    p=O/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);changed.append(sub+'/'+name)
def view(tag,suffix):
    name=f'runs/{tag}-{suffix}/remote-result.json';data=(R/name).read_bytes();v=copy.deepcopy(json.loads(data))
    c=v['client']['result'] if tag=='L039-R091' else v['client']['data']['result'];omitted=0
    for response in c.get('batch_responses',{}).values():
        response['body_summary']=[{'id':x['meta_info']['id'],'prompt_tokens':x['meta_info'].get('prompt_tokens'),'completion_tokens':x['meta_info'].get('completion_tokens')} for x in response['body']];del response['body']
    for response in c['requests'].values():
        body=response.get('body')
        if isinstance(body,dict) and 'prompt_token_ids' in body:
            ids=body.pop('prompt_token_ids');omitted+=len(ids);body['prompt_token_ids_fingerprint']={'count':len(ids),'sha256_json_compact':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'validated_exact_in_remote_worker':True}
    provenance={'original_local_path':str(R/name),'original_sha256':hashlib.sha256(data).hexdigest(),'original_bytes':len(data),'changes':'Replace echoed prompt ID arrays by count/hash; remove duplicated batch bodies. Keep all output IDs/text, logprobs, target-score rows, timings, failures, source guards, model logs and exit receipts. No sample/error filtered.','replaced_prompt_ID_count':omitted,'frozen_scope_text_is_not_verdict':True}
    v['public_view_provenance']=provenance;out=f'runs/{tag}-{suffix}/public-view.json';write(out,json.dumps(v,ensure_ascii=False,indent=2)+'\n');views.append({'view':out,**provenance})
for number,suffix in [(91,'pd-functional'),(92,'pd-lifecycle')]:
    tag=f'L039-R{number:03}';transfer('specs/'+tag+'.json')
    for name,digest in json.loads((R/f'specs/{tag}.json').read_text())['inputs'].items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
        transfer(name)
    for s in ['zcode','frozen-controller',suffix]:transfer(f'runs/{tag}-{s}/receipt.json')
    view(tag,suffix)
    for filename in ['manifest.json','result.json']:
        paths=list((R/'tasks/sglang-pd-spec-36/loops/L039-910c-runtime-recovery/runs').glob('*/'+tag+'/'+filename));assert len(paths)==1;transfer(str(paths[0].relative_to(R)))
for name in ['notes/L039-eos-review.json','notes/L039-cancel-review.json','notes/L039-group-memory-zcode-audit.json','notes/L039-server-migration-handoff.json','notes/L039-next-memory-candidate.json','runtime-contract.json','CONTRACT.md','scripts/audit_L039_group_memory.py','scripts/package_L039_lifecycle_pause.py']:transfer(name)
write('public-view-sources.json',json.dumps(views,indent=2)+'\n')
write('README.md','''# 910C：自然结束、分块取消、同ID重试与暂停交接

按用户要求，完成当前R092后提交记录并暂停，等待新服务器。原GLM-5.3-W8A8C8、Ascend、PD、MTP EAGLE三步/topk1/draft4、TP16/EP16、BF16 KV与原角色graph配置保留；无生产源码补丁。整体仍E0，accepted_baseline=null，未建立稳定性能基线或优化A/B。

## R091：保留整体失败，分别记录自然EOS和上限结束

这是额外生命周期用例：原两条512-token输入，greedy、max_new_tokens=4096、ignore_eos=false，无人工停止词或logit bias。原512→128、32768→256基准未改变。冻结断言要求两条都自然生成EOS，此条件未满足，内层真实退出1，Run保留FAIL。

| 请求 | 实际输出 | 结束原因 | 可支持的结论 |
| --- | ---: | --- | --- |
| 0 | 4096 | length，无EOS | 正常达到输出上限；不能要求任意模型输出必在预算内自然结束 |
| 1 | 359 | stop，最后ID154827 | 第359个token为配置EOS，此后无多余输出 |

四个P/D响应均HTTP200，原prompt IDs逐个匹配，P首token等于D首token，MTP接受直方图与计数一致，最终两端ready与空闲flush均200。这是一个自然EOS分支及一个上限分支的实测，不能将原Run改写为PASS。原客户端scope字符串在失败时也会写入，因此scope不是结论；以实际响应、断言和失败回执为准。

## R092：真实分块取消、其余请求完成、同ID换房间重试

额外生命周期测试保留原三条32768输入、每条256输出。先单独提交victim的真实P/D请求；44日志确认首个4096-token分块、剩余28672 token后，向46客户端传递事件，再释放两条survivor的四个独立HTTP线程，并对两端发送唯一victim RID的取消。引擎使用RID前缀匹配，所以三个RID特意互不为前缀；没有abort_all或服务重启。

取消API两端均200，但通过判据还要求victim实际结束：P返回HTTP200、finish=abort、零输出token；D返回HTTP500，消息含同一RID、原bootstrap_room及明确的`Aborted by AbortReq.`。实际源码中接收器abort会标记KV传输失败，Decode将其转成500；这是冻结前允许的显式取消表示，不能作为普通生成成功HTTP200，也不是从任意500推断正确。

| 完成请求 | 输入/输出token | MTP verify | 正确/提出草稿 | 完整输出 |
| --- | --- | ---: | --- | --- |
| survivor-1 | 32768 / 256 | 91 | 164 / 273 | 与R090请求1哈希相同 |
| survivor-2 | 32768 / 256 | 100 | 156 / 300 | 与R090请求2哈希相同 |
| victim重试 | 32768 / 256 | 92 | 163 / 276 | 与R090请求0哈希相同 |

原victim与两个survivor全部响应结束后，复用victim的同一RID、换新bootstrap_room重试。取消、survivor与重试之间**没有主动flush**。三条完整结果逐个prompt ID及P/D边界匹配，全部768个输出token等于各自返回目标最大分数，接受直方图守恒，完整输出哈希与冻结R090参考相同。父进程wait确认客户端真实退出0；最后两端ready与空闲flush均200，原P067/D073精确CID仍运行。

测试通过的是一次事件驱动的分块取消、survivor隔离及同RID/新房间复用，不替代三条同时到达的性能用例。P前端的通用“waiting queue”取消文案无法标识精确内部队列；阶段依据是首个分块时仅victim已提交，随后P在完整prefill前以零输出终止。已有分块可能仍在执行，不能声称即时取消或从跨机器时钟相减推算延迟。

功能复用和最终空闲flush不等于物理KV逐字节/每页代际与引用数oracle，也不证明长期无泄漏、全部断连/重复取消场景或独立模型精度。累计目标分数一致性观察为2816个token（此前2048，本轮768），仍不是BF16独立金标准。所有时延只作生命周期诊断，不报告加速比。

## 续跑交接

本轮后不再派发新推理实验。记录原模型文件、源码、镜像与通信配置，以及未执行的graph/内存候选。新服务器必须重新盘点资源归属、校验模型/环境、建立新运行合同，再验证原长短PD＋MTP及生命周期；旧硬件结果不能直接成为新服务器性能基线。之后先冻结适合数值路径的质量判据与稳态A/A，再做单变量优化。36机制覆盖、硬件可达上界和端到端优化仍未完成。

- [迁移与暂停交接](notes/L039-server-migration-handoff.json)：包括最新服务处置状态；没有在公开文件保存凭据。
- [EOS失败的独立复核](notes/L039-eos-review.json)、[取消/复用独立复核](notes/L039-cancel-review.json)。
- [R091全部输出与失败视图](runs/L039-R091-pd-functional/public-view.json)、[R092完整输出、分数及退出视图](runs/L039-R092-pd-lifecycle/public-view.json)。
- [原始数据SHA及视图变换说明](public-view-sources.json)：回执SHA对应原始数据，公开视图另命名；未删失败、输出或计时样本。
- [40条冻结命令各执行一次的元数据审计](notes/L039-group-memory-zcode-audit.json)，不公开ZCode执行器完整模型I/O。
- [上个检查点：原三路32K两次完成及实际容量边界](../2026-10-19-910c-target-scores-long/README.md)。
''')
path=prefix+'/mechanisms.json';old=json.loads((P/'base'/path).read_text());new=copy.deepcopy(old)
states={'M004':'真实分块victim取消、两条survivor与同RID新房间重试通过；性能基线待建立','M005':'32K取消后的survivor/同RID重试无中间flush通过；物理页代际与长期回收未证','M006':'原4096 chunk日志触发取消，survivor及重试完整32K通过；未做chunk性能A/B','M031':'累计2816个输出token与返回目标最大分数一致；EOS与取消重试有条件覆盖，独立精度/性能未闭合','M033':'原生EOS154827第359token结束且无额外输出；另一请求正常4096上限，冻结整体失败保留','M035':'实际32K分块取消/显式传输abort、survivor与同RID新房间复用通过；物理KV/长期drain未全证'}
for x in new:
    if x['id'] in states:
        x['status']=states[x['id']];x['native_lifecycle_R091_R092']={'EOS_run':'FAIL_all_prompts_EOS_predicate','natural_EOS_observed_one':True,'length_cap_observed_one':True,'cancel_survivor_retry':'PASS_CONDITIONAL','intermediate_flushes':0,'full_completed_outputs':3,'target_max_checks':768,'same_RID_new_room':True,'physical_KV_oracle':False,'performance_gain':'NOT_MEASURED','evidence':'evidence/2026-10-19-910c-lifecycle-pause/README.md'}
assert len(new)==36 and all(a==b for a,b in zip(old,new) if a['id'] not in states)
(M/path).write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n');changed.append(path)
for path in [prefix+'/README.md','tuning-brainstorm/README.md']:
    text=(P/'base'/path).read_text()
    if path==prefix+'/README.md':
        i=text.index('**2026-10-19原生进展：**');end=text.index('\n\n',i)
        text=text[:i]+'**2026-10-19原生进展及暂停：** 原长短PD＋MTP功能已实测；本轮追加一次真实32K分块取消、两条survivor与同RID新房间重试，无中间flush，768个输出与此前参考/返回目标最大分数一致。自然EOS一条通过、一条达到4096上限，原整体失败保留。按用户要求完成本轮、提交交接后暂停，等待新服务器；稳定性能基线与完整36机制工作未完成，整体E0。见[生命周期与迁移交接](evidence/2026-10-19-910c-lifecycle-pause/README.md)。'+text[end:]
        lines=text.splitlines()
        for i,line in enumerate(lines):
            for id,status in states.items():
                if line.startswith('| ['+id+']'):
                    parts=line.split('|');parts[-3]=' '+status+' ';lines[i]='|'.join(parts)
        text='\n'.join(lines)+'\n'
    else:
        i=text.index('- [910C原长用例：');text=text[:i]+'- [910C生命周期与暂停交接：32K分块取消、survivor与同ID重试通过，EOS整体失败保留](sglang-pd-spec-36/evidence/2026-10-19-910c-lifecycle-pause/README.md)\n'+text[i:]
    (M/path).write_text(text);changed.append(path)
for name in changed:
    data=(M/name).read_bytes();assert not re.search(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{20,}|\bghp_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}',data),name
(P/'changed-files.json').write_text(json.dumps(changed,indent=2)+'\n')
print(json.dumps({'files':len(changed),'bytes':sum((M/f).stat().st_size for f in changed),'mechanisms_updated':list(states),'baseline':None}))
