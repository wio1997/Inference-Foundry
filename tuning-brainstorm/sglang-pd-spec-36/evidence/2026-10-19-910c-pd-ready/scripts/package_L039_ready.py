"""Append actual two-engine readiness without claiming a PD baseline."""
import hashlib,json,pathlib,re,shutil
R=pathlib.Path(__file__).resolve().parents[1];M=R.parent/'repo-export';P=R/'publication-L039-ready-final';prefix='tuning-brainstorm/sglang-pd-spec-36';sub=prefix+'/evidence/2026-10-19-910c-pd-ready';O=M/sub;O.mkdir(parents=True,exist_ok=False);changed=[];seen=set()
def copy(n):
 if n in seen:return
 seen.add(n);p=O/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,p);changed.append(sub+'/'+n)
for n in [75,76]:
 tag=f'L039-R{n:03}';copy('specs/'+tag+'.json')
 for name,digest in json.loads((R/f'specs/{tag}.json').read_text())['inputs'].items():
  assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,(tag,name)
  copy(name)
 for suffix in ['zcode','frozen-controller','full-model-state']:copy(f'runs/{tag}-{suffix}/receipt.json')
 copy(f'runs/{tag}-full-model-state/stdout.json')
 for filename in ['manifest.json','result.json']:copy(f'tasks/sglang-pd-spec-36/loops/L039-910c-runtime-recovery/runs/source-contract/{tag}/'+filename)
for n in ['notes/L039-model-handles-r076.json','notes/L039-group-memory-review.json','notes/L039-group-memory-zcode-audit.json','runtime-contract.json','scripts/audit_L039_group_memory.py','scripts/package_L039_ready.py']:copy(n)
text='''# 910C原模型双端API就绪

R076只读核对：原Prefill R067（44节点）与新Decode R073（46节点）均保持Running，纯ready、model_info、server_info各200，两个精确UID资源预留Pod均Running/Ready。**完整PD外部请求与性能基线尚未建立；整体E0、accepted_baseline=null。**

两台服务器各16个逻辑NPU，共32个。每端加载同一套原GLM-5.3-W8A8C8，186权重分片、764089835336字节；全文件封存匹配证据见前序记录。ModelSlim加载、实际BF16 KV、TP16/EP16/DP1、Ascend传输/SDMA、EAGLE三步/topk1/draft4均在实际server_info确认。context33792、max_running3、静态显存比例0.90不变。Mac为控制器，所有计算和golden检查在910C。

上个检查点R074尚在加载。本次R075见target verify图完成，R076进一步确认draft decode和draft extend捕获、应用启动及内部PD warmup完成。Decode日志给出内部启动计时：weight91.81s、target_verify92.87s、draft_decode3.81s、draft_extend2.55s；这是单次启动日志，不是请求时延或吞吐结果。原先draft extend申请194MiB失败的反例保留。新配置只在D把MoE/default通信缓冲2048改为1024MB，普通HCCL仍200MB，原P维持2048MB MoE并未重启。原角色graph配置保留，D的decode backend仍为full。

实际KV容量为P43904、D43776 token；两端max_req_input_len33786。三路32768输入到达不能直接解释为三路同时驻留，后续需观察准入与排队。512→128、32768→256的原输入候选不变，输出/EOS/采样/到达/取消及质量合同须在首个真实请求前冻结。

| 角色 | 节点 | 精确自有容器 | 本次状态 |
| --- | --- | --- | --- |
| Prefill | 44 | if-mech36-l039-r067-prefill-44 | 三项API200，原服务保持 |
| Decode | 46 | if-mech36-l039-r073-decode-46 | 三项API200，MTP图与内部warmup完成 |

内部warmup的generate不代表真实P→D业务请求或语义正确性通过。观察器本身inference_requests=0，不调用可能生成请求的health接口。还没有验证外部请求的KV交接、MTP有效输出/接受分布、EOS、取消/重试和页回收，也没有A/A基线。

下一步保持这两个精确CID，先审查真实PD路由的bootstrap_room/host/port合同并冻结请求策略，然后验证短请求的实际P→D和MTP输出/回收，再扩展到原短/长到达用例及A/A噪声基线。模型运行期间不再在同一设备执行55GiB压力探针。

- [R076双端原始观测与实际server_info字段](runs/L039-R076-full-model-state/stdout.json)
- [精确句柄、环境与API记录](notes/L039-model-handles-r076.json)
- [主控审查及两组原生缓冲对照](notes/L039-group-memory-review.json)
- [24条冻结命令各一次的元数据审计](notes/L039-group-memory-zcode-audit.json)
- [前序进程退出、显存压力A/B、原D图OOM与仅D重启](../2026-10-19-910c-group-memory-bootstrap/README.md)
'''
(O/'README.md').write_text(text);changed.append(sub+'/README.md')
path=prefix+'/mechanisms.json';old=json.loads((P/'base'/path).read_text());new=json.loads(json.dumps(old))
for x in new:
 if x['id']=='M028':
  x['status']='原GLM-5.3双端PD+MTP API就绪；16rank缓冲支持性通过，真实PD输出/生命周期与性能基线待测'
  x['native_bootstrap_candidate']['group_memory_diagnostic']['D_ready']=True
  x['native_bootstrap_candidate']['two_engine_readiness']={'run':'L039-R076','P':'L039-R067','D':'L039-R073','both_API_ready':True,'MTP_graphs_and_internal_warmup':True,'full_external_PD_workload':False,'accepted_baseline':None,'evidence':'evidence/2026-10-19-910c-pd-ready/README.md'}
assert len(new)==36 and all(a==b for a,b in zip(old,new) if a['id']!='M028')
(M/path).write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n');changed.append(path)
for path in [prefix+'/README.md','tuning-brainstorm/README.md']:
 text=(P/'base'/path).read_text()
 if path==prefix+'/README.md':
  i=text.index('**2026-10-19原生进展：**');j=text.index('\n\n',i);text=text[:i]+'**2026-10-19原生进展：** 两台910C各16逻辑NPU，原GLM-5.3-W8A8C8、PD与MTP保留，双端API及内部warmup已就绪。真实外部PD输出、KV交接/回收与A/A性能基线仍待验证，整体E0，见[双端实际配置、图捕获与就绪证据](evidence/2026-10-19-910c-pd-ready/README.md)。'+text[j:]
  lines=text.splitlines()
  for i,line in enumerate(lines):
   if line.startswith('| [M028]'):parts=line.split('|');parts[-3]=' '+next(x['status'] for x in new if x['id']=='M028')+' ';lines[i]='|'.join(parts)
  text='\n'.join(lines)+'\n'
 else:
  i=text.index('- [910C分组缓冲：');text=text[:i]+'- [910C双端就绪：原GLM-5.3-W8A8C8、PD与MTP图/内部warmup通过，真实请求与基线待测](sglang-pd-spec-36/evidence/2026-10-19-910c-pd-ready/README.md)\n'+text[i:]
  text=text.replace('Prefill就绪、Decode启动中，PD＋MTP基线待完成','双端API就绪，真实PD＋MTP请求与基线待完成')
 (M/path).write_text(text);changed.append(path)
for name in changed:
 data=(M/name).read_bytes();assert not re.search(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{20,}|\bghp_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}',data),name
(P/'changed-files.json').write_text(json.dumps(changed,indent=2)+'\n');print(json.dumps({'files':len(changed),'bytes':sum((M/f).stat().st_size for f in changed),'overall':'E0_SPEC_ONLY','external_PD_workload':False}))
