# -*- coding: utf-8 -*-
"""Audit G4, integrate scoped observations and preserve unresolved evidence."""
from pathlib import Path
from collections import Counter
import datetime,hashlib,json
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads((T/p).read_text())
def write(p,x): (T/p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256((T/p).read_bytes()).hexdigest()
g=read('graphs/source-templates-G4.json');ts={t['id']:t for t in g['templates']};assert not g['whole_model_complete']
for t in g['templates']:
 ns=[n['id'] for n in t['nodes']];assert len(ns)==len(set(ns)),t['id'];ids=set(ns)
 for e in t['edges']:assert e['producer'] in ids and e['consumer'] in ids,(t['id'],e)
d=ts['DSPARK_BOUNDARIES'];pairs={(x['producer'],x['consumer']) for x in d['edges']}
assert ('logits','accept') not in pairs and ('norm','confidence') not in pairs
for i in range(5):
 for pair in [('seed' if i==0 else f'sample_{i-1}',f'embed_{i}'),(f'embed_{i}',f'bias_{i}'),(f'bias_{i}',f'add_{i}'),('logits',f'add_{i}'),(f'add_{i}',f'sample_{i}'),(f'sample_{i}','draft_token_output')]:assert pair in pairs,pair
e=ts['ENGRAM'];en={n['id']:n for n in e['nodes']};ep={(x['producer'],x['consumer']) for x in e['edges']};assert en['hash']['resource']=='NPU' and en['lookup']['resource']=='CONDITIONAL_NPU_OR_CPU'
for pair in [('lookup','rows_collective_pad'),('rows_collective_pad','ready'),('mask_derive','mask_ready'),('mask_ready','gate'),('gate','gate_result')]:assert pair in ep,pair
r=read('records.json')
for x in r['fusion_candidates']:
 if x.get('candidate_graph_path') and x.get('candidate_graph_sha256'):assert sha(x['candidate_graph_path'])==x['candidate_graph_sha256'],x['id']
for fid in ['F24','F27','F28','F30']:assert next(x for x in r['fusion_candidates'] if x['id']==fid)['candidate_graph_path'].endswith('-G4.json')
clock=read('evidence/clock-source-and-anchor-audit.json');assert len(clock['rows'])==16 and clock['spans']=={'device_offset_ns':9520,'host_offset_ns':4760,'host_intercept_ns':100}
for z in clock['sources'].values():assert sha(z['snapshot_path'])==z['sha256']
b=read('evidence/fixed-output-conditional-bound.json');assert [x['total_text_utf8_bytes'] for x in b['samples']]==[32628,33024] and b['not_universal_or_numeric']
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
write('evidence/g4-clock-bound-integration-audit.json',{'timestamp':now,'checks':['Graph node IDs and edge endpoints valid','Five indexed Markov/sample chains verified','Engram lookup and mask readiness paths represented','Candidate SHA256 paths match','16-rank clock anchors and source file SHA256 reconstructed','Fixed output byte counts and conditional status preserved'],'not_verified':['Complete primitive graph, alias and all consumers','Physical cross-rank clock error bound','Uncompressed protocol and B_out_max','Universal numerical throughput upper bound'],'status':'INTEGRITY_ONLY_NOT_SEMANTIC_EXHAUSTION'})
s=read('STATE.json');s.update(last_update=now,checkpoint_status='G4_FUSION_BOUNDARIES_AND_CLOCK_EVIDENCE_REVIEWED',next_read_only_action='Inspect missing Engram and eager module attribution and G4 downstream candidate semantics; hardware and output capacities remain UNKNOWN.');write('STATE.json',s)
with (T/'REPORT.md').open('a') as f:f.write('''\n\n## 时钟元数据与条件输出约束\n\n对 16 个 rank 的采集锚点和当前镜像 profiler 源码做了可复现核对：device wall/raw 截距观察跨度 9.520 微秒，host 为 4.760 微秒，host counter 截距为 0.100 微秒。这些是不同时间采集的元数据跨度，不能视为跨设备事件同步误差上界；原先跨 DP 配对及时间差仍是有条件观察。源码快照、哈希、逐 rank 数据与复现脚本见 `evidence/clock-source-and-anchor-audit.json` 和 `tools/capture_clock_metadata.py`。\n\n两轮固定输出各 8192 token，其文本 UTF-8 载荷分别为 32628 / 33024 字节。若指定接口必须在计时内分别无压缩交付这些确定文本，且接口硬容量上界 B 已证明，则分别有 `T >= D/B`，吞吐 `U <= 8192 B/D`。计时覆盖本机客户端接收，B 是相关本地路径容量而非自动等于网卡带宽。压缩、复用及硬容量缺证据，因此仅是条件符号界，不能推广为整个模型无条件数字上界；见 `evidence/fixed-output-conditional-bound.json`。\n\n更宽实现域若能证明某统一工作量 F 必须完成，且所有资源池对同一工作单位分别具有硬速率上限 C_i，则放松资源约束得到 `T >= F/sum(C_i)`；计算、访存、通信如可重叠，已有有效下界取最大值，不可机械相加。当前 F、不同资源可比速率及共享容量约束缺乏证明，不能把已有指定稠密 INT8 MAC 数代入当作普遍必要工作量。\n''')
print(json.dumps({'status':'PASS_SCOPED_REVIEW','timestamp':now,'overall_goal':'ACTIVE','global_fusion_exhaustion':False,'physical_error_bound':'UNKNOWN'}))
