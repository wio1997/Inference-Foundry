#!/usr/bin/env python3
"""Persist bounded latent-delivery judgement and the next unreviewed native entry."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
from update_request_lifecycle_maps import prepend

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit/execution-search'
REPORT='latent双缓存整段与保留GEMM净价判断.md'
NOTE=('介绍：从四KV源输入到FP32投影/pool/ring、BF16 latent、index-K量化双发布和同latent原地long-RoPE/发布。'
'到了哪里：256首token源窗/224发布/40无query仍发布、133 typed子集匹配；另112完整decode诊断周期/448源窗/38条件。'
'实际token-aligned C2保留清零、原RMS和pool读旧ring先于尾写；wk末读前不得跨流原地RoPE。'
'结果：尾段span全免条件P90 .2802%；保留原GEMM其余全免 .4856%；仅挪早理想重叠 .7990%，结束同独立1%小方案。'
'完整root含cast/zero/span全免1.1518%、half/h20us .5545%；扩大到含保留query/SWA的输入ready联合parent为2.1327%、half/h20us1.0449%，仅待确认，不与旧C2/host/QLI/SMLA相加。'
'graph样本root span全免周期最大.9972%，非全请求/六普通收益；本轮无新实现/原语/性能准入。'
'下一段已进DFC GMM1真实tile排程，10源hash一致仅所列range审阅；GMM2/BlockMmad/fixpipe/SwiGLU仍待扫。全链路未扫全，每替换原语逐位一致。')


def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main():
    ident=json.loads((OUT/'dfc-gemm-tile-execution-identity.json').read_text());sources=[]
    for r in ident['sources']:
        p=Path(r['local_path']);assert sha256(p.read_bytes()).hexdigest()==r['runtime_sha256']
        ranges=[]
        if p.name=='dispatch_ffn_combine.h':ranges=[[1,80],[190,280]]
        elif p.name=='dispatch_ffn_combine_kernel.hpp':ranges=[[15,55],[205,230],[435,535]]
        sources.append(dict(path=str(p),runtime_path=r['runtime_path'],sha256=r['runtime_sha256'],reviewed_ranges=ranges))
    entry=dict(status='ENTRY_ONLY_ACTUAL_DFC_GMM1_SCHEDULE_NOT_FULL_GMM_OR_VALUE',whole_chain_complete=False,every_replacement_bitwise=True,sources=sources,
        introduced='Actual routed GMM1 FP16 tile -> dequantSwiGLU INT8 rows+scales -> GMM2 FP16 -> pertoken combine/remote return, including peer input/state waits.',
        reached=['Selected wrapper L1tile128x256x512/L0tile128x256x128, asynchronous fixpipe policy, FP16 CType, two workspace stages; not sharedINT32 Up.','AIC invokes allGMM1 then waitsV2C thenGMM2; AIV DispatchAndCombine supplies peer gathered rows and quantized activation.','GMM1 currentM comes from receiver cumsum, clips maxOutputSize, loads per-expert tensor lists/scales; M<=128 disables weightL2 cache hint. Empty sender alone cannot delete receiver work.','Rotating startCoreIdx distributes tile loops across expert groups. Two granularity publications use SynchronizeBlock/Finalize C2V; actual tile/copy/FP16 completion semantics not read yet.','GMM1 gmC expert offset advances currentM*problemK while logical output width isproblemN; inspect selected LayoutC/activation addressing before proposing compact storage. This is a review question, not a proved wasted-stride optimization.'],
        pending=['Read GMM2 remaining scheduling and actualBlockShape/legalM/list/NZ branches.','Read selected BlockMmad + copyL0C/fixpipe policy and actual swiglu epilogue consumers/scales/flags, including resource lifetimes and original FP16/INT8 boundaries.','Bind matched wholeDFC cost/peer critical exposure before splitting local activation fusions; current10pins are not full body coverage or live tiling.','Standalone shared4/8 native gates do not prove co-residency with actual20AIC40AIV DFC. Preserve prior failed changedM/empty-credit recipes.'],
        result='待确认：new source-derived unreviewed native entry, no full value/implementation or bitwise admission.',new_NPU_work=0,model_loads=0,service_requests=0,new_profile_collection=0,remote_writes=[],service_changes=[])
    write(OUT/'dfc-gemm-tile-entry.json',entry)
    x=json.loads((OUT/'index-key-publication-profile.json').read_text());g=json.loads((OUT/'index-key-decode-publication-profile.json').read_text())
    frontier=dict(status='BOUNDED_LATENT_SOURCE_DELIVERY_VALUE_PIVOTED',whole_chain_complete=False,every_replacement_bitwise=True,
        report=REPORT,source_review='index-key-publication-source-review.json',
        first_token=dict(windows=len(x['windows']),active=224,cache_only=40,geometry=x['geometry'],matched_signature_subset_conditions=len(x['independent_signature_subset_conditions']),conditional_values=x['conditional_values'],raw=dict(path='index-key-publication-profile.json',sha256=sha256((OUT/'index-key-publication-profile.json').read_bytes()).hexdigest())),
        graph_diagnostic=dict(periods=len(g['steps']),sources=len(g['windows']),matched_signature_subset_conditions=len(g['signature_subset_conditions']),summary=g['summary'],conditional_values=g['conditional_values'],cohort=g['cohort'],limits=g['limits'],raw=dict(path='index-key-decode-publication-profile.json',sha256=sha256((OUT/'index-key-decode-publication-profile.json').read_bytes()).hexdigest())),
        stopped=['Same independent1%C8 first-token tail/root-retainedGEMM epilog/pure early-root-move recipes.','Same sampledgraph standalone tail/root recipe, not an ordinary/full-request bound.'],
        confirmed_implementable=[],pending=['Input-ready/query/SWA/QLI/SMLA joint actual-ready/resource/bit/net schedule, same shared budget.','Whole original source geometry owner sharing must paycommunication/consensus/readiness; original full-M owner critical compute remains.','Native RMS/dynamicquant/scatter internals and bit gates only if larger parent becomes worthwhile; currentleaf bodies not marked reviewed.','Six ordinary and originalC8 full-decode per-source native delivery binding; no extrapolation from6/12eager or112candidate graph samples.','Actual complete-pair/slots/physical epochs and final device readers before changing representation/cache lifetime.'],
        next_entry='dfc-gemm-tile-entry.json',new_NPU_work=0,new_profile_collection=0,model_loads=0,service_requests=0,remote_writes=[],service_changes=[])
    write(OUT/'index-key-publication-frontier.json',frontier)
    p=OUT/'source-frontier.json';s=json.loads(p.read_text());s.update(status='ACTIVE_LOOP032_DFC_GMM_TILE_ENTRY',region='Latent dualcache whole-source firsttoken/diagnosticgraph prices bounded; next actual routedGMM tile/activation native delivery',completeness_established=False,
        next_action='loop032: GMM2 scheduling -> selected BlockMmad/copyL0C FP16 -> Swiglu INT8/scale consumers/flags and wholeDFC net value before child fusion; native leaf entry only.',
        index_key_publication_review=frontier,dfc_gemm_tile_entry=entry)
    write(p,s)
    for module in ['M00','M03','M11']:
        prepend('audit/modules/'+module+'.md','## 2026-11-10 latent双缓存整段与净价\n\n'+NOTE+'见[本轮判断](../execution-search/'+REPORT+')。')
    prepend('audit/modules/M06.md','## 2026-11-10 DFC专家GMM内部入口（未完成）\n\n介绍：沿实际FP16 GMM1→SwiGLU INT8/scale→GMM2→combine交付继续。到了哪里：10源当前hash一致，GMM1逐expert/receiverM/滚动core/两组C2V发布和wrapper policy入口已读；BlockMmad/fixpipe/activation内部尚未读。结果：待确认，无新候选准入；loop032继续实际消费者/整段净价，不能以10个pin当已扫。见[入口](../execution-search/dfc-gemm-tile-entry.json)。每替换原语逐位一致。')
    prepend('audit/团队开发清单.md','## 2026-11-10 latent双缓存停止小配方，继续DFC内部\n\n'+NOTE+'\n\n待确认开发：输入ready到QLI/SMLA联合、原geometry owner协议、跨层表示/地址计划；必须扣ready/通信/资源/位一致成本。已停止：上述限定尾段/保留GEMM/单独挪早。已确认可实现：本轮新增0。[详细判断](execution-search/'+REPORT+')。')
    prepend('HANDOFF.md','## 当前断点 · 2026-11-10：latent双缓存整段净价已判，进入DFC专家GMM内部'
        '\n\n**Goal ACTIVE；loop031 PIVOTED（有限源码/价格），loop032 RUNNING；全链路未扫全，每替换原语逐位一致。**'
        '\n\n- **介绍、到了哪里、结果**：'+NOTE+
        '\n- **证据**：[整段判断](audit/execution-search/'+REPORT+')、index-key-publication-{frontier,profile,source-review,dependencies,exploration-notes}、index-key-decode-publication-profile。公开summary列完整本地原件SHA并省略逐窗，不冒充全部原始数据已公开。'
        '\n- **重要限制**：原生RMS/quant/scatter内部本轮未读。实际六普通/原C8全decode逐源时序仍缺；112candidate周期仅样本。TP8合计算术不能作8倍请求信用；native root1.15%含间隙，原M几何和zero/row/ring代次不能删。'
        '\n- **环境**：171主机名核对成功，同owned容器/镜像；本轮两次只读source身份+本地profile复算，无Torch/模型/NPU计算/请求/新profile/远端写入/服务改动，未操作其它controller。analyzer MatMul族/cast+zero边界修订和本地JSON读取错误见notes，不是native bit结果。'
        '\n- **下一条待扫**：loop032 Run `run-20261110T005938Z`，[DFC-GMM入口](audit/execution-search/dfc-gemm-tile-entry.json)。已读GMM1 wrapper/type和435–535，10源pin匹配；继续GMM2→实际BlockMmad异步fixpipe/copyL0C→FP16/Swiglu INT8+scale→两组flags/GM生命周期，先整段计价。注意GMM1 logicalN与expertC stride K的实际consumer解释未核，不能直接当padding节省。4/8独立原语≠与20/40 DFC已共驻。')
    prepend('ENVIRONMENT.md','2026-11-10 loop031/032接续：171 hostname核对成功；同owned容器/镜像，6份latent依赖3旧hash匹配、3本地补齐；随后10份DFC tile源全部匹配冻结副本。只读stdlib，无Torch/模型/NPU/新profile/请求/远端写入/服务变更，未操作其它controller。原始source文件与已审范围分开。身份记录 index-key-publication-dependencies.json、dfc-gemm-tile-execution-identity.json；不当loaded bytecode/device key证明。')
    for fn in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
        p=ROOT/fn;s=p.read_text();hint='<div class="hint"><strong>2026-11-10 latent双缓存 / DFC内部：</strong>'+NOTE+'<a href="audit/execution-search/'+REPORT+'">整段价格/停止与待确认</a> · <a href="audit/execution-search/dfc-gemm-tile-entry.json">下一处实际入口</a>。</div>'
        if hint not in s:s=s.replace('<main>','<main>\n'+hint,1)
        s=s.replace('<b>loop-018 ACTIVE</b>实际三层draft/TP-EP联合producer','<b>loop-032 ACTIVE</b>DFC专家GMM内部交付')
        p.write_text(s)
    payload=[]
    names=[REPORT,'index-key-publication-frontier.json','index-key-publication-source-review.json','index-key-publication-dependencies.json','index-key-publication-exploration-notes.json','dfc-gemm-tile-entry.json','dfc-gemm-tile-execution-identity.json','source-frontier.json']
    for n in names:payload.append(dict(path='handoff/evidence/'+n,content=(OUT/n).read_text()))
    for n,drop in [('index-key-publication-profile.json',['windows','independent_signature_subset_conditions']),('index-key-decode-publication-profile.json',['windows','steps','signature_subset_conditions'])]:
        p=OUT/n;d=json.loads(p.read_text());projection={k:v for k,v in d.items() if k not in drop};projection['projection']=dict(full_local_sha256=sha256(p.read_bytes()).hexdigest(),omitted_keys=drop,full_local_path=str(p),omission_scope='Full exact-task windows/signature records remain locally; summary is explicitly not the full raw dataset.')
        payload.append(dict(path='handoff/evidence/'+n,content=json.dumps(projection,ensure_ascii=False,indent=2)+'\n'))
    for n in ['analyze_index_key_publication.py','analyze_index_key_decode_publication.py','read_index_key_publication_dependencies.py','read_dfc_gemm_tile_identity.py','update_index_key_publication_maps.py']:
        payload.append(dict(path='tools/'+n,content=(ROOT/'tools'/n).read_text()))
    for r in json.loads((OUT/'index-key-publication-dependencies.json').read_text())['sources']:
        p=Path(r['local_path'])
        if 'index-key-producer-runtime-source' in str(p):payload.append(dict(path='handoff/evidence/index-key-producer-runtime-source/'+p.as_posix().split('index-key-producer-runtime-source/',1)[1],content=p.read_text()))
    tmp=Path('/tmp/index-key-publication-payload.json');tmp.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
    print(json.dumps(dict(local_evidence_files=len(names)+2,public_evidence_tool_source_files=len(payload),payload_characters=len(tmp.read_text()),entry_sources=len(entry['sources']),new_NPU_work=0),ensure_ascii=False))

if __name__=='__main__':main()
