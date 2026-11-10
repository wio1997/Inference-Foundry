#!/usr/bin/env python3
"""Record actual native DFC resource correction/value and the allocator frontier."""
from hashlib import sha256
import json
from pathlib import Path
from update_request_lifecycle_maps import prepend
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit/execution-search'
REPORT='DFC原生交付与24核资源更正.md'
NOTE=('介绍：沿DFC receiver行→GMM1 FP16→SwiGLU INT8/scale→GMM2 FP16→combine，核原位数值、flags和别名后价整段。'
'到了哪里：31实际source pins/所列ranges；16旧DB逐SHA、2560任务时间/globalid匹配，全部MIX_AIC block24/mix48；8 typed成本与24专家计数匹配。'
'结果：撤回“DFC固定20/40留4/8空核”，40只是内部routing tiler参数；独立shared4/8原语事实保留，不准入共驻。'
'C1/C2、A/activation已分别共bank；compact4608单独容量信用0，2171计数窗有潜在WAR见证。'
'arch2201 L0C→UB为空unsupported、TSCM仍GM staging，同现成接口删GM配方停止。'
'大DFC共同尾段全免条件P9020.09%–22.69%；阶段占25%/半省/h5us为2.458%–2.783%，阶段耗时/曝光/位一致/资源代价仍待确认。'
'960MiB是原payload共bank源公式，非新释放显存；下一段已追到GetWorkspaceSize→at::empty/raw pointer→OpCommand callback入口，allocator/queue/capture/末读尚未读。'
'本轮无新实现/原语/性能基线，全链路未扫全，每替换原语逐位一致。')

def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main():
 x=json.loads((OUT/'dfc-gemm-tile-profile.json').read_text());launch=json.loads((OUT/'dfc-archived-launch-fields.json').read_text())
 ident=json.loads((OUT/'native-workspace-execution-identity.json').read_text());sources=[];missing=[]
 for r in ident['sources']:
  if r.get('missing'):missing.append(r['runtime_path']);continue
  p=Path(r['local_path']);assert sha256(p.read_bytes()).hexdigest()==r['runtime_sha256'];rr=[]
  if p.name=='op_api_common.h':rr=[[700,772]]
  elif p.name=='dispatch_ffn_combine_torch_adpt.h':rr=[[15,93]]
  sources.append(dict(runtime_path=r['runtime_path'],path=str(p),sha256=r['runtime_sha256'],reviewed_ranges=rr))
 entry=dict(status='ENTRY_ONLY_NATIVE_WORKSPACE_MACRO_AND_DFC_ADAPTER',whole_chain_complete=False,every_replacement_bitwise=True,sources=sources,missing_guessed_cpp=missing,
  introduced='Actual native caller GetWorkspaceSize -> at::empty Byte -> raw workspace pointer captured by custom handler -> OpCommand::Run; follow allocation/cache/capture and final device consumers, not source requested bytes alone.',
  reached=['DFC int8 branch dispatches aclnnDispatchFFNCombine; W4/BF16 branches separately visible but not frozen INT8 cost.','Macro uses current stream(false), ConvertTypes, original GetWorkspaceSize/executor, byte tensor allocation, captures rawpointer/size/executor/converted params and launches via SetCustomHandler/Run.','workspace_tensor declared in the if block; whether raw pointer remains safe/reusable depends on actual allocator/queued execution/stream capture semantics. Source syntax alone is not a proved use-after-free or release.'],
  unreviewed=['OpCommand Run/custom handler inline vs taskqueue and thread lifetime; installed header identities only.','Actual CachingAllocator stream ownership/free-list/record-event/capture pools and LibApi executor scratch.','Existing matching profiler memory allocation/capture records and pointer/size high-water union acrossDFC/QLI/SMLA, not perlayer addition.','Safe sameM/precision capacity bound use, original prefill peak, queued other streams and final device-reader epoch before reusing storage.'],
  result='待确认：actual producer/callback entry only, no physical release/net value or primitive admission.',new_NPU_work=0,model_loads=0,service_requests=0,new_profile_collection=0,remote_writes=[],service_changes=[])
 write(OUT/'native-workspace-entry.json',entry)
 correction=dict(status='WITHDRAW_FIXED20_40_WHOLE_DFC_PREMISE',old_claim='Entire actualDFC fixed20AIC40AIV leaves4/8spare on24/48.',reason='BLOCK_NUM20 -> aivNumInitRouting40 only feeds innerrouting tiler. Whole info.aivNum/blockDim read platform; exact archived tasks all24/48 MIX_AIC.',actual_archived_launch_groups=launch['groups'],exact_task_binding=2560,all16_database_SHA_matched=True,launch_proof=dict(path='dfc-archived-launch-fields.json',sha256=sha256((OUT/'dfc-archived-launch-fields.json').read_bytes()).hexdigest()),
  source='dispatch_ffn_combine_tiling.cpp:170-182,246-252,262-275; actual sameSHA current/frozen source. Not percall loaded tiling buffer.',independent_shared4_8_primitive='Preserved limited M416 byte/launch observations; no fullDFC co-residency or freed core grant.',
  old_reports_immutable=['共享专家4核原语准入.md','shared-4cube-service-gate-summary.json','moe-publication.json'],
  current_decision='Stop recipe relying on fixed idle4/8. Explicit DFC resource reservation/nonexclusive schedule/joint kernel still pending true resource/bit/net cost. No stage timing from whole task block count.')
 write(OUT/'dfc-resource-correction.json',correction)
 frontier=dict(status='BOUNDED_NATIVE_DFC_DELIVERY_RESOURCE_CORRECTED_AND_PAID_PRICE',whole_chain_complete=False,every_replacement_bitwise=True,report=REPORT,source_review='dfc-gemm-tile-source-review.json',resource_correction=correction,
  profile_raw=dict(path='dfc-gemm-tile-profile.json',sha256=sha256((OUT/'dfc-gemm-tile-profile.json').read_bytes()).hexdigest()),population=x['populations'],source_resource_formula=x['source_resource_formula'],alias_geometry=x['alias_geometry'],integer_partial_sum_bound=x['integer_partial_sum_bound'],
  conditional_values=[{k:v for k,v in s.items() if k!='requests'} for s in x['conditional_values']],one_percent_stage_thresholds=x['one_percent_stage_thresholds'],shared_reservation=[{k:v for k,v in s.items() if k!='requests'} for s in x['shared_resource_reservation_scenarios']],
  stopped=['Fixed20/40 leaves4/8core recipe; actual archivedwholeDFC24/48.','DirectUB/renamedTSCM path claiming goneGM under selected current APIs.','CompactC1 stride/four-bank or FP16/max/scale deletion without actual reader contract.'],confirmed_implementable=[],
  pending=['Whole routed/shared scheduling: original phase fraction/exposure, same primitive bits, actual resource/flags and added costs.','SecondV2C wait boundary retains flag counts/original row addresses; actual DMA/bit and min(exposed wait,group20time) unknown.','Safe percall/global row capacity vs real allocator/graph/workspace high-water, prefill peak and allstream last reader.','Full originalC8 decode/six ordinary routed phase timestamps/eligible populations; no prefill extrapolation.','Included lower MMA/vector implementation details not reviewed beyond declared source ranges; native key/source equivalence not newly attested.'],
  next_entry='native-workspace-entry.json',new_NPU_work=0,new_profile_collection=0,model_loads=0,service_requests=0,remote_writes=[],service_changes=[])
 write(OUT/'dfc-gemm-tile-frontier.json',frontier)
 p=OUT/'source-frontier.json';s=json.loads(p.read_text());s.update(status='ACTIVE_LOOP033_NATIVE_WORKSPACE_ENTRY',region='Selected DFC tile/activation/alias/native transport bounded; actual24/48 launch corrects fixed20/40; actual workspace allocator/capture readers next',completeness_established=False,next_action='loop033: OpCommand::Run/customhandler queue -> NPU allocator/free/event/capture cache -> existing memory record and physical workspace union/last device readers. Macro/adapter entry only.',dfc_gemm_tile_review=frontier,native_workspace_entry=entry)
 s['MoE_review']['actual_host_cores']=dict(status='CORRECTED_FROM_EXACT_ARCHIVED_TASKS',AIC=24,AIV=48,scope='2560originaleager MIX_AIC block/mix fields; not cycle occupancy/current-call buffer.',internal_routing_tiler_AIV=40,old20_40='WITHDRAWN whole-op inference from BLOCK_NUM20/internal40; see dfc-resource-correction.json')
 s['MoE_review']['prior_48_CPU_adapter']='Standalone48CPU adapter is not actual inner routing-work proof. WholeDFC48AIV is now bound by separate exact2560 native task fields; inner tiler parameter40 remains distinct.'
 s['MoE_review']['shared_4cube_8vector_resource_correction']='Independent4/8 raw bit/launch gate remains; no fixedspare4/8, current source/archived wholelaunch24/48. Explicit reservation or other scheduling must pay cost.'
 if 'dfc_gemm_tile_entry' in s:
  s['dfc_gemm_tile_entry']['status']='HISTORICAL_ENTRY_SUPERSEDED_BY_LOOP032_REVIEW'
  s['dfc_gemm_tile_entry']['resource_premise_superseded']='Fixed20/40 wholeDFC premise withdrawn; exact archived2560 tasks24/48, internalroutingtiler40 distinct. See dfc-resource-correction.json.'
  s['dfc_gemm_tile_entry']['pending']=[v.replace('actual20AIC40AIV DFC','actual wholeDFC grant (archived24/48; old20/40 premise withdrawn)') for v in s['dfc_gemm_tile_entry'].get('pending',[])]
 write(p,s)
 for m in ['M05','M06','M07','M11']:prepend('audit/modules/'+m+'.md','## 2026-11-10 DFC原生交付、24/48实录与净价\n\n'+NOTE+'见[本轮判断](../execution-search/'+REPORT+')。旧固定20/40解释以此更正为准。')
 prepend('audit/团队开发清单.md','## 2026-11-10 DFC整段/资源前提更正\n\n'+NOTE+'\n\n值得继续：原GM/数值下的21/3与消费等待排程、付代价的资源重写、跨op workspace物理复用。实际阶段耗时/位一致/净成本未准入，不把独立4/8当共驻方案。[详细判断](execution-search/'+REPORT+')。')
 prepend('HANDOFF.md','## 当前断点 · 2026-11-10：DFC交付已判并更正24/48，进入native workspace实际分配\n\n**Goal ACTIVE；loop032 PIVOTED限定source/资源/价格，loop033 RUNNING；全链路未扫全，每替换原语逐位一致。**\n\n- **介绍、到了哪里、结果**：'+NOTE+
 '\n- **关键更正**：旧固定20/40来自误用内部routing常量40；实际wholeDFC平台tiling，原16DB SHA及2560task全部24/48 MIX_AIC。旧4/8 bit gate不撤回，闲核前提撤回；current source-frontier已改，旧Run/证据原件保留。'
 '\n- **边界/价格**：GMM1/2 INT8→INT32→VDEQF16；SwiGLU全2304维max与原clamp/Exp/Div/RINT链保留。C1/C2同640MiB、A/activation同320MiB并非四bank；压stride不能减GMM2bank且有潜在WAR。半省/h5us要1%需相关曝光阶段占共同尾段9.29%–10.49%，h20us需10.70%–12.09%。让核负值保留，非实测。'
 '\n- **证据**：[整段与更正](audit/execution-search/'+REPORT+')；dfc-{resource-correction,archived-launch-fields,gemm-tile-profile,gemm-tile-frontier,gemm-tile-source-review,gemm-tile-exploration-notes}。首个字段校验误用connection_id为global_task_id的reader/ERROR说明保留；正确列7重读通过。'
 '\n- **环境**：171 hostname核对成功，同owned容器/镜像；只读当前source/header和host原DB（容器内不挂DB路径）。全部SQL/read-only，NPU/模型/请求/新profile/显式远端写入/服务改动0，未操作其它controller。'
 '\n- **下一条待扫**：loop033 Run `run-20261110T013519Z`，[workspace入口](audit/execution-search/native-workspace-entry.json)：6实际source pin/猜测cpp missing；已读op_api_common.h700–772和DFC adapter15–93。GetWorkspaceSize→if内at::empty Byte→rawptr capture→OpCommand.SetCustomHandler/Run；继续Run/queue、allocator/free/stream/capture与现有memory records。不要仅凭tensor作用域宣称use-after-free，也不要仅凭640+320MiB公式宣称物理可释放或跨层累加。')
 prepend('ENVIRONMENT.md','2026-11-10 loop032/033：171同owned容器镜像，10DFC源hash重核一致，Catlass/CANN选择依赖31实际源pin；16个原v168 host DB逐SHA匹配、2560DFC任务精确绑定报告MIX_AIC block24/mix48，撤回固定20/40整个DFC推断，内部routing tiler40另列。只读SQL/header，无模型/NPU/新profile/服务请求/显式远端写入/服务改动。当前macro workspace来源为at::empty+rawpointer+OpCommand，allocator/capture/末读尚未核；参见[资源更正](audit/execution-search/dfc-resource-correction.json)。')
 for fn in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
  p=ROOT/fn;v=p.read_text();hint='<div class="hint"><strong>2026-11-10 DFC24/48更正 / workspace：</strong>'+NOTE+'<a href="audit/execution-search/'+REPORT+'">原生/资源/价格与缺口</a> · <a href="audit/execution-search/native-workspace-entry.json">下一处实际分配入口</a>。</div>'
  if hint not in v:v=v.replace('<main>','<main>\n'+hint,1)
  v=v.replace('<b>loop-032 ACTIVE</b>DFC专家GMM内部交付','<b>loop-033 ACTIVE</b>native workspace分配/捕获/末读')
  p.write_text(v)
 names=[REPORT,'dfc-resource-correction.json','dfc-gemm-tile-frontier.json','dfc-gemm-tile-source-review.json','dfc-gemm-tile-exploration-notes.json','dfc-gemm-tile-execution-identity.json','dfc-gemm-extra-dependencies.json','dfc-gemm-native-catalog.json','native-workspace-execution-identity.json','native-workspace-entry.json','source-frontier.json']
 payload=[dict(path='handoff/evidence/'+n,content=(OUT/n).read_text()) for n in names]
 for n,drop in [('dfc-gemm-tile-profile.json',['windows','stages','matched_cost_conditions']),('dfc-archived-launch-fields.json',['profiles'])]:
  p=OUT/n;d=json.loads(p.read_text());projection={k:v for k,v in d.items() if k not in drop};projection['projection']=dict(full_local_sha256=sha256(p.read_bytes()).hexdigest(),full_local_path=str(p),omitted_keys=drop,scope='Full exact-task/per-expert windows retained locally. Explicit summary, not full raw dataset.')
  if n=='dfc-archived-launch-fields.json':projection['database_pins']=[{k:v for k,v in p.items() if k!='rows'} for p in d['profiles']]
  payload.append(dict(path='handoff/evidence/'+n,content=json.dumps(projection,ensure_ascii=False,indent=2)+'\n'))
 for n in ['read_dfc_gemm_native_catalog.py','read_dfc_launch_fields.py','read_dfc_gemm_extra_dependencies.py','analyze_dfc_gemm_tile_delivery.py','read_native_workspace_identity.py','update_dfc_gemm_tile_maps.py']:
  payload.append(dict(path='tools/'+n,content=(ROOT/'tools'/n).read_text()))
 for n in ['reader.py','error.json']:payload.append(dict(path='handoff/evidence/dfc-launch-read-a1/'+n,content=(OUT/'dfc-launch-read-a1'/n).read_text()))
 selected=['dispatch_ffn_combine_tiling.cpp','dispatch_ffn_combine_kernel.hpp','block_mmad_preload_async_fixpipe_quant.hpp','block_epilogue_pertoken_swiglu.hpp','arch/arch.hpp','layout/matrix.hpp','kernel_operator_fixpipe_v2_impl.h','kernel_operator_fixpipe_intf_impl.h','kernel_operator_data_copy_impl.h','gemm/tile/copy_l0c_to_ub.hpp','op_api_common.h','dispatch_ffn_combine_torch_adpt.h']
 manifest=[]
 for fn in ['dfc-gemm-tile-execution-identity.json','dfc-gemm-extra-dependencies.json','native-workspace-execution-identity.json']:
  for r in json.loads((OUT/fn).read_text())['sources']:
   if r.get('missing') or not any(r['runtime_path'].endswith('/'+n) for n in selected):continue
   p=Path(r['local_path']);public='handoff/evidence/dfc-selected-source/'+r['runtime_path'].lstrip('/');payload.append(dict(path=public,content=p.read_text()));manifest.append(dict(runtime_path=r['runtime_path'],sha256=sha256(p.read_bytes()).hexdigest(),public_path=public,scope='Exact raw source, not complete source review or loaded binary proof.'))
 write(OUT/'dfc-gemm-selected-source-public-manifest.json',dict(files=manifest,review='dfc-gemm-tile-source-review.json',workspace_review='native-workspace-entry.json'))
 payload.append(dict(path='handoff/evidence/dfc-gemm-selected-source-public-manifest.json',content=(OUT/'dfc-gemm-selected-source-public-manifest.json').read_text()))
 pp=Path('/tmp/dfc-gemm-tile-public-payload.json');pp.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
 print(json.dumps(dict(public_evidence_tools_sources=len(payload),characters=len(pp.read_text()),native_workspace_actual_sources=len(sources),whole_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
