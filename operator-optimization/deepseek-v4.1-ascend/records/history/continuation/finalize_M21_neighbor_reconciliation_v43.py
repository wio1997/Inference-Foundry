"""Persist finite S13 guard reconciliation while retaining original whole-scope gaps."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

from finalize_origin_coverage_v9 import N, O, R, read, write
from persist_profile_refinement_v3 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v43.json').exists()
    for a in read(O/'checkpoint-manifest-v42.json')['artifacts']:
        b=Path(a['path']).read_bytes()
        assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],a['path']
    cross=read(O/'M21-adjacent-guard-crosswalk-v43.json')
    sem=read(O/'M21-semantic-branch-inventory-v43.json')
    audit=read(O/'M21-original-requirement-completion-audit-v43.json')
    proof=read(O/'F06-F12-row-reuse-and-index-format-domain-review-v43.json')
    variantsfile='M21-neighbor-conditional-variants-v43.json'
    variants=read(O/variantsfile)
    assert len(cross['guard_dataflow_reconciled'])==63 and cross['remaining_specific_guard_crosswalk']==[]
    assert len({x['id']for x in cross['guard_dataflow_reconciled']})==63
    assert len(sem['branches'])==18 and len(sem['static_branch_sites'])==268
    assert len(audit['requirements'])==8 and not any(x['completion_proved']for x in audit['requirements'])
    assert proof['row_local_quantizer_examples']==5 and proof['index_format_cases']==36
    assert len(variants['variants'])==2 and len(variants['conditions'])==14
    assert not(O/'M21-quantize-wrapper-point-source-v43.json').exists()
    evidence=[
        'M21-remaining-guard-point-source-v43.json',
        'M21-reused-point-source-receipts-v43.json',
        'M21-shared-index-bank-allocation-point-source-v43.json',
        'M21-topology-index-width-field-source-v43.json',
        'M21-adjacent-guard-crosswalk-v43.json',
        'M21-neighbor-guard-and-index-publication-contract-v43.json',
        'F06-F12-row-reuse-and-index-format-domain-review-v43.json',
        variantsfile,'M21-neighbor-publication-graph-v43.json',
        'M21-semantic-branch-inventory-v43.json',
        'M21-original-requirement-completion-audit-v43.json',
        'M22-saved-profile-local-payload-availability-v43.json',
    ]
    common=dict(guard_crosswalk=str(O/evidence[4]),contract=str(O/evidence[5]),
                conditions=str(O/variantsfile),implementation=False,new_performance_result=False)
    candidates={
        'F06':dict(review_v43=dict(common,relation='Existing metadata-identity/type/communication guarded quant reuse analyzed; partial query/cache row alternative requires equivalent nontrivial row quantizer and exact row-ready routing.'),
            variants_v43=[v for v in variants['variants']if v['parent']=='F06'],
            profile_review_state='INPUT_SHARING_GUARDS_AND_DISJOINT_CP_ROWS_RECONCILED;NO_CURRENT_QUANT_SAVINGS_COUNT',
            conclusion_changing_gap='Positive Q_A/KV primitives are unquantized; exact optional quantizer body not locally frozen. Do not infer nontrivial quant work or equal quantizers from method class, shapes or metadata. Altered row/weight/event/layout equivalence remains conditional.'),
        'F12':dict(review_v43=dict(common,relation='Role/candidate/length/publication branches and local query-row readers bound. Existing postprocess already accepts direct output bank; ordinary same-width INT32 bank formatting need not execute padding/cast kernels.'),
            variants_v43=[v for v in variants['variants']if v['parent']=='F12'],
            profile_review_state='SOURCE_POSTPROCESS_OUTPUT_AND_SHARED_BANK_READERS_BOUND;NORMAL_FORMAT_VIEWS_BASELINE',
            conclusion_changing_gap='Exact prepare_indexer_indices return alias and actual copy_/version/capture observers remain unverified; no current redundant-copy savings counted. Candidate block IDs and token-index mapping remain distinct; actual same-forward/next-forward bank generations are not proven by shape/address.'),
        'F31':dict(review_v43=dict(common,relation='All63 adjacent static guard dataflows reconciled. Shared reset is no-op; optional lengths and zero-cache publication are explicit. Preserve captured/all-reader/reuse generation audit as S15.'),
            profile_review_state='ALL_63_S13_GUARD_DATAFLOWS_RECONCILED;FINITE_NEIGHBOR_FRONTIER_REMOVED',
            conclusion_changing_gap='S13 finite guard gap resolved; exact S01/S14 helper predicates, main/nested graph/QLI bank/all readers, same-denominator resource and complete-process transport ranges remain.',
            next_profile_or_contract_action='Advance M22 actual pinned full graph/body/bank evidence and M23-M25/global26 scope. Compact summaries/old169 samples do not replace current80graph body. No repeat63guard audit.'),
        'F09':dict(review_v43=dict(common,relation='Index-source/long-context roles and five-term quantize guard, query/weight inputs, RoPE order and quantize_done consumer wait explicitly reconciled; floating query and optional quantized query remain separate.')),
        'F08':dict(review_v43=dict(common,relation='Packed mixed/prefill SWA writes on main after aux join; other selected cases write on aux before join. Preserve kind=win/flat-vs-coordinate slots and exactly-one publication.')),
        'F17':dict(review_v43=dict(common,relation='C1 BF16 versus C2 FP32 projections/pool, COMPRESSOR metadata wait and source cos/sin fallback are source-case bound; old index-latent read before long RoPE retained.'),
            conclusion_changing_gap='Local two-projection/pool/state consumer source contracts are already answered by v24/v27/v39 and current guard reconciliation. Actual owner/generation/mask/alias/native rounding/captured reader and causal gain remain conditional; same-shape GEMMs alone are not their proof.'),
        'F33':dict(review_v43=dict(common,relation='Reuse v27 old-latent WAR/independent norm/stream-overlap proofs; source-case crosswalk adds no new pair count or idle-gap saving. Main-vs-aux cache publication retained.')),
    }
    fw={
        'H06':dict(review_v43=dict(common,relation='Shared-state reset issues no clear; existing direct output/empty-cache initialization and normal index formatting views are baseline. Do not count phantom fills/copies or erase length outputs.')),
        'H02':dict(review_v43=dict(common,relation='Layer-plane-only global metadata resolution is located; this does not measure EngineCore/producers/transport CPU cost or prove reusable metadata generations.')),
    }
    modules={'M21':dict(
        finding='S13 finite63guard dataflows now27prior+36new,0pending crosswalk.12small methods+3line tail andselected model-bank fields supplemented only where prior profile/source could not decide. Normal same-width INT32 TopK formatting/direct output/no-op reset are baseline;2conditional refinements14constraints, no measured gain.',
        evidence=evidence,candidates=['F06','F12','F31','F09','F08','F17','F33'],
        gap='Original whole8gates remain partial: exact S01/S14 helper selectors; actual main/nested graph/QLI/parameter-bank/all-reader/phase/generation; S16resource and S18process/transport. Unimplemented conditional variants alone are not missing analysis.',
        scope_audit=str(O/'M21-original-requirement-completion-audit-v43.json'),
        profile_scope=str(O/'M21-original-profile-gap-reconciliation-v41.json'))}
    remaining=deepcopy(read(O/'module-coverage-gaps-v42.json')['remaining'])
    retired=[s for s in remaining if s.startswith(('M21v42','NextM21'))]
    write(O/'M21-remaining-history-before-v43.json',dict(previous_version=42,retired_progress_paragraphs=retired,
        reason='S13 remaining36 exact guard-dataflow frontier is now resolved; retain authentic graph/dispatch/resource/process gaps without repeating closed local questions.',overall_complete=False))
    remaining=[s for s in remaining if s not in retired]
    remaining[16:16]=[
        'M21v43:S13 all63finite exact static guard dataflows reconciled(27prior+36new,0remaining crosswalk);18semantic families/268navigation sites retained. Existing direct indices_output/lengths_output, no-op reset, same-width INT32 contiguous TopK→SMLA views and branch-specific cache stream publication are baseline, not new device savings.2conditional F06/F12 refinements14constraints; quantizer/helper alias not locally frozen and no actual savings counted.',
        'NextM21:M22 actual main/nested graph/QLI body/parameter banks/inputready/first-lastreaders/reuse and16phase-unbound graphs; S16 same-denominator critical resource bounds andS18 full EngineCore/transport withM23-M25/global26module audit. Current saved-graph receipt holds summary, notfull80graphbodies; old16911graph firstsamples cannot fill current166 bank gaps. Full pinned v14 result stays on166; auth failed v42rc255, credential question pending, no login retry. Exact S01/S14 enable_pcp/decode-shard/normalized/packed predicates remain. Other local finite module audits may proceed; do not redo63guards or require implementation to record conditional opportunity.',
    ]
    action=(
        '继续Loop004，v43为实质进展：S13剩余36具体guard已逐项dataflow对照，累计63全部本地语义case已分析，仍无整模块关闭。先读M21-neighbor-reconciliation-summary-v43.json、M21-adjacent-guard-crosswalk-v43.json、M21-neighbor-guard-and-index-publication-contract-v43.json、M21-original-requirement-completion-audit-v43.json及M22-saved-profile-local-payload-availability-v43.json。'
        '保持profile优先，仅缺结论时定点查代码，禁整库/SDK审计和模型/torch/NPU/service/health工作；本轮0remote，无后台job，lasthealth仍v12post。4runs3PASS1ERROR：source小方法/数学guard/正式audit通过，CVLinear.quantize本地缺freeze/zip初次FileNotFound保留，未有wrapper结果也未重试166。前v42SSHrc255凭据问题仍pending，不能拿失败当谓词证据。'
        '共有18稳定M21families/268static导航sites，S13的63guard=27prior+36new，0待逐guard对照；不要再次把36/63当unknown。Exact CP _global_layer_metadata只选本layer四cacheplane、不遍历混有draftnative的全map；base/CPindexerHidden用本地rows，CPnon-indexsource按qr.shape0复用sharedtopk。'
        'prepareindexsource role、query+weights不同输入、RoPE早于compression scratch、quant5-termguard及quantize_done等待已审；C1BF16/C2FP32+pool+COMPRESSORwait/sourcecos/sin任一None时fallback保留；indexoldlatent读先于longinplaceRoPE复用v27已审。'
        '多流sharequant为metadata identity+量化方法type+commflag，非通用同shape等价；当前draftquery2/3与globalKV10/20不同，不能整份共用。部分行quantreuse仅对同参数相同行函数含scale且有非trivialquant路径；positiveQA/KV原语F.linear，不能报新quantkernel收益。5toyrow例与batch-scale反例仅数学条件。'
        'Index select直接传indices_output和lengths_output已实现；emptycache填-1并清长度，candidate为blockID非token。Normal sameconfig topkbank [maxTokens,config.index_topk] I32，topology.index_topk=int(config.index_topk)，K同宽无padding，前导row slice contiguous，unsqueeze/contiguous/int可只是views/identity；不虚算3kernels。K>topk通用分支不截断。36toyformatcase只是合同。SharedState.reset为no-op，source-before-consumer每forward覆盖要求不能替换成跨迭代pointer CSE。'
        'F12 conditional stablebank self-copy elision仍需实际preparehelper return alias/fullwrite/version/capture observer证明；该helper本地snapshot/zip缺失，未读取不宣称redundantcopy或实际省时。packedprefill/mixed maincachewrite在auxjoin后，其余auxwrite后join，保留恰好一个writer及既有overlap。'
        '下一步推进真正未证范围：M22实际80graph/16unboundphase/QLIbody/parambank/allreader/ready/reuse，S16同分母关键bounds，S18/M03/M25全进程transport；再M23-M25/global26module。Savedgraphv14本地receipt只有summary/MC2regimes，不含full80body；fullresultSHAa035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991在166。old16911graphsample不顶替current166。认证未更新时可准备有限CPU读脚本并继续其他已有资料模块原scope审计，不自动重试或设blocked。'
        'S01/S14 exactdispatch/PCPdecoder/packedhelper仍缺但PCP接口反事实已审，勿整段重复。26理论INCOMPLETE/3136资源/849PMU缺/16未绑定图、35模型7框架0module不变。8原mandatorygates仍partial/incomplete。只因真实覆盖证据缺口Goalactive，非因条件变体未实现。用persist_profile_refinement_v3且别改pinned旧helper。'
    )
    report='''本轮用已有 profile 和直接源码证据，对 S13 剩余36个具体静态 guard 逐项核对。累计63个 guard 的本地数据流、分支域和邻接读写协议均有记录；这一局部范围不再作为未分析项。未把它扩大成全部运行时字段、实际 alias、capture bank 或整模块已证明。

复用 v16/v27 的 stream/C1/C2/cache 证据、v35 的实际 CP 输入输出及空 query 调用栈、v40 的 QLI reader。新补12个小方法、3行多流尾部和有限 bank/topology 字段，未重新读整份 DSA、整个模型或 SDK。可核实的关键合同包括：global metadata 只解析本 layer 的4个 cache planes；CP 复用索引按本地 qr 行；index query 与 weights 输入不同，量化需要5项条件且 consumer 等实际事件；C2 元数据等待、source RoPE fallback、旧 latent 读先于 long 原地 RoPE；packed mixed/prefill cache write 在 main join 后，其余选中路径在 aux 写后 join。

还排除了几项容易虚算的“优化”：

- select_projected 已给 prepare_indexer_indices 传 indices_output、lengths_output；直接发布是已有行为，不能当成尚未实现的收益。
- 普通同配置初始化将 shared TopK bank 分配为 INT32 [max_tokens,config.index_topk]，topology 使用同一 index_topk。若没有后续替换，同宽前导行切片已连续，pad 不走，unsqueeze 是 view，contiguous/int 可以保持 storage。不能把源码三步当成必发的三个 kernel；通用 K>topk 分支也不截断。
- shared state reset 本身返回 None，不清零。跨层复用依赖 source layer 当轮覆盖有效行，不支持跨迭代仅按地址缓存旧值。
- 空 cache 路径处理 indices=-1、topk lengths=0 和 candidate-source lengths=0；长度与候选 block ID 都是独立输出，不能只保留 token IDs。

保留2个条件细化、14项约束：F06 的部分行 quant reuse，仅适用于相同行量化函数/scale/参数/通信和非 trivial quant 路径，且必须保留局部 Q 就绪与当前 overlap；F12 的 stable-bank copy 省略，须证明实际 producer 返回同一 range、已完整写入且没有 version/debug/capture observer。当前正例 Q_A/KV 是 unquantized F.linear，没有计入任何实际量化收益；精确 CVLinear.quantize 与 prepare_indexer_indices helper 不在本地冻结资料，未假装已读。5个行函数例与一个 batch-scale 反例、36个格式例只证明数学/布局条件，不证明 native 性能或精度。

一次补查 CVLinear.quantize 由于本地缺文件失败，原错误保留；随后只核对 archive member 也不存在，没有重试堡垒机。登记4项：定点 source、条件/guard 分析和正式审计3 pass，缺冻结文件1 error。本轮没有远端作业、模型/torch/NPU/服务/健康操作。

整体仍未完成。M21 保持18个稳定语义范围和268个导航 site，原8项 mandatory gate 仍 partial/incomplete；真实剩余为精确 dispatch/PCP helper、实际 main/nested graph/QLI body 与 parameter banks、所有读者和跨代复用、同分母资源上界、完整 EngineCore/transport，以及其余模块全范围审计。本地 saved-graph receipt 只有当前80图摘要，full result 仍在166；旧169的11图样本不替代当前 bank 证据。前轮166 SSH鉴权问题尚未恢复，保留缺口但继续可用本地分析。

35模型候选、7框架候选、0整模块关闭，26理论比较 INCOMPLETE、3136资源行、849缺PMU、16 phase-unbound 图均保持。Goal active；条件优化未实现本身不被当成未分析缺口。
'''
    jobs=[
        dict(kind='review',status='pass',description='Bounded12small guard/row/reset methods+3line tail and selected bank/topology fields;4prior embedded source receipt hashes verified, no remote query.',artifacts=[N/'read_M21_remaining_guard_points_v43.py']+[O/x for x in evidence[:4]]),
        dict(kind='review',status='error',description='Exact CVLinear.quantize supplement failed local frozen-file read; archive member absent, no emitted wrapper source and no remote retry.',artifacts=[O/'M21-quantizer-local-freeze-gap-v43.json']),
        dict(kind='review',status='pass',description='36new+27prior exactguard dataflows reconciled; normal TopK format/directout/reset baselines identified;5row/36format math cases plus scale counterexample;2conditionalvariants14constraints.',artifacts=[N/'build_M21_neighbor_reconciliation_v43.py']+[O/x for x in evidence[4:9]]),
        dict(kind='review',status='pass',description='Original18family268site8gate audit retained;0remaining S13crosswalk is not wholemodule closure; current80graph receipt vsold169sample availability checked; previous73file manifest verified.',artifacts=[O/x for x in evidence[9:]]+[N/'persist_profile_refinement_v3.py']),
    ]
    summary=dict(filename='M21-neighbor-reconciliation-summary-v43.json',previous_turn_classification='PROGRESS:v42fixedcase',
        original_static_sites=268,semantic_branch_families=18,original_mandatory_gates=8,completion_audit='NOT_COMPLETE',
        neighbor_S13_sites=63,prior_reconciled_neighbor_guards=27,new_reconciled_neighbor_guards=36,total_reconciled_neighbor_guards=63,remaining_neighbor_guard_crosswalk=0,
        conditional_variants=2,condition_rows=14,row_math_examples=5,index_format_cases=36,
        refined_model_candidates=['F06','F12'],related_model_candidates=['F31','F09','F08','F17','F33'],new_model_candidates=[],
        remote_jobs=0,remote_pass=0,remote_retained_errors=0,retained_local_errors=1,
        authentication_question_pending=True,
        remote_job_status='No remote jobs/current server handles. All local v43 jobs terminal:3reviewPASS/1missing-freezeERROR retained. Previous166authenticationfailure remains, no retry. No model/torch/NPU/service/health work.',
        no_new_model_NPU_service_or_health_work=True)
    persist_refinement(version=43,previous=42,title='v43 M21 adjacent guard and index publication reconciliation',summary=summary,
        changes=['S13 original63 exact guard dataflows reconciled:27prior+36new,0remaining localcrosswalk;18semanticfamilies/268navigation sites preserved.',
            'Existing direct index postprocess destination, zero-cache lengths, no-op reset and normal same-width INT32 format views identified as baseline;no hypothetical kernel savings counted.',
            '2conditionalF06/F12 refinements14constraints;5row and36format math cases only. Actual graph/bank/allreader/dispatch/resource/process ranges remain open.'],
        candidate_updates=candidates,framework_updates=fw,module_updates=modules,semantic_branches={'M21':sem['branches']},source_read=True,
        remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=['M21-adjacent-guard-crosswalk-v43.json','M21-neighbor-guard-and-index-publication-contract-v43.json','M21-original-requirement-completion-audit-v43.json','M22-saved-profile-local-payload-availability-v43.json'],
        publication_rows={
            'F06':'| F06 | Q与KV投影共享输入或组合 | 条件 | v43核对metadata身份/量化方法/通信共享条件；CP局部Q与全局KV不能按同形整份复用。部分行量化重用需相同行函数、参数、scale、row map及就绪；当前unquantized原语无新量化kernel收益计数。 |',
            'F12':'| F12 | Indexer TopK、索引修整与共享发布 | 条件 | v43补role/候选/长度/emptycache/本地row合同；direct indices_output已存在。普通同宽I32连续bank的SMLA格式可只是view，不虚算pad/cast工作。copy省略需实际return alias/fullwrite/version/capture证明；block候选与token索引分开。 |',
            'F31':'| F31 | Metadata、slot、draft共享与输入输出发布重组 | 条件 | v43将S13累计63个具体guard数据流全部对照，0待逐guard项；shared reset为no-op且需当轮source先覆盖。18语义范围/268导航sites/8原gate保留，精确dispatch、main graph/QLI参数bank/所有读者、资源与全进程边界仍未闭合。 |',
        },
        gate_updates=dict(status='FINITE_63_NEIGHBOR_GUARD_DATAFLOWS_RECONCILED;WHOLE_SCOPE_OPEN',answered_by=evidence,
            closed_scope='All63 finite adjacent guard local cases/dataflows/protocols; normal index-bank/directout/reset baseline clarification.',
            not_closed='Exact external quantizer/helper alias, S01/S14 selectors, main/nested graph/QLI parameter bank/allreader/phase/resource/process/global26 scope.',
            stop_rule='Do not repeat63localguard audit or count source formatting/directout as missing kernels. Advance actual full saved graph and other original module coverage; conditional implementation not required.'),
        scripts=[N/'read_M21_remaining_guard_points_v43.py',N/'build_M21_neighbor_reconciliation_v43.py'],
        executor_prefix='zcode-M21-',main_script=N/'finalize_M21_neighbor_reconciliation_v43.py')


if __name__=='__main__':
    main()
