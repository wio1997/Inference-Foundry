"""Real-worker eager MTP preflight. No substitute pool/config/model is built."""
import hashlib
import inspect
import json
import os
from pathlib import Path


def _require(ok, message):
    if not ok:
        raise RuntimeError('GLM53 eager MTP preflight: ' + message)


def _leaves(backend):
    if backend is None:
        return []
    if hasattr(backend, 'attn_backends'):
        return [leaf for b in backend.attn_backends for leaf in _leaves(b)]
    return [getattr(backend, 'full_attn_backend', backend)]


def _ledger(model, role):
    path = getattr(model, '_glm53_weight_ledger_path', None)
    _require(path is not None, role + ' actual model has no load ledger')
    report = json.loads(Path(path).read_text())
    _require(report['role'] == role, role + ' ledger role mismatch')
    _require(report['schema'] == 'glm53_weight_ledger_g11', role + ' G11 ledger required')
    _require(not report['exception'] and not report['missing_parameters']
             and not report['unexpected_undispatched'], role + ' strict load coverage failed')
    _require(not report['deferred_target_shared_parameters'], role + ' target shares unresolved')
    if role == 'draft':
        from glm53_sglang_adapter.checkpoint_bindings import require_checkpoint_owned_bindings
        require_checkpoint_owned_bindings(model, report)
        _require(getattr(model, '_glm53_checkpoint_contract', {}).get('own_embedding_head_unquantized') is True, 'draft checkpoint metadata/unquantized contract missing')
        _require(getattr(model, '_glm53_rotation_configured', False), 'draft rotation not configured')
        rotation = getattr(model, '_glm53_rotation_evidence', {})
        _require(not rotation.get('required') or rotation.get('loaded'), 'required rotation not loaded')
    return {'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'status': report['status'], 'scope': 'dispatch coverage; tensor values/per-expert writes unverified'}


def preflight_worker(worker):
    """Called once after native target and every draft backend exist, before graphs."""
    from sglang.srt.runtime_context import get_parallel, get_spec
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, check_cuda_graph_backend
    from glm53_sglang_adapter import nope_consumer_runtime as helper
    from glm53_sglang_adapter.runtime import bind_mtp_eager_contract

    _require(os.environ.get('GLM53_NOPE_EAGER_MTP_TOPK1_W4') == '1', 'explicit eager opt-in required')
    _require(os.environ.get('GLM53_LOAD_LEDGER_STRICT', '1') == '1', 'strict load ledger required')
    _require(type(worker).__name__ == 'EagleDraftWorker' and worker.speculative_algorithm.is_eagle(),
             'only actual native EAGLE/NEXTN draft worker is covered')
    _require(not bool(getattr(worker.server_args, 'enable_multi_layer_eagle', False)), 'multi-layer EAGLE unsupported')
    from .mtp_graph_bucket import enabled
    ordered_candidate=enabled()
    if not ordered_candidate:_require(bool(worker.server_args.disable_overlap_schedule), 'serialized scheduler required')
    _require(not bool(getattr(worker.server_args, 'enable_two_batch_overlap', False)), 'TBO unsupported')
    if ordered_candidate:
        _require(check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED), 'Captured prefill is outside W4 candidate')
        from sglang.srt.managers.scheduler import Scheduler
        _require('GLM53_MTP_GRAPH_ASYNC_CANDIDATE' in inspect.getsource(Scheduler._apply_war_barrier), 'Native async whole-forward WAR fence missing')
    else:
        _require(all(check_cuda_graph_backend(phase, Backend.DISABLED)
                     for phase in (Phase.DECODE, Phase.PREFILL)), 'disable both decode and prefill graphs')
    p, s = get_parallel(), get_spec()
    _require(p.attn_cp_size == p.attn_dcp_size == p.attn_dp_size == 1, 'CP1/DCP1/attention-DP1 required')
    _require(s.speculative_eagle_topk == 1 and s.speculative_num_steps == 3
             and s.speculative_num_draft_tokens == 4, 'topk1/steps3/W4 required')
    target, draft = worker.target_worker.model_runner, worker.draft_runner
    _require(not target.is_draft_worker and draft.is_draft_worker, 'actual runner roles disagree')
    result = {'target_ledger': _ledger(target.model, 'target'), 'draft_ledger': _ledger(draft.model, 'draft')}
    import torch.distributed as dist
    result['rank'] = dist.get_rank() if dist.is_initialized() else None
    manifest = Path(__file__).resolve().parents[1] / 'recovery_mtp_manifest.json'
    _require(manifest.is_file(), 'installed runtime receipt missing')
    result['runtime_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if ordered_candidate:
        graph_manifest=Path(__file__).resolve().parents[1]/'recovery_graph_async_manifest.json'
        _require(graph_manifest.is_file(), 'Graph/async source-bound install receipt missing')
        result['graph_async_manifest_sha256']=hashlib.sha256(graph_manifest.read_bytes()).hexdigest()
    result['resolved_config'] = {'topk': 1, 'steps': 3, 'verify_width': 4,
                                 'disable_overlap_schedule': bool(worker.server_args.disable_overlap_schedule),
                                 'graph': not check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED),
                                 'ordered_candidate': ordered_candidate,
                                 'CP': p.attn_cp_size, 'DCP': p.attn_dcp_size, 'attention_DP': p.attn_dp_size}
    target_backend = target.attn_backend
    _require(callable(getattr(target_backend, 'update_mamba_state_after_mtp_verify', None)),
             'target native KDA accepted-state hook missing')
    _require(type(getattr(target_backend, 'linear_attn_backend', None)).__name__ == 'AscendKDAAttnBackend',
             'target must use native Ascend KDA, not generic GDN/Mamba')
    result['native_kda_commit_source'] = inspect.getsourcefile(target_backend.update_mamba_state_after_mtp_verify)
    backends = [('target', b) for b in _leaves(target_backend)]
    for root in (draft.attn_backend, worker.draft_extend_attn_backend, worker.draft_attn_backend):
        backends += [('draft', b) for b in _leaves(root)]
    seen, records = set(), []
    for role, backend in backends:
        if id(backend) in seen:
            continue
        seen.add(id(backend))
        _require(bool(backend.is_draft_worker) == (role == 'draft'), 'actual backend role mismatch')
        pool = getattr(backend.token_to_kv_pool, 'full_kv_pool', backend.token_to_kv_pool)
        helper._pool_contract(backend, pool)
        _require(getattr(backend, 'glm53_nope_mtp_optin', False), 'actual backend NoPE initializer missing')
        _require(getattr(backend, 'glm53_nope_sparse_consumer', None) is not None, 'NoPE consumer missing')
        _require(not backend.graph_mode, 'graph backend cannot enter eager candidate')
        records.append({'role': role, 'class': type(backend).__module__ + '.' + type(backend).__name__,
                        'pool_class': type(pool).__module__ + '.' + type(pool).__name__,
                        'native_draft_flag': pool.is_draft_worker,
                        'DCP1_role_metadata_warning': bool(pool.is_draft_worker) != (role == 'draft'),
                        'model_layer_ids': list(pool.model_layer_ids), 'indexer_layer_ids': list(pool.indexer_layer_ids),
                        'tail_capacity': [int(t.shape[1]) for t in pool._compress_tail_k]})
    _require(any(r['role'] == 'target' for r in records) and any(r['role'] == 'draft' for r in records),
             'both actual pool roles required')
    if ordered_candidate:
        from .runtime import bind_mtp_ordered_contract
        bind_mtp_ordered_contract(topk=s.speculative_eagle_topk,verify_width=s.speculative_num_draft_tokens,attn_dp_size=p.attn_dp_size)
    else:
        bind_mtp_eager_contract(topk=s.speculative_eagle_topk, verify_width=s.speculative_num_draft_tokens,
                                disable_overlap_schedule=worker.server_args.disable_overlap_schedule,
                                attn_dp_size=p.attn_dp_size)
    result.update(backends=records, bound_eager_contract=True, full_model_forward_verified=False,
                  graph_async_PD_EP8_performance_verified=False)
    root = Path(os.environ['GLM53_LOAD_LEDGER_DIR'])
    path = root / ('mtp_preflight.pid%d.json' % os.getpid())
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(temp, path)
    worker._glm53_mtp_preflight = result
    return result
