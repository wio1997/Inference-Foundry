"""G46 opt-in, eager bs1 QLI-only query partition over attention TP4.

No process groups or runtime patches are created here. Framework EXTEND owns
consecutive positions; CPU final lengths come from existing metadata planning.
"""
import logging
import os
from qli_query_rows import plan_query_rows, build_local_qli_metadata, call_local_qli

ENABLED = os.environ.get('SGLANG_NPU_QLI_QUERY_ROW_SHARD', '0') == '1'
_LOGGED = set()


def eligible_query_partition(*, enabled, ordinary_prefill, graph_mode,
                             rows, batch_size, max_query_len, global_end,
                             attn_tp_size, attn_cp_size, cp_active, sp_active,
                             has_tbo, encoder_replay, contract_matches):
    """Pure predicate: every unsupported mode keeps the original QLI path."""
    return bool(enabled and ordinary_prefill and not graph_mode
                and rows == 8192 and batch_size == 1 and max_query_len == rows
                and global_end >= rows and attn_tp_size == 4 and rows % 4 == 0
                and attn_cp_size == 1 and not cp_active and not sp_active
                and not has_tbo and not encoder_replay and contract_matches)


class PartitionedQli:
    """Original QLI callable ABI around tested local operator + stock gather."""
    def __init__(self, bindings, plan, metadata, tp_rank):
        self.bindings, self.plan, self.metadata = bindings, plan, metadata
        self.tp_rank = tp_rank

    def __call__(self, q, key, weights, q_scale, key_scale, topk,
                 quant_mode, **kwargs):
        from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor

        plan = self.plan
        mode = kwargs['candidate_mode']
        assert quant_mode == 2 and topk == 512
        assert kwargs['cmp_ratio'] == plan.ratio
        assert kwargs['layout_q'] == 'TND' and kwargs['layout_k'] == 'PA_BBND'
        assert kwargs['mask_mode'] == 3
        assert kwargs['candidate_topk_blocks'] == 2048 and kwargs['candidate_block_size'] == 8
        # run_a3_qli has already performed FULL query quantization and FP16
        # weights cast. Restore only the scale view removed by its call ABI;
        # helper removes it again, passing the same tensor layout to the op.
        local, candidates = call_local_qli(
            self.bindings, plan, q, q_scale, weights, key,
            key_scale.unsqueeze(-1), kwargs['block_table'], self.metadata,
            mode=mode, candidates=kwargs.get('candidate_topk_index'),
            topk=topk, candidate_blocks=2048, candidate_block_size=8)
        assert local.shape[0] == plan.rows
        full = local.new_empty((plan.total_rows, *local.shape[1:]))
        attn_tp_all_gather_into_tensor(full, local.contiguous())
        published = None
        if mode == 1:
            assert candidates.shape[0] == plan.rows
            published = candidates.new_empty((plan.total_rows, *candidates.shape[1:]))
            attn_tp_all_gather_into_tensor(published, candidates.contiguous())
        key_log = (self.tp_rank, mode)
        if key_log not in _LOGGED:
            _LOGGED.add(key_log)
            logging.getLogger(__name__).warning(
                'G46_QLI_QUERY_ROWS_MODE tp_rank=%s mode=%s rows=%s local=%s full_finalize=True',
                self.tp_rank, mode, plan.total_rows, plan.rows)
        # Scores are intentionally unused by run_a3_qli; no score gather or
        # precision change is introduced. Candidate state is full-row again.
        return full, weights.new_empty((0,)), published


def prepare_query_parallel(bindings, cfg, fb, *, rows, graph_mode,
                           max_query_len, global_end):
    """Build independent per-ratio local metadata outside model forward."""
    if not ENABLED:
        return None
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.runtime_context import get_forward, get_parallel

    parallel = get_parallel()
    contract = (int(cfg.index_n_heads), int(cfg.index_head_dim),
                int(cfg.index_topk), int(cfg.candidate_topk_blocks),
                int(cfg.candidate_block_size)) == (32, 128, 512, 2048, 8)
    applicable = eligible_query_partition(
        enabled=ENABLED,
        ordinary_prefill=fb.forward_mode.is_extend_without_speculative(),
        graph_mode=bool(graph_mode), rows=rows, batch_size=int(fb.batch_size),
        max_query_len=max_query_len, global_end=global_end,
        attn_tp_size=parallel.attn_tp_size, attn_cp_size=parallel.attn_cp_size,
        cp_active=is_cp_active(fb), sp_active=get_forward().sp_active,
        has_tbo=bool(getattr(fb, 'tbo_children', None)),
        encoder_replay=bool(getattr(fb, 'encoder_swa_replay', False)),
        contract_matches=contract)
    if not applicable:
        return None
    # In ordinary non-CP/SP EXTEND bs1, positions are the consecutive suffix
    # ending at the CPU final length; no per-layer device scalar reads occur.
    wrappers = {}
    for ratio in (1, 2):
        plan = plan_query_rows(rows, global_end, parallel.attn_tp_rank, 4, ratio)
        metadata = build_local_qli_metadata(bindings, plan, fb.positions.device)
        wrappers[ratio] = PartitionedQli(bindings, plan, metadata, parallel.tp_rank)
    key_log = (parallel.tp_rank, 'enabled')
    if key_log not in _LOGGED:
        _LOGGED.add(key_log)
        logging.getLogger(__name__).warning(
            'G46_QLI_QUERY_ROWS_ENABLED tp_rank=%s attn_tp_rank=%s rows=%s group=attention_tp world=4 graph=False',
            parallel.tp_rank, parallel.attn_tp_rank, rows)
    return wrappers
