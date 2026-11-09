"""Validate the repaired backend controls before publishing or applying a job.

The frontend and workers share this contract. Planning has no environment,
model or cache side effects; all validations complete before activation.
"""

def mode_environment(mode, *, shadow_enabled=False):
    environment = {}
    required = {'varlen', 'tail_ffn', 'engram_prefetch', 'engram_pinned', 'engram_uva', 'engram_device_hash', 'yoco_exact_compact', 'reset_prefix_cache', 'yoco_meta_fuse', 'zos_fast'}
    optional = {'cache_restore_event', 'prefill_device_hash', 'prefill_device_hash_check', 'prefill_engram_uva', 'prefill_engram_uva_check', 'prefill_kv_overlap', 'prefill_timing_detail', 'prefill_timing_split', 'recipe_tokenizer', 'yoco_bounded', 'yoco_bounded_140', 'yoco_merged_qb', 'yoco_official_mask', 'yoco_official_prefill', 'yoco_preplan', 'zos_c2_check', 'zos_c2_device', 'zos_engram_window', 'zos_engram_window_check', 'zos_target_device', 'zos_vote_reuse', 'zos_official_scheduler'}
    optional.update({'zos_worker_sync_ablation', 'mtp_ablation_fixed5', 'zos_async_sample_inputs', 'zos_continuous_admission', 'yoco_dp_local', 'zos_continuous_eos', 'zos_continuous_small', 'zos_continuous_cold', 'zos_official_multigraph'})
    if not isinstance(mode, dict) or not required.issubset(mode) or set(mode) - required - optional or any((type(value) is not bool for value in mode.values())):
        raise ValueError('explicit boolean single-point mode required')
    if mode.get('zos_async_sample_inputs', False) and (not mode.get('zos_official_scheduler', False)):
        raise ValueError('async sample inputs require the original Scheduler route')
    if mode.get('zos_continuous_admission', False) and (not mode.get('zos_official_scheduler', False)):
        raise ValueError('continuous admission requires original Scheduler')
    if mode.get('zos_continuous_eos', False) and (not mode.get('zos_continuous_admission', False)):
        raise ValueError('continuous EOS requires continuous admission')
    if mode.get('zos_continuous_small', False) and (not mode.get('zos_continuous_admission', False)):
        raise ValueError('continuous small batches require continuous admission')
    if mode.get('zos_official_multigraph', False) and (not mode.get('zos_official_scheduler', False)):
        raise ValueError('multigraph requires the original Scheduler')
    if mode.get('zos_continuous_cold', False) and (not mode.get('zos_continuous_admission', False)):
        raise ValueError('continuous cold requests require continuous admission')
    from .official_yoco_metadata910c import official_mode
    official_environment = official_mode(mode)
    environment['DEEPINFER_910C_SCHEDULED_ROUTE'] = '1' if mode.get('zos_official_scheduler', False) else '0'
    environment['DEEPINFER_910C_ZOS_ASYNC_SAMPLE_INPUTS'] = '1' if mode.get('zos_async_sample_inputs', False) else '0'
    environment['DEEPINFER_910C_ZOS_CONTINUOUS_ADMISSION'] = '1' if mode.get('zos_continuous_admission', False) else '0'
    environment['DEEPINFER_910C_ZOS_CONTINUOUS_EOS'] = '1' if mode.get('zos_continuous_eos', False) else '0'
    environment['DEEPINFER_910C_ZOS_CONTINUOUS_SMALL'] = '1' if mode.get('zos_continuous_small', False) else '0'
    environment['DEEPINFER_910C_ZOS_CONTINUOUS_COLD'] = '1' if mode.get('zos_continuous_cold', False) else '0'
    environment['DEEPINFER_910C_ZOS_OFFICIAL_MULTIGRAPH'] = '1' if mode.get('zos_official_multigraph', False) else '0'
    if mode.get('zos_worker_sync_ablation', False) and (not mode.get('zos_official_scheduler', False) or not mode.get('zos_continuous_admission', False)):
        raise ValueError('sync Worker ablation requires continuous original Scheduler route')
    if mode.get('mtp_ablation_fixed5', False) and (not mode['varlen'] or mode.get('zos_official_scheduler', False)):
        raise ValueError('fixed5 ablation requires native varlen graph route')
    environment['DEEPINFER_910C_MTP_ABLATION_FIXED5'] = '1' if mode.get('mtp_ablation_fixed5', False) else '0'
    environment['DEEPINFER_910C_ZOS_WORKER_SYNC_ABLATION'] = '1' if mode.get('zos_worker_sync_ablation', False) else '0'
    environment['DEEPINFER_ENABLE_MTP_VARLEN'] = '1' if mode['varlen'] else '0'
    environment['DEEPINFER_910C_ENGRAM_PREFETCH'] = '1' if mode['engram_prefetch'] else '0'
    environment['DEEPINFER_910C_ENGRAM_PINNED_STAGING'] = '1' if mode['engram_pinned'] else '0'
    environment['DEEPINFER_910C_ENGRAM_UVA'] = '1' if mode['engram_uva'] else '0'
    environment['DEEPINFER_910C_ENGRAM_DEVICE_HASH'] = '1' if mode['engram_device_hash'] else '0'
    environment['DEEPINFER_910C_YOCO_EXACT_COMPACT'] = '1' if mode['yoco_exact_compact'] else '0'
    if mode['yoco_meta_fuse'] and (not mode['yoco_exact_compact']):
        raise ValueError('metadata fusion requires exact compaction')
    environment['DEEPINFER_910C_YOCO_META_FUSE'] = '1' if mode['yoco_meta_fuse'] else '0'
    if mode['zos_fast'] and shadow_enabled:
        raise ValueError('fast ZOS and shadow diagnostics cannot share a job')
    environment['DEEPINFER_910C_ZOS_FAST'] = '1' if mode['zos_fast'] else '0'
    environment['DEEPINFER_910C_ZOS_VOTE_REUSE'] = '1' if mode.get('zos_vote_reuse', False) else '0'
    if mode.get('zos_engram_window_check', False) and (not mode.get('zos_engram_window', False)):
        raise ValueError('device Engram window check requires device window')
    if mode.get('zos_engram_window', False) and (not mode['zos_fast'] or not mode['engram_uva'] or (not mode['engram_device_hash'])):
        raise ValueError('device Engram window requires fast ZOS and device hash/UVA')
    if mode.get('zos_c2_check', False) and (not mode.get('zos_c2_device', False)):
        raise ValueError('device C2 check requires device C2')
    if mode.get('zos_c2_device', False) and (not mode['zos_fast']):
        raise ValueError('device C2 requires fast ZOS')
    if mode.get('zos_target_device', False) and (not mode['zos_fast']):
        raise ValueError('device target input requires fast ZOS')
    environment['DEEPINFER_910C_ZOS_TARGET_DEVICE'] = '1' if mode.get('zos_target_device', False) else '0'
    environment['DEEPINFER_910C_ZOS_C2_DEVICE'] = '1' if mode.get('zos_c2_device', False) else '0'
    environment['DEEPINFER_910C_ZOS_C2_CHECK'] = '1' if mode.get('zos_c2_check', False) else '0'
    environment['DEEPINFER_910C_ZOS_ENGRAM_WINDOW'] = '1' if mode.get('zos_engram_window', False) else '0'
    environment['DEEPINFER_910C_ZOS_ENGRAM_WINDOW_CHECK'] = '1' if mode.get('zos_engram_window_check', False) else '0'
    if mode.get('yoco_bounded', False) and (not mode['tail_ffn'] or mode['yoco_exact_compact'] or mode['yoco_meta_fuse']):
        raise ValueError('bounded replay requires tail FFN and an exclusive replay path')
    if mode.get('yoco_bounded_140', False) and (not mode.get('yoco_bounded', False)):
        raise ValueError('140-token replay requires bounded replay')
    if mode.get('yoco_preplan', False) and (not mode.get('yoco_bounded', False)):
        raise ValueError('YOCO preplan requires bounded replay')
    if mode.get('prefill_timing_detail', False) and (not mode.get('prefill_timing_split', False)):
        raise ValueError('prefill timing detail requires split timing')
    if mode.get('prefill_engram_uva', False) and (not mode['engram_uva'] or mode['engram_prefetch']):
        raise ValueError('prefill UVA requires UVA and excludes CPU row prefetch')
    if mode.get('prefill_engram_uva_check', False) and (not mode.get('prefill_engram_uva', False)):
        raise ValueError('prefill UVA check requires prefill UVA')
    if mode.get('prefill_device_hash', False) and (not mode.get('prefill_engram_uva', False) or not mode['engram_device_hash']):
        raise ValueError('prefill device hash requires prefill UVA and device hash')
    if mode.get('prefill_device_hash_check', False) and (not mode.get('prefill_device_hash', False)):
        raise ValueError('prefill device hash check requires device hash')
    environment['DEEPINFER_910C_YOCO_BOUNDED'] = '1' if mode.get('yoco_bounded', False) else '0'
    environment['DEEPINFER_910C_YOCO_BOUNDED_140'] = '1' if mode.get('yoco_bounded_140', False) else '0'
    environment['DEEPINFER_910C_PREFILL_TIMING_SPLIT'] = '1' if mode.get('prefill_timing_split', False) else '0'
    environment['DEEPINFER_910C_PREFILL_TIMING_DETAIL'] = '1' if mode.get('prefill_timing_detail', False) else '0'
    environment['DEEPINFER_910C_PREFILL_KV_OVERLAP'] = '1' if mode.get('prefill_kv_overlap', False) else '0'
    environment['DEEPINFER_910C_YOCO_PREPLAN'] = '1' if mode.get('yoco_preplan', False) else '0'
    environment['DEEPINFER_910C_PREFILL_ENGRAM_UVA'] = '1' if mode.get('prefill_engram_uva', False) else '0'
    environment['DEEPINFER_910C_PREFILL_ENGRAM_UVA_CHECK'] = '1' if mode.get('prefill_engram_uva_check', False) else '0'
    environment['DEEPINFER_910C_PREFILL_DEVICE_HASH'] = '1' if mode.get('prefill_device_hash', False) else '0'
    environment['DEEPINFER_910C_PREFILL_DEVICE_HASH_CHECK'] = '1' if mode.get('prefill_device_hash_check', False) else '0'
    environment.update(official_environment)
    return environment
