# SPDX-License-Identifier: Apache-2.0
"""Apply checkpoint EOS IDs to an empty native scheduler before requests."""


def configure_scheduler_eos(scheduler, generation_config, *, tokenizer_eos_id, vocab_size):
    """Configure all checkpoint EOS IDs; existing scheduler EOS is retained.

    This only configures the explicit scheduler instance. The caller must supply
    the checkpoint's verified generation configuration and tokenizer EOS, before
    adding requests; per-request ignore_eos remains the scheduler's decision.
    """
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError('positive vocabulary size required')
    raw=generation_config.get('eos_token_id')
    ids=[raw] if type(raw) is int else raw
    if not isinstance(ids,list) or not ids or any(type(v) is not int or not 0<=v<vocab_size for v in ids):
        raise ValueError('generation configuration requires valid EOS IDs')
    if type(tokenizer_eos_id) is not int or tokenizer_eos_id not in ids:
        raise ValueError('tokenizer EOS must be in the checkpoint EOS list')
    additional=sorted(set(ids)-{tokenizer_eos_id})
    scheduler.set_additional_eos_token_ids(additional)
    return tuple(additional)
