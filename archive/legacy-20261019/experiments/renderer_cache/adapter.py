"""Opt-in test adapter. Does not monkeypatch installed vLLM source."""
from threading import Lock
from .token_cache import TextTokenCache
_init_lock = Lock()

def cached_tokenize_prompt(original, renderer, prompt, params):
    if (renderer._wants_offsets(prompt, params)
            or prompt.get('multi_modal_data') or prompt.get('multi_modal_uuids')):
        return original(renderer, prompt, params)
    tokenizer = renderer.get_tokenizer()
    with _init_lock:
        if getattr(renderer, '_foundry_cache_tokenizer', None) is not tokenizer:
            renderer._foundry_text_token_cache = TextTokenCache()
            renderer._foundry_cache_tokenizer = tokenizer
        cache = renderer._foundry_text_token_cache
    ids = cache.encode(prompt['prompt'], params.get_encode_kwargs(), tokenizer)
    return renderer._build_tokens_prompt(ids, prompt)
