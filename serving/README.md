# Input token cache integration

Opt-in fixed-tokenizer text encoding cache. Run672 supports KEEP candidate; integrated reverse-order formal confirmation is Run673. No model outputs are cached.

`input_token_cache.py` bounds immutable token IDs to 128 entries and an estimated 128 MiB of retained objects. Full rendered text and typed encoding options form the key. Every request gets a fresh prompt/token list and retains normal validation. Offset/multimodal paths bypass caching. Tokenizer identity changes reset the cache; in-place tokenizer reconfiguration is unsupported and requires service restart. Warm repeated-input performance does not imply cold or unique-input acceleration.

Install only with service stopped using `scripts/install_input_token_cache.py install --state-dir <new-directory> --record <record.json> --offline-confirmed`; installer pins exact original BaseRenderer SHA. Enable with `EXTREME_TEXT_TOKEN_CACHE=1` and `EXTRA_SERVE_ARGS="--renderer-num-workers 4"` when invoking `scripts/serve.sh`. The default launch is unchanged. Stop service before the matching restore command. Prefer the guarded Run673 controller for validation: it owns lock, source pins, service lifecycle, correctness and restoration.
