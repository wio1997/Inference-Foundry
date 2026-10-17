"""Strict native per-request tier attribution guard; no service or SDK actions."""
def storage_proof(prefix_raw_tokens, page_size, p_sglext, d_sglext):
    assert type(prefix_raw_tokens) is int and prefix_raw_tokens > 1
    assert type(page_size) is int and page_size > 0
    # Exact B103 MTP bigram keys have one fewer logical position than raw IDs.
    expected = ((prefix_raw_tokens - 1) // page_size) * page_size
    result = {'expected_complete_pages': expected // page_size,
              'expected_storage_tokens': expected, 'native_tiers': {}, 'pass': True}
    for role, extension in [('P', p_sglext), ('D', d_sglext)]:
        details = extension.get('cached_tokens_details') if isinstance(extension, dict) else None
        valid = bool(isinstance(details, dict) and
                     all(type(details.get(k)) is int for k in ['device', 'host', 'storage']) and
                     details['device'] == 0 and details['host'] == 0 and details['storage'] == expected)
        result['native_tiers'][role] = {'details': details, 'pass': valid}
        result['pass'] &= valid
    return result
