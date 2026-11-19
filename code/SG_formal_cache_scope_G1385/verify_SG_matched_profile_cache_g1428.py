"""Validate the G1317 formal cache scope before an independent matched profile."""

from math import isfinite

from scope_SG_formal_prefix_evidence_g1385 import COUNTER_FIELDS, PREFILL_COUNTER


DECODE_UNOBSERVED = {
    'verified': False,
    'reason': 'No reviewed prefix-hit counter supplied',
}


def validate_formal_cache_scope(profile, formal_gate):
    """Require measured Prefill counts and preserve Decode as unobserved."""
    assert formal_gate['stage'] == 'formal' and formal_gate['completed_warmups'] == 5
    assert set(formal_gate['roles']) == {'prefill', 'decode'}
    assert not formal_gate['roles']['decode'].keys() & set(COUNTER_FIELDS)
    original_decode = formal_gate['D_prefill_counter_observation']
    assert original_decode['prefix_hit_counter'] == PREFILL_COUNTER
    assert original_decode['prefix_hit_counter_labels']['engine_type'] == 'decode'

    prefill = formal_gate['roles']['prefill']
    labels = prefill['prefix_hit_counter_labels']
    assert prefill['prefix_hit_counter'] == PREFILL_COUNTER
    assert labels['engine_type'] == 'prefill' and labels['mode'] == 'device_hit'
    assert labels['model_name'] == formal_gate['model_id'] and labels['tp_rank'] == '0'
    assert profile['cold_cache_counter_verified'] is False
    phases = profile['cache_counter_evidence']
    assert set(phases) == {'cold', 'hot'}
    for phase, evidence in phases.items():
        assert set(evidence) == {'prefill', 'decode'}
        assert evidence['decode'] == DECODE_UNOBSERVED
        measured = evidence['prefill']
        assert measured['verified'] is True
        assert measured['counter'] == PREFILL_COUNTER and measured['labels'] == labels
        before, after, delta = (measured[key] for key in ('before', 'after', 'delta'))
        assert all(type(value) in (int, float) and isfinite(value) for value in (before, after, delta))
        assert 0 <= before <= after and delta == after - before
        if phase == 'cold':
            assert delta == 0, 'Prefill recorded a device prefix hit during the cold phase'

    return {
        'prefill': {
            'cold_device_hit_delta': phases['cold']['prefill']['delta'],
            'hot_device_hit_delta': phases['hot']['prefill']['delta'],
            'counter': PREFILL_COUNTER,
            'labels': dict(labels),
        },
        'decode': dict(DECODE_UNOBSERVED),
    }
