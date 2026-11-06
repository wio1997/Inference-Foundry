"""Add CPU plan sharing to the pinned, already accepted CP-length elision."""
import ast
from hashlib import sha256

ACCEPTED_SHA256 = 'e79e503d0426f2a28a405ad335b49a7c0a22d4ed5d293cfb0bc16ba2bca292ee'


def render(raw):
    assert sha256(raw).hexdigest() == ACCEPTED_SHA256
    source = raw.decode()
    import_line = 'import torch\n'
    assert source.count(import_line) == 1
    source = source.replace(import_line, import_line +
        'from wio_cp_cpu_plan import lookup as _wio_cpu_plan_lookup, store as _wio_cpu_plan_store\n')
    start = source.index('        start, end, per_rank, padded, qsl, seq_lens = AscendDSACPMetadataBuilder._build_local_token_metadata(')
    end = source.index('        query_start_loc = self._cp_query_start_loc[: qsl.numel()]', start)
    original_plan = source[start:end]
    assert 'qsl = qsl.clamp_max(actual_end - actual_start)' in original_plan
    replacement = '''        _wio_plan, _wio_scope = _wio_cpu_plan_lookup(self, common, seq_lens_cpu, kwargs)
        if _wio_plan is None:
'''
    replacement += ''.join('    '+line if line.strip() else line for line in original_plan.splitlines(keepends=True))
    replacement += '''        else:
            start, end, per_rank, padded, qsl, seq_lens, actual_start, actual_end = _wio_plan["values"]
'''
    source = source[:start] + replacement + source[end:]
    substitutions = {
        '        query_start_loc.copy_(qsl.pin_memory(), non_blocking=True)\n':
        '        query_start_loc.copy_(qsl.pin_memory() if _wio_plan is None else _wio_plan["pinned_qsl"], non_blocking=True)\n',
        '            max_query_len=int((qsl[1:] - qsl[:-1]).max()) if common.num_reqs else 0,\n':
        '            max_query_len=(int((qsl[1:] - qsl[:-1]).max()) if common.num_reqs else 0) if _wio_plan is None else _wio_plan["max_query_len"],\n',
        '            max_seq_len=int(seq_lens.max()) if common.num_reqs else 0,\n':
        '            max_seq_len=(int(seq_lens.max()) if common.num_reqs else 0) if _wio_plan is None else _wio_plan["max_seq_len"],\n',
        '        local = super().build(common_prefix_len, local_common, fast_build, **kwargs)\n':
        '''        local = super().build(common_prefix_len, local_common, fast_build, **kwargs)
        if _wio_plan is None:
            _wio_cpu_plan_store(_wio_scope,
                (start, end, per_rank, padded, qsl, seq_lens, actual_start, actual_end),
                local_common.max_query_len, local_common.max_seq_len)
''',
    }
    for old, new in substitutions.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    ast.parse(source)
    return source.encode()
