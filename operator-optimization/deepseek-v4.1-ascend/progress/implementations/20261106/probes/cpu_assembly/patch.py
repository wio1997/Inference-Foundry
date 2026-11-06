"""Replace only the two CPU assembly expressions in the admitted CP source."""
import ast
from hashlib import sha256

PARENT_SHA = 'ef972657414294559c89c04e93ecb2be0965b5c149891dd32c8cd2f09ad69cf4'


def render(raw, mode='copy'):
    assert sha256(raw).hexdigest() == PARENT_SHA
    assert mode in ('copy', 'owned')
    original = raw.decode()
    source = original
    marker = 'class _ReplicatedCacheMetadataBuilder'
    prefix = ('from wio_cpu_assembly import make_replacer as _wio_make_replacer, publish_owned as _wio_publish_owned\n'
              '_wio_replace_common = _wio_make_replacer(AscendCommonAttentionMetadata)\n'
              '_wio_replace_result = _wio_make_replacer(AscendDSAV41Metadata)\n\n\n')
    assert source.count(marker) == 1
    source = source.replace(marker, prefix + marker)
    old_common = '        local_common = common.replace(\n'
    new_common = '        local_common = _wio_replace_common(common, common.replace,\n'
    old_return = '        return replace(local, global_metadata=global_metadata, cp_token_range=(start, end, per_rank, padded))'
    new_return = ('        return _wio_replace_result(local, global_metadata=global_metadata, cp_token_range=(start, end, per_rank, padded))'
                  if mode == 'copy' else
                  '        return _wio_publish_owned(local, AscendDSAV41Metadata, global_metadata, (start, end, per_rank, padded))')
    for before, after in ((old_common, new_common), (old_return, new_return)):
        assert source.count(before) == 1
        source = source.replace(before, after)
    ast.parse(source)
    recovered = source.replace(prefix, '').replace(new_common, old_common).replace(new_return, old_return)
    assert recovered == original
    return source.encode()
