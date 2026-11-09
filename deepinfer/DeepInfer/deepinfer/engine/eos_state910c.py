"""Local serving-state adapter prototype; no model or kernel replacement.

The original sampler acceptance count is independent of output clipping.
EOS is inclusive and wins a simultaneous length limit, as in pinned _check_stop.
Inputs stay unchanged; no device-to-host read or synchronization is introduced.
Not installed into the service. Context admission must already be valid.
"""
import torch


def clip_output(target, commit, finished, ignore_eos, *, eos_token_id):
    if type(eos_token_id) is not int or not 0 <= eos_token_id < 2**31:
        raise ValueError('invalid EOS token')
    if target.shape != (4, 6) or target.dtype != torch.int32:
        raise ValueError('expected int32 target[4,6]')
    for value, dtype in ((commit, torch.int32), (finished, torch.bool),
                         (ignore_eos, torch.bool)):
        if value.shape != (4,) or value.dtype != dtype or value.device != target.device:
            raise ValueError('expected matching four-slot device state')
    columns = torch.arange(6, dtype=torch.int32, device=target.device)[None, :]
    valid = (columns < commit[:, None]) & ~finished[:, None]
    stop = valid & ~ignore_eos[:, None] & (target == eos_token_id)
    first = torch.where(stop, columns, 6).amin(dim=1)
    count = torch.minimum(commit, first + 1)
    count = torch.where(finished, torch.zeros_like(count), count)
    last = torch.where((columns == count[:, None] - 1) & (count[:, None] > 0),
                       target, -1).amax(dim=1)
    emitted = torch.where(columns < count[:, None], target, -1)
    return {'commit': count, 'last': last, 'emitted': emitted,
            'eos_hit': (first < 6) & (count > 0)}


def finish_state(remaining, finished, decision):
    # Preserve the real length budget after EOS. The independent finished bit
    # prevents future emission even if the remaining length budget is positive.
    count = decision['commit']
    left = remaining - count
    done = finished | decision['eos_hit'] | ((count > 0) & (left == 0))
    return left, done


def finish_reason(last_token, output_length, max_tokens, *, ignore_eos, eos_token_id):
    if not ignore_eos and last_token == eos_token_id:
        return 'stop'
    if output_length == max_tokens:
        return 'length'
    return None
