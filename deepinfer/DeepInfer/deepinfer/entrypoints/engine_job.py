"""Existing HTTP file transport over original LLMEngine public methods.

The original Engine owns admission, scheduling, cache actions, sampling and
stopping. This module routes wire requests to DP drivers and copies outputs
before the original output pool can recycle them.
"""
import base64
import json
import inspect
from pathlib import Path
import time

from .file_transport import PROTOCOL, atomic_json


def completion_request(body, tokenize, *, max_model_len):
    """Map HTTP fields to the original parameter object and wire data."""
    from deepinfer.sampling_params import SamplingParams
    params = dict(body.get('sampling_params') or {})
    for name in inspect.signature(SamplingParams).parameters:
        if name in body:
            if name in params and params[name] != body[name]:
                raise ValueError(f'conflicting sampling parameter {name}')
            params[name] = body[name]
    if 'seed' in body:
        if 'seq_seed' in params and params['seq_seed'] != body['seed']:
            raise ValueError('conflicting seed and seq_seed')
        params['seq_seed'] = body['seed']
    params = SamplingParams(**params)
    prompt = body['prompt']
    ids = list(tokenize(prompt)) if isinstance(prompt, str) else prompt
    if not isinstance(ids, list) or not ids or any(type(token) is not int or not 0 <= token < 2**31 for token in ids):
        raise ValueError('prompt must be text or a nonempty INT32 token list')
    if len(ids)+params.max_tokens > max_model_len:
        raise ValueError('prompt and output exceed this deployment context limit')
    return dict(token_ids=ids, sampling_params=vars(params).copy(), max_tokens=params.max_tokens,
        ignore_eos=params.ignore_eos, user_id=str(body.get('user_id', '0')),
        extra=body.get('extra'), multi_modal_data=body.get('multi_modal_data'),
        chunk_checkpoints=body.get('chunk_checkpoints'))


def request_kwargs(request):
    import numpy as np
    from deepinfer.sampling_params import SamplingParams

    ids = request['token_ids']
    if not isinstance(ids, list) or not ids or any(type(token) is not int or not 0 <= token < 2**31 for token in ids):
        raise ValueError('token_ids must be a nonempty INT32 token list')
    params = dict(request.get('sampling_params', {}))
    for key in ('max_tokens', 'ignore_eos'):
        if key in request:
            if key in params and params[key] != request[key]:
                raise ValueError(f'conflicting {key} in wire request')
            params[key] = request[key]
    extra = dict(request.get('extra') or {})
    mm = request.get('multi_modal_data')
    if 'multi_modal_data' in extra:
        mm = extra['multi_modal_data']
    if mm is not None:
        mm = dict(mm)
        if 'image_bytes' in mm and mm['image_bytes'] is not None:
            mm['image_bytes'] = [base64.b64decode(value, validate=True) if isinstance(value, str) else value
                                 for value in mm['image_bytes']]
        if 'multi_modal_data' in extra:
            extra['multi_modal_data'] = mm
    return dict(user_id=str(request.get('user_id', '0')), request_id=request['id'],
        sampling_params=SamplingParams(**params), prompt_token_ids=np.asarray(ids, dtype=np.int32),
        multi_modal_data=mm, extra=extra, arrival_time=request.get('arrival_time'),
        chunk_checkpoints=request.get('chunk_checkpoints'))


def output_fields(output):
    return {name:getattr(output, name) for name in (
        'request_id','prompt_len','token_id','finish_reason','finished','token_logprobs',
        'nextn_logprobs','sum_smaller_probs','top_logprobs','top_token_ids','prompt_logprobs',
        'num_output_tokens','max_num_batched_tokens','draft_tok','accept_tok','confidence_score')}


def run_engine_job(engine, body, directory, *, progress=None):
    import torch.distributed as dist
    from deepinfer._C.engine import RequestOutput
    from deepinfer.model_executor.parallel_utils.parallel_state import PG

    if body.get('protocol') != PROTOCOL:
        raise ValueError('request transport does not target the original Engine')
    if 'single_point_ablation' in body:
        raise ValueError('original Engine variants use immutable startup configurations')
    directory = Path(directory)
    dp, size = engine.dp_rank, engine.parallel_config.data_parallel_size
    group = PG.cpu_data_parallel_group()
    records, indices, request_ids, pending_cancel = {}, set(), set(), set()
    transaction = 1
    began = time.monotonic()
    continuous = bool(body.get('continuous_admission', False))
    stream = (directory/f'events-dp{dp}.jsonl').open('w', buffering=1)

    def emit(value):
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')

    def add(rows):
        for row in rows:
            index, request = row['index'], row['request']
            if type(index) is not int or index < 0 or index in indices:
                raise ValueError('transport request indices must be distinct nonnegative integers')
            indices.add(index)
            request_id = request.get('id')
            if not isinstance(request_id, str) or not request_id or request_id in request_ids:
                raise ValueError('transport request IDs must be distinct nonempty strings')
            request_ids.add(request_id)
            if index % size != dp:
                continue
            record = dict(index=index, sequence=None, done=False, output_tokens=0, cached_tokens=0,
                          prompt_tokens=len(request.get('token_ids', [])), request=request)
            records[request_id] = record
            try:
                record['sequence'] = engine.add_request(**request_kwargs(request))
                if record['sequence'] is None:
                    raise ValueError('original Engine rejected request tokens')
            except Exception as exc:
                # add_request is a host/control call. Let the original API
                # release any partially enqueued request before reporting it.
                engine.abort_request(request_id)
                emit(dict(type='error', index=index, request_id=request_id, message=str(exc), terminal=True))
                record['done'] = True

    def outputs(values):
        for pointer in values:
            output = RequestOutput.clone(pointer)
            fields = output_fields(output)
            record = records[output.request_id]
            if record['done']:
                raise RuntimeError('original Engine emitted output after a terminal wire event')
            if not record['output_tokens']:
                record['prompt_tokens'] = output.prompt_len
                record['cached_tokens'] = record['sequence'].cached_hit_tokens
                emit(dict(type='start', index=record['index'], request_id=output.request_id,
                    prompt_tokens=output.prompt_len, cached_tokens=record['cached_tokens']))
            emit(dict(type='token', index=record['index'], step=output.num_output_tokens-1,
                elapsed=time.monotonic()-began, **fields))
            record['output_tokens'] += 1
            if output.finished:
                record['done'] = True
                emit(dict(type='done', index=record['index'], request_id=output.request_id,
                    prompt_tokens=record['prompt_tokens'], cached_tokens=record['cached_tokens'],
                    output_tokens=record['output_tokens'], finish_reason=output.finish_reason))

    def poll():
        control = None
        if dp == 0:
            try:
                path = directory/'admissions'/f'{transaction:08d}.json'
                admission = json.loads(path.read_text()) if path.exists() else None
                if admission is not None and admission.get('number') != transaction:
                    raise ValueError('transport admission transaction is out of order')
                cancels = [json.loads(path.read_text())['request_id'] for path in directory.glob('cancel-*.json')]
                control = dict(admission=admission, cancel=cancels,
                               close=not continuous or (directory/'admission-close.json').exists())
            except Exception as exc:
                control = dict(error=f'{type(exc).__name__}: {exc}')
        if size > 1:
            packet = [control]
            dist.broadcast_object_list(packet, src=dist.get_global_rank(group, 0), group=group)
            control = packet[0]
        if 'error' in control:
            raise RuntimeError(control['error'])
        return control

    try:
        add([dict(index=index, request=request) for index, request in enumerate(body['requests'])])
        atomic_json(directory/f'admission-ready-dp{dp}.json', dict(protocol=PROTOCOL, dp_rank=dp))
        while True:
            if progress is not None:
                progress()
            control = poll()
            admission = control['admission']
            if admission is not None:
                add(admission['requests'])
            pending_cancel.update(control['cancel'])
            for request_id in list(pending_cancel):
                record = records.get(request_id)
                if record is None or record['done']:
                    pending_cancel.discard(request_id)
                elif engine.abort_request(request_id)[0]:
                    pending_cancel.discard(request_id)
            active = engine.has_unfinished_requests()
            # The original DP vote above also confirms every driver has
            # consumed the transaction. Transport ACK does not allocate slots.
            if admission is not None:
                if dp == 0:
                    atomic_json(directory/f'admission-result-{transaction:08d}.json',
                        dict(number=transaction, accepted=[dict(index=row['index']) for row in admission['requests']], deferred=[]))
                transaction += 1
            outputs(engine.step() if active else engine.flush_inflight_reqs())
            if not active:
                for request_id, record in records.items():
                    if record['done']:
                        continue
                    status = str(record['sequence'].status)
                    if 'FINISHED_' in status:
                        # Ignored/cancelled requests may have no RequestOutput.
                        # Preserve the original status instead of deciding EOS.
                        emit(dict(type='error', index=record['index'], request_id=request_id,
                                  message=status, sequence_status=status, terminal=True))
                        record['done'] = True
                if control['close']:
                    if any(not record['done'] for record in records.values()):
                        raise RuntimeError('original Engine drained without a terminal request status')
                    break
                time.sleep(.001)
        result = dict(protocol=PROTOCOL, dp_rank=dp, requests=[dict(id=key,index=value['index'],
            prompt_tokens=value['prompt_tokens'],cached_tokens=value['cached_tokens'],
            output_tokens=value['output_tokens']) for key,value in records.items()], elapsed=time.monotonic()-began)
        atomic_json(directory/f'result-dp{dp}.json', result)
        return result
    finally:
        stream.close()
