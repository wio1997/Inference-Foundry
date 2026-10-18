"""Opt-in local completion of a complete max1 request at the native P engine.

Inserted into a copy of the pinned official proxy. Native fallback is unchanged.
Both remote flags are false before P admission: no remote KV lease is created.
"""
import anyio
from starlette.background import BackgroundTask


def _glm53_local_complete_eligible(api, body):
    fields = {'model', 'prompt', 'stream', 'stream_options', 'return_token_ids',
              'temperature', 'top_p', 'seed', 'cache_salt', 'min_tokens',
              'max_tokens', 'n', 'ignore_eos'}
    if api != '/completions' or not isinstance(body, dict) or set(body) - fields:
        return False
    prompt = body.get('prompt')
    return bool(
        body.get('model') == 'glm5'
        and type(body.get('max_tokens')) is int and body['max_tokens'] == 1
        and type(body.get('min_tokens')) is int and body['min_tokens'] == 1
        and body.get('ignore_eos') is True
        and type(body.get('temperature')) in (int, float) and body['temperature'] == 0
        and type(body.get('top_p')) in (int, float) and body['top_p'] == 1
        and type(body.get('seed')) is int
        and type(body.get('n', 1)) is int and body.get('n', 1) == 1
        and isinstance(body.get('cache_salt'), str) and bool(body['cache_salt'])
        and isinstance(prompt, list) and len(prompt) in (256, 257, 1536, 1537)
        and all(type(t) is int and 0 <= t < 154880 for t in prompt)
        and type(body.get('stream')) is bool
        and ((body['stream'] and body.get('stream_options') == {'include_usage': True})
             or (not body['stream'] and 'stream_options' not in body)))


async def _glm53_maybe_complete_at_P(api, request):
    if request.headers.get('X-GLM53-Route') != 'p-complete':
        return None
    body = await request.json()
    if not _glm53_local_complete_eligible(api, body):
        # Original handler and service return their original validation/result.
        return None
    local = dict(body, kv_transfer_params={'do_remote_decode': False, 'do_remote_prefill': False})
    runtime = get_runtime()
    args = get_global_args()
    request_id = request.headers.get('X-Request-Id') or next_req_id()
    score = calculate_prefill_score(len(await request.body()))
    prefiller = await runtime.schedule('begin_request', score)
    key = prefiller['key']
    opened = None
    handed_off = False
    released = False

    async def cleanup():
        nonlocal released
        # Starlette disconnect cancellation must still close upstream HTTP and
        # release the proxy's own load counter. Engine KV has no remote owner.
        with anyio.CancelScope(shield=True):
            try:
                if opened is not None:
                    await opened.aclose()
            finally:
                if not released:
                    await _abort_prefill_selection(runtime, key, score, is_initial_request=True)
                    released = True

    try:
        client = await runtime.get_client(ServerRole.PREFILL, key)
        opened = stream_service_response(client, api, local, request_id=request_id,
                                         max_retries=args.max_retries, base_delay=args.retry_delay)
        try:
            first = await opened.__anext__()
        except StopAsyncIteration:
            first = b''
        except httpx.HTTPStatusError as exc:
            return Response(content=exc.response.content, status_code=exc.response.status_code,
                            media_type='application/json')
        except httpx.RequestError as exc:
            return JSONResponse(status_code=502, content={'error': {
                'message': 'P local completion failed: ' + str(exc), 'type': 'upstream_error'}})
        logger.info('glm53 local complete request=%s remote_decode=false remote_prefill=false P=%s', request_id, key)

        async def generate():
            try:
                if first:
                    yield first
                async for chunk in opened:
                    yield chunk
            finally:
                await cleanup()

        response = StreamingResponse(generate(), media_type='text/event-stream' if body['stream'] else 'application/json',
                                     headers={'X-GLM53-Route': 'p-local-complete'}, background=BackgroundTask(cleanup))
        handed_off = True
        return response
    finally:
        if not handed_off:
            await cleanup()
