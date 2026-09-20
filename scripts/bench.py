import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * q))] if values else None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--url', default='http://127.0.0.1:8080/v1/chat/completions')
    p.add_argument('--limit', type=int, default=48)
    p.add_argument('--concurrency', type=int, default=12)
    p.add_argument('--max-tokens', type=int, default=1024)
    a = p.parse_args()
    prompts = [json.loads(x)['question'] for x in Path(a.dataset).read_text().splitlines()[:a.limit]]
    sem = asyncio.Semaphore(a.concurrency)
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(i, prompt):
            async with sem:
                start = time.perf_counter()
                first = None
                token_events = []
                usage = {}
                error = None
                body = {'model': 'dsv4', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                        'max_tokens': a.max_tokens, 'ignore_eos': True, 'stream': True,
                        'stream_options': {'include_usage': True}}
                try:
                    async with session.post(a.url, json=body) as resp:
                        if resp.status != 200:
                            error = f'HTTP {resp.status}: {(await resp.text())[:500]}'
                        else:
                            buf = b''
                            async for chunk in resp.content.iter_any():
                                buf += chunk
                                while b'\n\n' in buf:
                                    event, buf = buf.split(b'\n\n', 1)
                                    for line in event.splitlines():
                                        if not line.startswith(b'data: '):
                                            continue
                                        payload = line[6:].strip()
                                        if payload == b'[DONE]':
                                            continue
                                        try:
                                            data = json.loads(payload)
                                        except json.JSONDecodeError:
                                            continue
                                        if data.get('usage'):
                                            usage = data['usage']
                                        for choice in data.get('choices', []):
                                            delta = choice.get('delta') or {}
                                            if delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning'):
                                                now = time.perf_counter()
                                                if first is None:
                                                    first = now
                                                token_events.append(now)
                except Exception as exc:
                    error = repr(exc)
                end = time.perf_counter()
                tokens = usage.get('completion_tokens') or len(token_events)
                return {'i': i, 'start': start, 'end': end, 'ttft_ms': (first - start) * 1000 if first else None,
                        'tpot_ms': (end - first) * 1000 / (tokens - 1) if first and tokens > 1 else None,
                        'output_tokens': tokens, 'input_tokens': usage.get('prompt_tokens'),
                        'chunks': len(token_events), 'error': error}
        wall_start = time.perf_counter()
        results = await asyncio.gather(*(one(i, prompt) for i, prompt in enumerate(prompts)))
        wall_end = time.perf_counter()
    good = [r for r in results if r['error'] is None and r['ttft_ms'] is not None and r['tpot_ms'] is not None]
    ttf = [r['ttft_ms'] for r in good]
    tpot = [r['tpot_ms'] for r in good]
    summary = {'n': len(results), 'success': len(good), 'fail': len(results)-len(good),
               'concurrency': a.concurrency, 'max_tokens': a.max_tokens,
               'duration_s': wall_end - wall_start,
               'output_tps': sum(r['output_tokens'] for r in good)/(wall_end-wall_start),
               'ttft_ms_mean': statistics.mean(ttf) if ttf else None,
               'ttft_ms_p50': pct(ttf, .5), 'ttft_ms_p90': pct(ttf, .9),
               'tpot_ms_mean': statistics.mean(tpot) if tpot else None,
               'tpot_ms_p50': pct(tpot, .5), 'tpot_ms_p90': pct(tpot, .9)}
    Path(a.out).write_text(json.dumps({'summary': summary, 'requests': results}, indent=2))
    print(json.dumps(summary, indent=2))
    if len(good) != len(results):
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
