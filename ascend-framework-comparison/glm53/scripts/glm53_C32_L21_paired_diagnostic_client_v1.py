"""L21 CPU client, fresh sixth suffix after current metadata readiness: one framework cohort, C32, 80000/600, no warmups/retries.

Adapted from glm53_C32_L13_D_diagnostic_client_v1.py. Frozen canonical AIS v2,
request-terminal and gateway files are imported, not reimplemented. This is a
diagnostic count/terminal/cache observation, never a performance certificate.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata as md
import json
import time
from pathlib import Path

import aiohttp
import torch
from aiohttp import web
from ais_bench.benchmark.models.output import RequestOutput

from glm53_formal_ais_client_v2 import Observed, cache_observation, dump, ids_sha
from glm53_request_terminal_v1 import RequestTerminals
from glm53_runtime_metrics_v1 import MetricRecorder
import glm53_C32_L20_paired_profile_control_v1 as profiler

FRAMEWORKS = {'sglang': 'sg', 'vllm': 'vl'}


def prior_gate(settings, prior):
    assert prior['project_id'] == 'glm53'
    assert prior['framework'] == settings['framework']
    assert prior['concurrency'] == settings['concurrency'] == 32
    assert prior['phase'] == 'COMPLETE'
    assert prior['baseline_accepted'] is True
    assert prior['run_id'] == settings['prior_run_id']
    assert prior['namespace'] == settings['cache_namespace']
    assert prior['input_manifest_sha256'] == settings['canonical_manifest_sha256']
    summary = prior['summary']
    assert summary['formal_requests'] == 128
    assert summary['strict_count_lifecycle_successes'] == 128
    assert summary['strict_count_lifecycle_failures'] == 0
    assert summary['request_count_and_clock_complete'] is True
    assert summary['all_formal_prefix_hits_verified'] is True
    gate = prior['cache_gate']
    assert gate['project_id'] == 'glm53'
    assert gate['framework'] == settings['framework']
    assert gate['namespace'] == settings['cache_namespace']
    assert gate['prefix_count'] == 32
    assert gate['all_expected_keys_present'] is True
    assert gate['actual_payload_samples_read'] is True
    assert gate['close_rc'] == 0
    assert gate['pass'] is True
    prefixes = {row['prefix_id']: row['prefix_ids_sha256_le_u32'] for row in gate['prefixes']}
    assert set(prefixes) == set(range(32))
    prior_ids = {row['request_id'] for row in prior['formal']['rows']}
    prior_ids |= {row['request_id'] for row in prior['warmup']}
    return prefixes, prior_ids


def load_inputs(settings):
    manifest_path = Path(settings['dataset']) / 'manifest.json'
    raw = manifest_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == settings['input_manifest_sha256']
    manifest = json.loads(raw)
    assert manifest['project_id'] == 'glm53'
    assert manifest['diagnostic_only'] is True
    assert manifest['performance_certificate_allowed'] is False
    assert manifest['fresh_suffix_lcp_exact'] is True
    assert manifest['concurrency'] == 32
    assert manifest['requests'] == 32
    assert manifest['warmups'] == 0
    assert manifest['canonical_manifest_sha256'] == settings['canonical_manifest_sha256']
    assert len(manifest['rows']) == 32
    inputs = []
    for index, meta in enumerate(manifest['rows']):
        item = json.loads((manifest_path.parent / meta['file']).read_text())
        assert item['prefix_id'] == index
        assert item['variant'] == 6
        assert len(item['input_ids']) == 80000
        assert item['requested_output_tokens'] == 600
        assert ids_sha(item['input_ids']) == meta['ids_sha256_le_u32']
        assert ids_sha(item['input_ids'][:74400]) == meta['prefix_sha256_le_u32']
        assert item['lcp_against_previous_five'] == [74400] * 5
        inputs.append(item)
    return manifest, inputs


def endpoints_for(framework):
    if framework == 'sglang':
        return [{'role': 'D', 'node': node, 'url': f'http://172.16.12.{node}:30600/metrics'}
                for node in (36, 41)]
    return [{'role': 'D', 'node': node, 'url': f'http://172.16.12.{node}:30400/metrics'}
            for node in (43, 45)]


async def run(settings):
    assert torch.__version__ == '2.10.0+cpu'
    assert md.version('ais_bench_benchmark') == '3.1.0'
    framework = settings['framework']
    assert framework in FRAMEWORKS
    assert settings['project_id'] == 'glm53'
    assert settings['diagnostic_only'] is True
    assert settings['performance_certificate_allowed'] is False
    assert settings['execution_authorized'] is True
    assert settings['concurrency'] == 32
    assert settings['logical_requests'] == 32
    assert settings['warmups'] == 0 and settings['retries'] == 0
    assert settings['cache_clear_or_repopulate'] is False

    output = Path(settings['output'])
    assert output.is_dir()
    with (output / 'diagnostic-client.once').open('x') as file:
        file.write(settings['run_id'])

    prior_raw = Path(settings['prior_report']).read_bytes()
    assert hashlib.sha256(prior_raw).hexdigest() == settings['prior_report_sha256']
    prior = json.loads(prior_raw)
    prefixes, prior_ids = prior_gate(settings, prior)
    acceptance_raw = Path(settings['prior_acceptance']).read_bytes()
    assert hashlib.sha256(acceptance_raw).hexdigest() == settings['prior_acceptance_sha256']
    acceptance = json.loads(acceptance_raw)
    assert acceptance['project_id'] == 'glm53'
    assert acceptance['framework'] == framework
    assert acceptance['measurement_valid'] is True
    assert acceptance['run_id'] == settings['prior_run_id']

    manifest, inputs = load_inputs(settings)
    for item in inputs:
        assert prefixes[item['prefix_id']] == item['prefix_sha256_le_u32']

    namespace = settings['cache_namespace']
    tag = FRAMEWORKS[framework]
    request_ids = [namespace + f'-diag-L21-{tag}-p{item["prefix_id"]:02d}-v6' for item in inputs]
    assert len(set(request_ids)) == 32
    assert not (set(request_ids) & prior_ids)
    assert all('-diag-L13-' not in rid for rid in request_ids)
    assert not any('-diag-L21-' in rid for rid in prior_ids)

    report = {
        'project_id': 'glm53',
        'framework': framework,
        'cohort': settings['cohort'],
        'run_id': settings['run_id'],
        'concurrency': 32,
        'diagnostic_only': True,
        'performance_certificate_allowed': False,
        'cache_namespace': namespace,
        'prior_report_sha256': settings['prior_report_sha256'],
        'prior_acceptance_sha256': settings['prior_acceptance_sha256'],
        'input_manifest_sha256': settings['input_manifest_sha256'],
        'canonical_manifest_sha256': settings['canonical_manifest_sha256'],
        'warmups': 0,
        'retries': 0,
        'cache_clear_or_repopulate': False,
        'request_id_policy': 'namespace + -diag-L21-{sg|vl}-pNN-v6; disjoint from L13/L17 IDs',
        'planned': {
            'logical_requests': 32,
            'backend_HTTP_attempts': 64,
            'P_output_tokens': 32,
            'D_output_tokens': 19200,
            'all_successful_backend_output_tokens_if_all_pass': 19232,
        },
        'phase': 'STARTING',
        'rows': [],
        'gateway': [],
        'SDK_init_calls': 0,
        'automatic_model_request_retries': False,
    }

    runner = None
    recorder = None
    metrics_task = None
    profile_task = None
    stop = asyncio.Event()
    try:
        if framework == 'sglang':
            from glm53_sglang_pd_gateway_v3 import SGLangPDGateway as Gateway
        else:
            from glm53_vllm_pd_gateway_v3 import VLLMPullGateway as Gateway
        timeout = aiohttp.ClientTimeout(total=420, connect=30)
        connector = aiohttp.TCPConnector(limit=0, force_close=True)
        async with aiohttp.ClientSession(trust_env=False, timeout=timeout,
                                         connector=connector) as upstream:
            terminals = RequestTerminals()
            gateway = Gateway(upstream, [namespace], report['gateway'],
                              terminal_registry=terminals)
            runner = web.AppRunner(gateway.application())
            await runner.setup()
            site = web.TCPSite(runner, '127.0.0.1', 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            async with aiohttp.ClientSession(trust_env=False, timeout=timeout,
                                             connector=aiohttp.TCPConnector(
                                                 limit=0, force_close=True)) as client:
                model = Observed(path='', model='glm53-w8a8', stream=True,
                                 host_ip='127.0.0.1', host_port=port,
                                 run_namespace=namespace,
                                 expected_input_tokens=80000,
                                 expected_output_tokens=600)
                recorder = MetricRecorder(framework, endpoints_for(framework),
                                          output / 'diagnostic-D-metrics.jsonl.gz')
                before = await recorder.sample(upstream)
                assert before['complete_all_D_DP']
                assert before['sampled_total_running'] == 0
                assert before['sampled_total_waiting'] == 0
                metrics_task = asyncio.create_task(recorder.run(upstream, stop))

                def profile_save(value):
                    path = output / 'D-profile-control-progress.json'
                    temporary = path.with_suffix('.tmp')
                    dump(temporary, value)
                    temporary.replace(path)

                profile_task = asyncio.create_task(asyncio.to_thread(
                    profiler.control, settings['profile_settings'], profiler.request,
                    save=profile_save))
                admission = asyncio.Event()
                inflight = 0
                events = []

                async def submit(item, rid):
                    nonlocal inflight
                    await admission.wait()
                    out = RequestOutput(perf_mode=True)
                    out.uuid = rid
                    terminals.reserve(rid)
                    begin = time.perf_counter_ns()
                    inflight += 1
                    events.append({'event': 'dispatch', 'request_id': rid,
                                   'perf_counter_ns': begin, 'inflight': inflight})
                    row = {'request_id': rid, 'prefix_id': item['prefix_id'],
                           'variant': 5, 'input_sha256_le_u32': item['ids_sha256_le_u32'],
                           'outer_dispatch_perf_counter_ns': begin}
                    try:
                        await model.generate(item['input_ids'], 600, out, session=client)
                        row['terminal'] = await terminals.wait_terminal(rid, 15)
                    except Exception as error:
                        row['error'] = repr(error)
                    finally:
                        out.input = None
                        row['native'] = out.to_dict()
                        row['outer_terminal_perf_counter_ns'] = time.perf_counter_ns()
                        gw = row.get('terminal', {}).get('gateway', {})
                        row['gateway_status'] = gw.get('status')
                        row['cache_observation'] = cache_observation(
                            framework, gw, row['native'])
                        usage = row['native'].get('extra_details_data', {}).get(
                            'glm53_service_usage', {})
                        prefill = gw.get('prefill_usage') or {}
                        row['prefill_count_pass'] = bool(
                            prefill.get('prompt_tokens') == 80000
                            and prefill.get('completion_tokens') == 1)
                        row['count_and_terminal_pass'] = bool(
                            not row.get('error')
                            and row['native']['success']
                            and row['native']['input_tokens'] == usage.get('prompt_tokens') == 80000
                            and row['native']['output_tokens'] == usage.get('completion_tokens') == 600
                            and row['native'].get('extra_details_data', {}).get(
                                'glm53_finish_reason') == 'length'
                            and gw.get('status') == 'D_STREAM_COMPLETE')
                        inflight -= 1
                        events.append({'event': 'terminal', 'request_id': rid,
                                       'perf_counter_ns': row['outer_terminal_perf_counter_ns'],
                                       'inflight': inflight})
                        report['rows'].append(row)
                        dump(output / 'diagnostic-progress.json', report)

                tasks = [asyncio.create_task(submit(item, rid))
                         for item, rid in zip(inputs, request_ids)]
                report['phase'] = 'DIAGNOSTIC'
                admission.set()
                await asyncio.gather(*tasks)
                await gateway.wait_idle(15)
                report['pending_terminal_ids'] = terminals.pending
                stop.set()
                await metrics_task
                metrics_task = None
                report['D_metrics'] = recorder.close()
                recorder = None
                report['profile_control'] = await profile_task
                profile_task = None
                assert inflight == 0 and len(report['rows']) == 32
                report['client_inflight_events'] = events
                report['wall_seconds'] = (
                    max(row['outer_terminal_perf_counter_ns'] for row in report['rows'])
                    - min(row['outer_dispatch_perf_counter_ns'] for row in report['rows'])
                ) / 1e9
                report['wall_bound_seconds'] = 420
                report['wall_bound_exceeded'] = report['wall_seconds'] > 420
                report['count_and_terminal_successes'] = sum(
                    row['count_and_terminal_pass'] for row in report['rows'])
                report['prefill_count_successes'] = sum(
                    row['prefill_count_pass'] for row in report['rows'])
                report['raw_cache_observation_passes'] = sum(
                    row['cache_observation']['pass'] for row in report['rows'])
                report['actual'] = actual_counts(report)
                report['diagnostic_pass'] = bool(
                    report['count_and_terminal_successes'] == 32
                    and report['prefill_count_successes'] == 32
                    and not report['pending_terminal_ids']
                    and report['actual']['all_64_backend_successes'])
                report['phase'] = 'COMPLETE'
    except BaseException as error:
        report['phase'] = 'FAILED'
        report['error'] = repr(error)
        raise
    finally:
        stop.set()
        if metrics_task:
            await metrics_task
        if profile_task:
            report['profile_control'] = await profile_task
        if recorder:
            report['D_metrics'] = recorder.close()
        if runner:
            await runner.cleanup()
        report['actual'] = actual_counts(report)
        report['backend_HTTP_attempts_actual'] = report['actual']['backend_HTTP_attempts']
        report['profile_completion'] = (
            report.get('profile_control', {}).get('profile_completion', 'NOT_AVAILABLE'))
        dump(output / 'diagnostic-result.json', report)
        print(json.dumps({key: report.get(key) for key in [
            'project_id', 'framework', 'run_id', 'phase', 'diagnostic_only',
            'performance_certificate_allowed', 'backend_HTTP_attempts_actual',
            'count_and_terminal_successes', 'prefill_count_successes',
            'raw_cache_observation_passes', 'profile_completion']}), flush=True)
    return report


def actual_counts(report):
    rows = report['rows']
    gateway = report['gateway']
    prefill_attempts = sum('prefill_dispatch_monotonic_ns' in row for row in gateway)
    decode_attempts = sum('decode_dispatch_monotonic_ns' in row for row in gateway)
    prefill_ok = [row for row in gateway
                  if row.get('prefill_http_status') == 200
                  and row.get('prefill_usage', {}).get('completion_tokens') == 1]
    decode_http_ok = [row for row in gateway if row.get('decode_http_status') == 200]
    strict_decode = [row for row in rows if row.get('count_and_terminal_pass')]
    p_tokens = sum(row['prefill_usage'].get('completion_tokens', 0) for row in prefill_ok)
    d_rows = [row for row in rows
              if row['native'].get('success')
              and type(row['native'].get('output_tokens')) is int
              and row['native']['output_tokens'] > 0
              and row.get('gateway_status') == 'D_STREAM_COMPLETE']
    d_tokens = sum(row['native']['output_tokens'] for row in d_rows)
    all_64 = (prefill_attempts == 32 and decode_attempts == 32
              and len(prefill_ok) == 32 and len(strict_decode) == 32
              and report.get('count_and_terminal_successes') == 32)
    return {
        'backend_HTTP_attempts': prefill_attempts + decode_attempts,
        'successful_backend_inferences': len(prefill_ok) + len(strict_decode),
        'decode_HTTP_200': len(decode_http_ok),
        'successful_P_output_tokens': p_tokens,
        'successful_D_output_tokens': d_tokens,
        'all_successful_backend_output_tokens': p_tokens + d_tokens,
        'planned_if_all_pass': {
            'backend_HTTP_attempts': 64,
            'P_output_tokens': 32,
            'D_output_tokens': 19200,
            'all_successful_backend_output_tokens': 19232,
        },
        'all_64_backend_successes': bool(all_64),
        'booking_rule': 'Actual values only; planned 64/19232 is never booked.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    raw = Path(args.settings).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    asyncio.run(run(json.loads(raw)))


if __name__ == '__main__':
    main()
