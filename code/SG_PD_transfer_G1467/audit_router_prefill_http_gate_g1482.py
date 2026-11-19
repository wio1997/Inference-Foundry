"""Verify the Prefill response gate in the exact active AArch64 Router build."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from audit_cold_transfer_histograms_g1467 import read_receipt


RECEIPTS = {
    'mapped': ('c1545f31387ccbc65c1a4a582cd5ee238405a89e6e5dbfac020d69b6b893d89f',
               'G1482_BOUND_ROUTER_ELF_JSON'),
    'source': ('4d3d0729fb06a4bc2980c40c8cc421a8fe9bba15bea5f1c19bd9f0f0e638dfe4',
               'G1482_ROUTER_PATH_SOURCE_JSON'),
    'symbols': ('4540ee9967d54dec72f94d99f65f98d9a6a7a374b129673a4d7564b0d651520c',
                'G1482_ROUTER_ELF_MARKERS_JSON'),
    'poll_guard': ('0477c92e19d2eb2751c3a1a53a17c5976b494f17a666d4f60d5b1a229dd78e44',
                   'G1482_ROUTER_P_HTTP_GUARD_EXTENDED_JSON'),
    'client': ('e4b75f6effd8f2bfad964001fc825fa1d8cb3da56a38ba29ec1e16509ff8b51b',
               'G1482_ACTIVE_CLIENT_HTTP_JSON'),
}


def check_matches(code, expected):
    table = dict(code['instructions'])
    for address, token in expected.items():
        assert token in ' '.join(table[address].split()), (address, token)


def verify(mapped, source, symbols, poll_guard, client):
    assert all(row['epoch'] == 1317 for row in (mapped, source, symbols, poll_guard, client))
    assert mapped['host_listener_pid_verified_in_original_container'] and mapped['wheel_record_sha_verified']
    assert mapped['mapping_evidence']['loaded_elf_bytes_match_wheel']
    assert mapped['wheel_metadata_version'] == source['wheel_metadata_version'] == symbols['wheel_metadata_version'] == '0.3.2'
    digest = mapped['actual_ELF_sha256']
    assert digest == source['actual_ELF_sha256'] == symbols['actual_ELF_sha256'] == poll_guard['ELF_sha256']
    assert symbols['mapped_file_and_record_still_match'] and poll_guard['mapped_file_matches']
    assert source['elf_has_symtab'] and not source['elf_has_debug_line']
    assert source['source_wrapper_byte_matches']['launch_router.py']
    assert not source['source_wrapper_byte_matches']['router_args.py']
    texts = symbols['selected_literal_presence']
    for marker in ('src/routers/http/pd_router.rs', 'Decode response status:',
                   'Consuming prefill response body (non-logprob request)',
                   'Prefill response consumed successfully', 'Error consuming prefill response:'):
        assert texts[marker]
    assert not texts['Prefill failed, aborting paired decode request']
    assert client['actual_client_adapter'] == 'VLLMCustomAPI' and client['stream_true']
    assert not client['explicit_logprobs_in_config']
    wrapper = client['relevant_adapter_source_rows']['vllm_custom_api.py']
    assert 'endpoint = "v1/completions"' in dict(wrapper['selected']).values()

    dual = poll_guard['call_evidence']['dual']
    prefill = poll_guard['call_evidence']['prefill']
    check_matches(dual, {
        '0xa3a3ac': 'bl a34354 <smg::routers::http::pd_router::PDRouter::process_prefill_response',
        '0xa3a3b4': 'cmp x21, #0x4',
        '0xa3a3b8': 'b.ne a3a3cc',
        '0xa3a3c8': 'b a3c8a8',
        '0xa3c8c8': 'ret',
        '0xa3a410': 'drop_in_place<smg::routers::http::pd_router::PDRouter::process_prefill_response',
        '0xa3a414': 'cmp x21, #0x3',
        '0xa3a418': 'b.ne a3a71c',
        '0xa3a438': 'tbz w8, #0, a38d80',
        '0xa3a4d4': 'bl 23ca7c4 <reqwest::async_impl::response::Response::bytes_stream>',
        '0xa38be8': 'bl a34354 <smg::routers::http::pd_router::PDRouter::process_prefill_response',
        '0xa38bf0': 'cmp x21, #0x4',
        '0xa38d58': 'cmp x21, #0x3',
        '0xa38d5c': 'b.ne a390a8',
        '0xa38d7c': 'tbnz w8, #0, a3a43c',
        '0xa3ca0c': 'b a3a494',
    })
    check_matches(prefill, {
        '0xa34b98': 'bl a78a14 <reqwest::async_impl::response::Response::bytes::',
        '0xa34bc0': 'b a35918',
        '0xa34e74': 'bl a78a14 <reqwest::async_impl::response::Response::bytes::',
        '0xa34e90': 'b a35918',
        '0xa35938': 'ret',
    })
    branch_ins = [(entry[0],entry[1]) for entry in dual['external_direct_branch_into_first_range']]
    assert branch_ins == [('0xa38d7c','0xa3a43c'),('0xa3ca0c','0xa3a494')]
    return {
      'epoch': 1317,
      'installed_router_version': mapped['wheel_metadata_version'],
      'active_listener_mapped_elf_and_wheel_record_sha256': digest,
      'http_client': {'route':'/v1/completions','streaming':True, 'configuration_logprobs_explicit':False,
                      'actual_adapter_sha256':wrapper['sha256'],'formal_config_sha256':client['formal_cfg_sha256']},
      'binary_scope': 'The ordinary streaming success path and its resumed Prefill future call '
                      'process_prefill_response before the Decode response is exposed as bytes_stream. '
                      'The pending-return branch exits before stream creation. The Prefill future calls '
                      'Response::bytes; the installed binary retains the non-logprob body-consumption site.',
      'prefill_processing_poll_sites_hex':['0xa3a3ac','0xa38be8'],
      'prefill_body_read_sites_hex':['0xa34b98','0xa34e74'],
      'decode_http_stream_creation_site_hex':'0xa3a4d4',
      'binary_excerpt_disassembly_sha256':{name:r['disassembly_sha256'] for name,r in poll_guard['call_evidence'].items()},
      'candidate_checkout_rust_source_sha256':source['source_files']['src/routers/http/pd_router.rs']['sha256'],
      'candidate_checkout_matched_exact_build':False,
      'same_request_timed_P_HTTP_D_HTTP_and_router_first_SSE':False,
      'verification_scope':'Static binary control flow at the bound live listener. Candidate source is context. '
                           'Actual time gated on Prefill HTTP requires per-request capture alongside Decode readiness.',
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in RECEIPTS:
        cli.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    args=cli.parse_args()
    data={key:read_receipt(getattr(args,key), *value)[0] for key,value in RECEIPTS.items()}
    report=verify(**data)
    report['private_receipts_sha256']={key:item[0] for key,item in RECEIPTS.items()}
    report['analysis_source_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
