"""Read two fallback bodies and their two direct recovery dependencies from the pinned snapshot."""
from datetime import datetime, timezone
import ast
import hashlib

from finalize_origin_coverage_v9 import N, O, R, W, read, write


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v49.json').exists()
    prior = read(O / 'M17-probability-consumer-and-RNG-contract-v33.json')
    remote = '/vllm-workspace/vllm-ascend/vllm_ascend/sample/rejection_sampler.py'
    expected = prior['source_files'][remote]
    source = W / 'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/sample/rejection_sampler.py'
    names = ['rejection_random_sample_pytorch', 'rejection_random_sample_block_verify_pytorch',
             'sample_recovered_tokens_pytorch', 'sample_recovered_tokens_blockwise_pytorch']
    gate = dict(
        status='PROFILE_CANNOT_DECIDE_UNEXECUTED_FALLBACK_SEMANTICS;BOUNDED_SOURCE_READ',
        previous_turn_classification='PROGRESS:v48profile_evidence_capability',
        profile_evidence=str(O / 'M17-current-probability-and-guard-scope-v33.json'),
        profile_facts='840samplerAPIs/1144TASKs/14chains include72greedyTriton tasks;recorded guard fields empty/unknown. Priorfallback row explicitly says interface-only.',
        question='What are the two PyTorch fallback acceptance/recovery/entropy/precision contracts,and which conditional distribution/fusion identities may or may not be shared across backends?',
        allowed_source=[dict(path=str(source), expected_sha256=expected, functions=names)],
        dependency_extension='The already saved sample_recovered_tokens dispatch729-795 names the two additional recovery helpers;only those direct missing bodies are added.',
        no_new_model_torch_NPU_service_health_process_work=True, no_new_remote_read=True,
        no_broad_repository_or_SDK_audit=True, whole_module_closed=False, overall_complete=False)
    write(O / 'PROFILE-CODE-QUESTION-GATE-v49.json', gate)
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    lines = raw.decode().splitlines()
    tree = ast.parse(raw)
    functions = []
    for name in names:
        matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        assert len(matches) == 1
        f = matches[0]
        functions.append(dict(name=name, start=f.lineno, end=f.end_lineno,
            source=[str(i + 1) + ': ' + lines[i] for i in range(f.lineno - 1, f.end_lineno)],
            guard_sites=[dict(line=node.lineno, condition=ast.unparse(node.test))
                         for node in ast.walk(f) if isinstance(node, ast.If)],
            direct_calls=sorted({ast.unparse(node.func) for node in ast.walk(f) if isinstance(node, ast.Call)})))
    write(O / 'M17-pytorch-fallback-source-v49.json', dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), path=str(source),
        remote_path=remote, sha256=expected, historical166source_hash_match=True,
        historical_pin=str(O / 'M17-probability-consumer-and-RNG-contract-v33.json'),
        functions=functions, no_model_import_or_execution=True, new_remote_work=False,
        scope='Two acceptance fallback bodies plus their two directly dispatched recovery dependencies;AST parsing does not import the module. This does not attest every live runtime dispatch or loaded kernel.',
        whole_module_closed=False, overall_complete=False))
    for f in functions:
        if f['name'].startswith('sample_recovered_tokens'):
            print('\n'.join(f['source']))


if __name__ == '__main__':
    main()
