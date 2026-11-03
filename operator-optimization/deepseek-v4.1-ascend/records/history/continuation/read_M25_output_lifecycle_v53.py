"""Read output publication consumers selected from the saved frontend profile,not the native decoder."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib

from finalize_origin_coverage_v9 import N, O, R, W, read, write
from finalize_Engram_whole_frontier_v30 import packet


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v53.json').exists()
    # Bounded availability check of the intended original-version source stores.
    missing_edges = [
        W / 'frozen-source-snapshot/vllm/vllm/distributed/eplb/eplb_state.py',
        R / 'sources/vllm-workspace/vllm/vllm/distributed/eplb/eplb_state.py',
        W / 'frozen-source-snapshot/vllm/vllm/model_executor/utils.py',
        R / 'sources/vllm-workspace/vllm/vllm/model_executor/utils.py',
    ]
    availability = [dict(path=str(p), exists=p.exists()) for p in missing_edges]
    assert not any(r['exists'] for r in availability), 'A same-run source edge became available;use it before declaring the source dependency missing.'
    write(O / 'F21-pinned-source-edge-availability-v53.json', dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), checked=availability,
        scope='Only these intended original-version stores;not a whole-disk or server absence proof. Other-version files are not a replacement.',
        original_source_route_restored=False, new_remote_query=False, overall_complete=False))
    frontend = read(O / 'frontend-function-contract-v14.json')
    prior = packet('frontend-exact-contracts166-v14', False)
    exact_detok = next(f for f in prior['result'] if f['path'].endswith('/engine/detokenizer.py'))
    root = W / 'frozen-source-snapshot/vllm/vllm/v1/engine'
    specs = [
        ('output_processor.py', 'e47b86cd69ce1c1e3f53dac655eaa75c32a4057d10f4a1db16bb83a397f2a329', False,
         {'RequestOutputCollector': ['__init__', 'put', 'get', 'get_nowait'],
          'RequestState': ['make_request_output', '_new_request_output', '_new_completion_output'],
          'OutputProcessor': ['process_outputs', '_finish_request']}),
        ('async_llm.py', '52cbf404f1dbc99c6bebf4a50a7eb18ff842a1f53106a45f365523192d816be1', False,
         {'AsyncLLM': ['generate', '_run_output_handler', 'abort']}),
        ('detokenizer.py', exact_detok['sha256'], True,
         {'BaseIncrementalDetokenizer': ['__init__', 'get_next_output_text']}),
    ]
    gate = dict(status='PROFILE_HAS_DECODER_AND_OUTPUT_PATHS;OUTPUT_LIFECYCLE_CONTRACT_NEEDS_POINT_SOURCE',
        previous_turn_classification='PROGRESS:v52quant_callback_and_plane_producers',
        profile_evidence=['frontend-function-contract-v14.json', 'module-phase-branch-matrix-v52.json#M25'],
        question='Which output modes already coalesce buffered results,and how do stop/finish/error/cancel and incremental text/token/logprob ownership constrain output-pipeline fusion or splitting?',
        allowed_source=[dict(path=str(root / name), sha256=sha, historical166_hash_matched=pin,
                             selected_methods=methods) for name, sha, pin, methods in specs],
        reused_source='Detokenizer.update96-143andnativeDecodeStreamconstructor/protected_stepalreadyreadinv14;reuse,do notreopenpromptprimingorRust/GILcauses.',
        source_identity_scope='Detokenizerhashmatchesv14remote166source. Outputprocessor/async_llmarefrozenframeworkreferencefilesuntilindividual166hashcanbechecked.',
        no_model_torch_import_or_device_service_health_process_work=True, no_remote_work=True,
        no_broad_repository_or_SDK_audit=True, whole_module_closed=False, overall_complete=False)
    write(O / 'PROFILE-CODE-QUESTION-GATE-v53.json', gate)
    files = []
    for name, sha, historical, methods in specs:
        p = root / name;raw = p.read_bytes();assert hashlib.sha256(raw).hexdigest() == sha
        tree = ast.parse(raw);lines = raw.decode().splitlines();parts = []
        for cls_name, names in methods.items():
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
            for method in names:
                matches = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method]
                assert len(matches) == 1, (cls_name, method)
                f = matches[0]
                parts.append(dict(name=cls_name + '.' + method, start=f.lineno, end=f.end_lineno,
                    source=[str(i + 1) + ': ' + lines[i] for i in range(f.lineno - 1, f.end_lineno)],
                    direct_calls=sorted({ast.unparse(n.func) for n in ast.walk(f) if isinstance(n, ast.Call)})))
        frozen = N / 'M25-output-source-v53' / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.exists():assert hashlib.sha256(frozen.read_bytes()).hexdigest() == sha
        else:frozen.write_bytes(raw)
        files.append(dict(path=str(p), frozen_path=str(frozen), sha256=sha,
                          historical166_hash_matched=historical, parts=parts))
    write(O / 'M25-output-lifecycle-source-v53.json', dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), files=files,
        evidence_profile=str(O / 'frontend-function-contract-v14.json'),
        existing_no_per_function_onCPU_claim=frontend['no_per_function_onCPU_claim'],
        scope=gate['source_identity_scope'], implementation_changed=False,
        no_model_import_remote_or_device_work=True, whole_module_closed=False, overall_complete=False))
    for f in files:
        for p in f['parts']:
            if p['name'].startswith(('RequestOutputCollector.', 'RequestState.')):print('\n'.join(p['source']))


if __name__ == '__main__':
    main()
