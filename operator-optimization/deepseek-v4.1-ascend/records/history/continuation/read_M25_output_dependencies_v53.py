"""Read direct terminal/error/logprob/stop callees;defer unavailable same-version add implementation."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib

from finalize_origin_coverage_v9 import N, O, R, W, read, write


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v53.json').exists()
    root = W / 'frozen-source-snapshot/vllm/vllm'
    gate = read(O / 'PROFILE-CODE-QUESTION-GATE-v53.json')
    output = root / 'v1/engine/output_processor.py';raw = output.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'e47b86cd69ce1c1e3f53dac655eaa75c32a4057d10f4a1db16bb83a397f2a329'
    tree = ast.parse(raw);lines = raw.decode().splitlines();parts = []
    for cls_name, names in {'RequestOutputCollector':['close'], 'RequestState':['apply_streaming_update'],
                            'OutputProcessor':['abort_requests','propagate_error']}.items():
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
        for name in names:
            f = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            parts.append(dict(name=cls_name + '.' + name, start=f.lineno, end=f.end_lineno,
                source=[str(i + 1) + ': ' + lines[i] for i in range(f.lineno - 1, f.end_lineno)]))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RequestState')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    stream_input_origins = [dict(line=n.lineno, constructor=ast.unparse(n.func), expression=ast.unparse(k.value))
        for n in ast.walk(cls) if isinstance(n, ast.Call) for k in n.keywords if k.arg == 'stream_input']
    for n in ast.walk(init):
        targets = getattr(n, 'targets', [])
        if isinstance(n, ast.AnnAssign):targets = [n.target]
        if any(ast.unparse(a) in ['self.stream_interval', 'self.sent_tokens_offset', 'self.streaming_input'] for a in targets):
            parts.append(dict(name='RequestState.output_mode_state', start=n.lineno, end=n.end_lineno,
                source=[str(i + 1) + ': ' + lines[i] for i in range(n.lineno - 1, n.end_lineno)]))
    for n in tree.body:
        targets = getattr(n, 'targets', [])
        if isinstance(n, ast.AnnAssign):targets = [n.target]
        if any(isinstance(a, ast.Name) and a.id == 'STREAM_FINISHED' for a in targets):
            parts.append(dict(name='STREAM_FINISHED_identity_marker', start=n.lineno, end=n.end_lineno,
                source=[str(i + 1) + ': ' + lines[i] for i in range(n.lineno - 1, n.end_lineno)]))
    files = [dict(path=str(output), frozen_path=str(N / 'M25-output-source-v53/output_processor.py'),
                  sha256=hashlib.sha256(raw).hexdigest(), historical166_hash_matched=False, parts=parts)]
    gate['allowed_source'] = [r for r in gate['allowed_source'] if not
        (r['path'].endswith('/v1/engine/utils.py') and r.get('selected_methods') == ['check_stop_strings'])]
    for rel, cls_name, function, historical in [('v1/engine/detokenizer.py', None, 'check_stop_strings', True),
                                               ('v1/engine/logprobs.py', 'LogprobsProcessor', 'pop_prompt_logprobs', False)]:
        p = root / rel;b = p.read_bytes();sha = hashlib.sha256(b).hexdigest()
        if historical:assert sha == '213d71cf6eefcea061b28b656cf8a08af60c9f3513403e724b16876995f3de93'
        descriptor = dict(path=str(p), sha256=sha, historical166_hash_matched=historical,
            class_name=cls_name, selected_methods=[function], reason='Directselectedoutput-pathcallee;detokenizerfilepinreusedwhereavailable.')
        if descriptor not in gate['allowed_source']:gate['allowed_source'].append(descriptor)
        write(O / 'PROFILE-CODE-QUESTION-GATE-v53.json', gate)
        t = ast.parse(b);ls = b.decode().splitlines()
        parent = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == cls_name) if cls_name else t
        f = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == function)
        frozen = N / 'M25-output-source-v53' / Path(rel).name
        if frozen.exists():assert hashlib.sha256(frozen.read_bytes()).hexdigest() == sha
        else:frozen.write_bytes(b)
        files.append(dict(path=str(p), frozen_path=str(frozen), sha256=sha, historical166_hash_matched=historical,
            parts=[dict(name=(cls_name + '.' if cls_name else '') + function, start=f.lineno, end=f.end_lineno,
                source=[str(i + 1) + ': ' + ls[i] for i in range(f.lineno - 1, f.end_lineno)])]))
    add_paths = [root / 'outputs.py', R / 'sources/vllm-workspace/vllm/vllm/outputs.py']
    add_availability = [dict(path=str(p), exists=p.exists()) for p in add_paths]
    assert not any(r['exists'] for r in add_availability), 'Same-run add implementation available;read it before declaring this edge unavailable.'
    gate['allowed_source'].append(dict(path=str(output), sha256=files[0]['sha256'], historical166_hash_matched=False,
        selected_methods=['RequestOutputCollector.close','RequestState.apply_streaming_update','OutputProcessor.abort_requests','OutputProcessor.propagate_error','STREAM_FINISHED marker'],
        reason='Directcleanup/error/streamingterminalcalleesfromselectedAsyncLLMoutputpipeline.'))
    write(O / 'PROFILE-CODE-QUESTION-GATE-v53.json', gate)
    write(O / 'M25-output-terminal-dependency-source-v53.json', dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), files=files,
        stream_input_constructor_origins=stream_input_origins,
        same_version_RequestOutput_add_lookup=add_availability,
        add_field_merge_semantics_attested=False,
        scope='Directselectedcallee/referencebytesonly. No claim of exactRequestOutput.addfieldmerges,actual166helperidentities orbackendGIL/CPUcost.',
        no_model_import_remote_or_device_work=True, implementation=False,
        whole_module_closed=False, overall_complete=False))
    for f in files:
        for p in f['parts']:print('\n'.join(p['source']))


if __name__ == '__main__':
    main()
