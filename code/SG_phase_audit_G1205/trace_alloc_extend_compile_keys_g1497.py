"""Opt-in capture of Triton cache-key inputs for alloc_extend_kernel diagnostics."""

from base64 import urlsafe_b64encode
from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Lock


COMPILER_SHA = '4e432986cb567b94d282f9a2c2f9911a6c3718337289fa936f240282a6a5d54b'
KERNEL = 'alloc_extend_kernel'


def fingerprint(value):
    return sha256(str(value).encode('utf-8')).hexdigest()


def key_record(actual_key, module, src, backend, options, env_vars):
    compiler = module.triton_key()
    ast_key = src.hash()
    backend_key = backend.hash()
    options_key = options.hash()
    env_key = str(sorted(env_vars.items()))
    rebuilt = f'{compiler}-{ast_key}-{backend_key}-{options_key}-{env_key}'
    assert rebuilt == actual_key, 'The installed cache-key recipe differs'
    digest = fingerprint(actual_key)
    return {
        'disk_metadata_hash': digest,
        'disk_directory_key': urlsafe_b64encode(bytes.fromhex(digest)).decode('utf-8').rstrip('='),
        'component_sha256': {
            'compiler_identity': fingerprint(compiler),
            'ast_source': fingerprint(ast_key),
            'backend_identity': fingerprint(backend_key),
            'backend_options': fingerprint(options_key),
            'invalidating_environment': fingerprint(env_key),
        },
        'ast_component_sha256': {
            'jit_function_identity': fingerprint(src.fn.cache_key),
            'attrs': fingerprint(src.attrs.hash()),
            'ordered_signature': fingerprint([v for _, v in sorted(src.signature.items())]),
            'ordered_constants': fingerprint(sorted((str(k), v) for k, v in src.constants.items())),
        },
        'numeric_constants': {str(k): v for k, v in src.constants.items()
                              if type(v) in (int, bool)},
    }


def install(module, output):
    """Install only on an explicitly selected diagnostic worker."""
    assert sha256(Path(module.__file__).read_bytes()).hexdigest() == COMPILER_SHA
    output = Path(output)
    assert output.is_absolute() and output.parent.is_dir()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    stream = os.fdopen(fd, 'w', buffering=1)
    original = module.get_cache_key
    observed = set()
    lock = Lock()

    @wraps(original)
    def traced(src, backend, backend_options, env_vars):
        actual = original(src, backend, backend_options, env_vars)
        if getattr(src, 'name', None) != KERNEL:
            return actual
        with lock:
            stamp = fingerprint(actual)
            if stamp not in observed:
                try:
                    record = key_record(actual, module, src, backend, backend_options, env_vars)
                except Exception as failure:
                    record = {'disk_metadata_hash': stamp, 'diagnostic_error': type(failure).__name__}
                stream.write(json.dumps({'pid': os.getpid(), **record}, sort_keys=True) + '\n')
                observed.add(stamp)
        return actual

    module.get_cache_key = traced
    return original
