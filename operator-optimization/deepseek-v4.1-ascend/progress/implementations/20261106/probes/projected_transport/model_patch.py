"""Render an isolated, source-pinned model overlay. Does not install or run it."""
import ast
from hashlib import sha256

PINS = {
    'model.py': '101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',
    'dsa_v41.py': '3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd',
    'dsa_v41_cp.py': '039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e',
}
ACCEPTED_PARENT_PINS = {
    'model.py': PINS['model.py'],
    'dsa_v41.py': '8998c8d51fcfbd2c993f47603f4fd62f35eb6dc277eddf3e1321a18224ff7e5d',
    'dsa_v41_cp.py': 'ef972657414294559c89c04e93ecb2be0965b5c149891dd32c8cd2f09ad69cf4',
}


def replace_once(source, before, after):
    assert source.count(before) == 1, before[:100]
    return source.replace(before, after)


def render(name, raw):
    assert sha256(raw).hexdigest() in (PINS[name], ACCEPTED_PARENT_PINS[name])
    source = raw.decode()
    if name == 'model.py':
        source = replace_once(source, 'import torch\n',
                              'import torch\nfrom model_owner import owner_entry as _wio_owner_entry\n')
        source = replace_once(source,
            '        if use_sequence_parallel:\n            x = sp_all_gather(x)[: positions.shape[0]]\n',
            '        owner_input = use_sequence_parallel and _wio_owner_entry(self.self_attn, positions, x)\n'
            '        if use_sequence_parallel and not owner_input:\n'
            '            x = sp_all_gather(x)[: positions.shape[0]]\n')
        source = replace_once(source,
            '        x = self.self_attn(positions, x, llama_4_scaling)\n',
            '        x = self.self_attn(positions, x, llama_4_scaling, owner_input=owner_input)\n')
        source = replace_once(source,
            '    def forward(self, positions, hidden_states, llama_4_scaling=None):\n        output = torch.empty_like(hidden_states)\n',
            '    def forward(self, positions, hidden_states, llama_4_scaling=None, *, owner_input=False):\n'
            '        output = (hidden_states.new_empty((positions.shape[0], hidden_states.shape[-1]))\n'
            '                  if owner_input else torch.empty_like(hidden_states))\n')
    elif name == 'dsa_v41.py':
        source = replace_once(source,
            '        prepared_indexer=None,\n    ):\n        compressor = attn.compressor\n',
            '        prepared_indexer=None,\n        projected_sources=None,\n    ):\n        compressor = attn.compressor\n')
        source = replace_once(source, '            latent = compressor(hidden_states)\n',
            '            latent = compressor(hidden_states) if projected_sources is None else projected_sources[0]\n')
        source = replace_once(source, '            hidden_states_fp32 = hidden_states.float()\n',
            '            hidden_states_fp32 = hidden_states.float() if projected_sources is None else None\n')
        source = replace_once(source,
            '            kv = compressor.wkv(hidden_states_fp32)\n            score = compressor.wgate(hidden_states_fp32)\n',
            '            if projected_sources is None:\n'
            '                kv = compressor.wkv(hidden_states_fp32)\n'
            '                score = compressor.wgate(hidden_states_fp32)\n'
            '            else:\n                kv, score = projected_sources\n')
    else:
        source = replace_once(source, 'import torch\n',
            'import torch\nfrom model_owner import eligible as _wio_owner_eligible, forward as _wio_owner_forward\n'
            'from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather as _wio_owner_fallback_gather\n')
        marker = 'class AscendDSAV41CPImpl(_ReplicatedCacheImpl):\n'
        source = replace_once(source, marker, marker + '''    def forward(self, attn, positions, hidden_states, output=None):
        if tuple(hidden_states.shape) == (3, 5120) and output is not None and tuple(output.shape) == (24, 5120):
            metadata = get_forward_context().attn_metadata
            if _wio_owner_eligible(self, attn, hidden_states, output, metadata):
                return _wio_owner_forward(self, attn, hidden_states, output)
            hidden_states = _wio_owner_fallback_gather(hidden_states)[:output.shape[0]]
        return super().forward(attn, positions, hidden_states, output)

''')
        source = replace_once(source,
            '    def _indexer_hidden_states(self, hidden_states, metadata):\n        start, _, _, _ = metadata.swa.cp_token_range\n',
            '    def _indexer_hidden_states(self, hidden_states, metadata):\n'
            '        if hidden_states.shape[0] == 3 and metadata.swa.global_metadata.num_actual_tokens == 24:\n'
            '            return hidden_states[:metadata.swa.num_actual_tokens]\n'
            '        start, _, _, _ = metadata.swa.cp_token_range\n')
        tree = ast.parse(source)
        cls, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AscendDSAV41CPImpl']
        method, = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_project_output']
        lines = source.splitlines(keepends=True)
        body = ''.join(lines[method.lineno-1:method.end_lineno])
        assert body.count('hidden_states.shape[0]') == 3
        source = replace_once(source, body, body.replace('hidden_states.shape[0]', 'projected.shape[0]'))
    ast.parse(source)
    return source.encode()
