#!/usr/bin/env python3
"""Source-only optional fixed metadata Graph candidate, exact-SHA pinned."""
from __future__ import annotations
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
OUT=ROOT/'evidence/20260929_loop081_bound/run662'
SOURCES={
 'metadata':(ROOT/'runtime/target_metadata.py','940a9861551455e64d8163b4c09fec91ca87ac7880ff8f67adecf8435d91a16a'),
 'runner':(Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'),'004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba'),
}

def sha(raw):return hashlib.sha256(raw).hexdigest()
def one(src,old,new):
    if src.count(old)!=1:raise RuntimeError(f'anchor count {src.count(old)}: {old[:70]}')
    return src.replace(old,new)

def metadata(src):
    src=one(src,
'''        self._request_index = torch.arange(12, device=self._device).unsqueeze(1)
''',
'''        self._request_index = torch.arange(12, device=self._device).unsqueeze(1)
        self._graph_enabled = os.getenv("EXTREME_TARGET_METADATA_GRAPH") == "1"
        self._graph = None
        self._graph_input_ptrs = None
        self._graph_replays = 0
''')
    src=one(src,
'''    @torch.inference_mode()
    def update(self, state: FixedDecodeState) -> None:
''',
'''    @torch.inference_mode()
    def capture_graph(self, state: FixedDecodeState) -> None:
        """Capture only in setup, before the timed fixed serving loop."""
        if not self._graph_enabled or self._graph is not None:
            raise RuntimeError("metadata Graph capture requires fresh enabled updater")
        if not self._static_kv_max or self._shadow_dynamic:
            raise RuntimeError("metadata Graph requires fixed KV maxima and no shadow")
        self._update_eager(state)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            self._update_eager(state)
        torch.npu.synchronize()
        self._graph_input_ptrs = (
            state.target_positions.data_ptr(), state.target_seq_lens.data_ptr())
        self._graph = graph

    @torch.inference_mode()
    def update(self, state: FixedDecodeState) -> None:
        if not self._graph_enabled:
            self._update_eager(state)
            return
        if self._graph is None or self._graph_input_ptrs != (
                state.target_positions.data_ptr(), state.target_seq_lens.data_ptr()):
            raise RuntimeError("metadata Graph missing or input address drift")
        self._graph.replay()
        self._graph_replays += 1

    @torch.inference_mode()
    def _update_eager(self, state: FixedDecodeState) -> None:
''')
    return src

def runner(src):
    src=one(src,
'''            _extreme_runtime = build_extreme_runtime(
                _extreme_runtime_inputs,
                _cfg,
            )
            if os.getenv("EXTREME_STOCK_TARGET_ABA") == "1":''',
'''            _extreme_runtime = build_extreme_runtime(
                _extreme_runtime_inputs,
                _cfg,
            )
            if os.getenv("EXTREME_TARGET_METADATA_GRAPH") == "1":
                if (not _extreme_serve or _extreme_runtime.target_metadata is None
                        or os.getenv("EXTREME_SCHEDULE_NEXT_TARGET_METADATA", "off") != "off"):
                    raise RuntimeError("metadata Graph requires fixed serving and schedule off")
                _extreme_runtime._state_machine.prepare_target_inputs()
                _extreme_runtime.target_metadata.capture_graph(_extreme_runtime.state)
            if os.getenv("EXTREME_STOCK_TARGET_ABA") == "1":''')
    src=one(src,
'''                    "target_graph_mode": str(
                        _target_inputs.aclgraph_runtime_mode
                    ),
                    "wall_seconds": _extreme_wall_seconds,''',
'''                    "target_graph_mode": str(
                        _target_inputs.aclgraph_runtime_mode
                    ),
                    "metadata_graph_capture": bool(
                        _extreme_runtime.target_metadata is not None and
                        _extreme_runtime.target_metadata._graph is not None
                    ),
                    "metadata_graph_replays": (
                        _extreme_runtime.target_metadata._graph_replays
                        if _extreme_runtime.target_metadata is not None else 0
                    ),
                    "wall_seconds": _extreme_wall_seconds,''')
    return src

PATCH={'metadata':metadata,'runner':runner}
def main():
    cand=OUT/'candidate';diff=OUT/'diff'
    cand.mkdir(parents=True,exist_ok=False);diff.mkdir(parents=True,exist_ok=False)
    manifest={'status':'source_only_not_installed','sources':{},'helpers':{}}
    for key,(path,expected) in SOURCES.items():
        raw=path.read_bytes()
        if sha(raw)!=expected:raise RuntimeError(f'source drift {key}')
        before=raw.decode();after=PATCH[key](before)
        ast.parse(after)
        (cand/f'{key}.py').write_text(after)
        d=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                                        fromfile=str(path),tofile=str(path)+'.run662'))
        (diff/f'{key}.diff').write_text(d)
        manifest['sources'][key]={'path':str(path),'before_sha256':expected,
                                  'candidate_sha256':sha(after.encode()),'diff_sha256':sha(d.encode())}
    (OUT/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'files':len(SOURCES)}))
if __name__=='__main__':main()
