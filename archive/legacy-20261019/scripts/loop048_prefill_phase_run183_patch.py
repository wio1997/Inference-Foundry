#!/usr/bin/env python3
"""Temporarily instrument unprofiled prefill custom-op host walls; exact restore."""
import argparse,hashlib,json
from pathlib import Path
BASE=Path('/data/wio/vllm_ascend_26/framework')
FILES={
 'dsa':BASE/'vllm-ascend/vllm_ascend/ops/dsa.py',
 'moe':BASE/'vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py',
 'runner':BASE/'vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
}
BKP=Path('/tmp/loop048_run183_patch_backup')
DSA=r'''
# EXTREME_LOOP048_RUN183_DSA
_extreme_run183_dsa_orig = dsa_forward
def dsa_forward(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    import builtins as _bi, time as _time
    _state = getattr(_bi, '_extreme_run183_phase', None)
    if _state is None:
        return _extreme_run183_dsa_orig(hidden_states, need_gather_q_kv, output, layer_name)
    _start = _time.perf_counter_ns()
    _out = _extreme_run183_dsa_orig(hidden_states, need_gather_q_kv, output, layer_name)
    _state['dsa_ns'].append(_time.perf_counter_ns() - _start)
    return _out

'''
MOE=r'''
# EXTREME_LOOP048_RUN183_MOE
_extreme_run183_moe_orig = _moe_forward_shared
def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import builtins as _bi, time as _time
    _state = getattr(_bi, '_extreme_run183_phase', None)
    if _state is None:
        return _extreme_run183_moe_orig(hidden_states, router_logits, shared_experts_input, input_ids, layer_name, hidden_dim_unpadded)
    _start = _time.perf_counter_ns()
    _out = _extreme_run183_moe_orig(hidden_states, router_logits, shared_experts_input, input_ids, layer_name, hidden_dim_unpadded)
    _state['moe_ns'].append(_time.perf_counter_ns() - _start)
    return _out

'''
RUNNER=r'''

# EXTREME_LOOP048_RUN183_FORWARD
if os.getenv('EXTREME_RUN183_PHASE_DIR'):
    _extreme_run183_forward_orig = NPUModelRunner._model_forward
    def _extreme_run183_forward(self, num_tokens_padded, *args, **kwargs):
        import builtins as _bi
        _arm = os.getenv('EXTREME_RUN183_ARM_FILE')
        if not _arm or not os.path.exists(_arm):
            return _extreme_run183_forward_orig(self, num_tokens_padded, *args, **kwargs)
        _state = {'dsa_ns':[], 'moe_ns':[]}
        _bi._extreme_run183_phase = _state
        _start = time.perf_counter_ns()
        try:
            _out = _extreme_run183_forward_orig(self, num_tokens_padded, *args, **kwargs)
        finally:
            _end = time.perf_counter_ns()
            _bi._extreme_run183_phase = None
        if _state['dsa_ns'] or _state['moe_ns']:
            _rank = int(get_tp_group().rank_in_group)
            _context = get_forward_context()
            _mode = str(getattr(_context, 'cudagraph_runtime_mode', 'unknown'))
            _actual = int(getattr(_context, 'num_actual_tokens', num_tokens_padded) or num_tokens_padded)
            _row = {'rank':_rank,'num_tokens_padded':int(num_tokens_padded),
                    'num_actual_tokens':_actual,'num_reqs':int(getattr(self.input_batch,'num_reqs',-1)),
                    'mode':_mode,'forward_wall_ms':(_end-_start)/1e6,
                    'dsa_count':len(_state['dsa_ns']),'moe_count':len(_state['moe_ns']),
                    'dsa_wall_ms':sum(_state['dsa_ns'])/1e6,'moe_wall_ms':sum(_state['moe_ns'])/1e6}
            _path = os.path.join(os.environ['EXTREME_RUN183_PHASE_DIR'],f'rank{_rank}.jsonl')
            os.makedirs(os.path.dirname(_path),exist_ok=True)
            with open(_path,'a',encoding='utf-8') as _f:
                _f.write(json.dumps(_row,separators=(',',':'))+'\n')
        return _out
    NPUModelRunner._model_forward = _extreme_run183_forward
'''
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['install','restore']);p.add_argument('--record',required=True);a=p.parse_args()
 result={'action':a.action,'files':{}}
 if a.action=='install':
  assert not BKP.exists();BKP.mkdir()
  edits={'dsa':('direct_register_custom_op(\n    op_name="dsa_forward",',DSA),'moe':('direct_register_custom_op(\n    op_name="moe_forward_shared",',MOE)}
  for k,path in FILES.items():
   before=path.read_bytes();assert b'EXTREME_LOOP048_RUN183' not in before
   (BKP/k).write_bytes(before)
   if k in edits:
    needle,insert=edits[k];s=before.decode();assert s.count(needle)==1
    after=s.replace(needle,insert+needle,1).encode()
   else:after=before+RUNNER.encode()
   compile(after,str(path),'exec');path.write_bytes(after)
   result['files'][k]={'original_sha256':sha(before),'patched_sha256':sha(after)}
 else:
  assert BKP.exists()
  for k,path in FILES.items():
   before=path.read_bytes();original=(BKP/k).read_bytes()
   assert b'EXTREME_LOOP048_RUN183' in before
   path.write_bytes(original)
   result['files'][k]={'patched_sha256':sha(before),'restored_sha256':sha(path.read_bytes())}
  for f in BKP.iterdir():f.unlink()
  BKP.rmdir()
 out=Path(a.record);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
