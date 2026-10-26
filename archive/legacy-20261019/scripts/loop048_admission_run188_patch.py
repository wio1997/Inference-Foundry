#!/usr/bin/env python3
"""Borrowed Core <=2s admission gate and model forward census; exact restore."""
import argparse,hashlib,json
from pathlib import Path
FILES={'core':Path('/data/wio/vllm_ascend_26/framework/vllm/vllm/v1/engine/core.py'),
       'runner':Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')}
BKP=Path('/tmp/loop048_run188_patch_backup')
CORE_NEEDLE='''            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()
'''
CORE_INSERT='''            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # EXTREME_LOOP048_RUN188_ADMISSION: opt-in bounded c12 first-cohort drain.
            _wait_file = os.getenv("EXTREME_RUN188_WAIT_FILE")
            if (_wait_file and os.path.exists(_wait_file)
                    and not self.scheduler.running
                    and 0 < sum(self.scheduler.get_request_counts()) < 12):
                _start = time.monotonic()
                _before = self.scheduler.get_request_counts()
                while (time.monotonic() - _start < 2.0
                       and sum(self.scheduler.get_request_counts()) < 12
                       and self.is_running()):
                    time.sleep(0.005)
                    self._process_input_queue()
                _row = {"t":_start,"wait_s":time.monotonic()-_start,
                        "counts_before":_before,"counts_after":self.scheduler.get_request_counts()}
                with open(os.environ["EXTREME_RUN188_WAIT_TRACE"],"a",encoding="utf-8") as _f:
                    _f.write(json.dumps(_row,separators=(",",":"))+"\\n")
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()
'''
RUNNER=r'''

# EXTREME_LOOP048_RUN188_FORWARD
if os.getenv('EXTREME_RUN188_FORWARD_DIR'):
    _extreme_run188_forward_orig = NPUModelRunner._model_forward
    def _extreme_run188_forward(self, num_tokens_padded, *args, **kwargs):
        _tag_file = os.getenv('EXTREME_RUN188_TAG_FILE')
        try:
            with open(_tag_file,'r',encoding='utf-8') as _f:
                _tag = _f.read().strip()
        except (OSError,TypeError):
            return _extreme_run188_forward_orig(self,num_tokens_padded,*args,**kwargs)
        if _tag not in ('A','B','A2'):
            return _extreme_run188_forward_orig(self,num_tokens_padded,*args,**kwargs)
        _start = time.perf_counter_ns()
        _out = _extreme_run188_forward_orig(self,num_tokens_padded,*args,**kwargs)
        _end = time.perf_counter_ns()
        _context = get_forward_context()
        _rank = int(get_tp_group().rank_in_group)
        _row = {'tag':_tag,'rank':_rank,'num_tokens_padded':int(num_tokens_padded),
                'num_actual_tokens':int(getattr(_context,'num_actual_tokens',num_tokens_padded) or num_tokens_padded),
                'num_reqs':int(getattr(self.input_batch,'num_reqs',-1)),
                'mode':str(getattr(_context,'cudagraph_runtime_mode','unknown')),
                'forward_wall_ms':(_end-_start)/1e6}
        _path = os.path.join(os.environ['EXTREME_RUN188_FORWARD_DIR'],f'rank{_rank}_{_tag}.jsonl')
        os.makedirs(os.path.dirname(_path),exist_ok=True)
        with open(_path,'a',encoding='utf-8') as _f:
            _f.write(json.dumps(_row,separators=(',',':'))+'\n')
        return _out
    NPUModelRunner._model_forward = _extreme_run188_forward
'''
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['install','restore']);p.add_argument('--record',required=True);a=p.parse_args()
 result={'action':a.action,'files':{}}
 if a.action=='install':
  assert not BKP.exists();BKP.mkdir()
  for k,path in FILES.items():
   before=path.read_bytes();assert b'EXTREME_LOOP048_RUN188' not in before
   (BKP/k).write_bytes(before)
   if k=='core':
    s=before.decode().replace('import gc\n','import gc\nimport json  # EXTREME_LOOP048_RUN188_ADMISSION\n',1)
    assert s.count(CORE_NEEDLE)==1
    after=s.replace(CORE_NEEDLE,CORE_INSERT,1).encode()
   else:after=before+RUNNER.encode()
   compile(after,str(path),'exec');path.write_bytes(after)
   result['files'][k]={'original_sha256':sha(before),'patched_sha256':sha(after)}
 else:
  assert BKP.exists()
  for k,path in FILES.items():
   before=path.read_bytes();original=(BKP/k).read_bytes();assert b'EXTREME_LOOP048_RUN188' in before
   path.write_bytes(original);result['files'][k]={'patched_sha256':sha(before),'restored_sha256':sha(path.read_bytes())}
  for f in BKP.iterdir():f.unlink()
  BKP.rmdir()
 out=Path(a.record);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
