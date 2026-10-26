#!/usr/bin/env python3
"""Reversible prefill-only shared-expert same-stream diagnostic."""
import argparse,hashlib,json
from pathlib import Path
SRC=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py')
BKP=Path('/tmp/loop052_run200_fused_moe_backup.py')
NEEDLE='''        if self._shared_experts is None:
            return None

        def maybe_wait_event(evt: torch.npu.Event | None):'''
REPLACE='''        if self._shared_experts is None:
            return None

        # EXTREME_LOOP052_RUN200: candidate applies only to eager prefill.
        import os as _extreme_os, json as _extreme_json
        _extreme_run200_overlap = self.multistream_overlap_shared_expert
        _extreme_run200_arm = _extreme_os.getenv("EXTREME_RUN200_ARM_FILE")
        if (
            _extreme_run200_overlap
            and _extreme_run200_arm
            and _extreme_os.path.exists(_extreme_run200_arm)
            and str(get_forward_context().cudagraph_runtime_mode) == "NONE"
        ):
            _extreme_run200_overlap = False
            _extreme_run200_markdir = _extreme_os.getenv("EXTREME_RUN200_MARK_DIR")
            if _extreme_run200_markdir:
                _extreme_os.makedirs(_extreme_run200_markdir, exist_ok=True)
                _extreme_run200_rank = int(get_tp_group().rank_in_group)
                _extreme_run200_mark = _extreme_os.path.join(_extreme_run200_markdir, f"rank{_extreme_run200_rank}.json")
                if not _extreme_os.path.exists(_extreme_run200_mark):
                    with open(_extreme_run200_mark, "w", encoding="utf-8") as _extreme_run200_f:
                        _extreme_json.dump({"rank": _extreme_run200_rank, "mode": "NONE", "pid": _extreme_os.getpid()}, _extreme_run200_f)

        def maybe_wait_event(evt: torch.npu.Event | None):'''
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['install','restore']);p.add_argument('--record',required=True);a=p.parse_args()
 if a.action=='install':
  assert not BKP.exists()
  before=SRC.read_bytes();s=before.decode();assert s.count(NEEDLE)==1
  after=s.replace(NEEDLE,REPLACE,1).replace('enabled=self.multistream_overlap_shared_expert):','enabled=_extreme_run200_overlap):',1)
  old='''        if self.multistream_overlap_shared_expert:
            torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())'''
  new='''        if _extreme_run200_overlap:
            torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())'''
  assert after.count(old)==1;after=after.replace(old,new,1)
  compile(after,str(SRC),'exec');BKP.write_bytes(before);SRC.write_text(after)
  result={'action':'install','original_sha256':sha(before),'patched_sha256':sha(SRC.read_bytes())}
 else:
  assert BKP.exists();before=SRC.read_bytes();assert b'EXTREME_LOOP052_RUN200' in before
  SRC.write_bytes(BKP.read_bytes());BKP.unlink()
  result={'action':'restore','patched_sha256':sha(before),'restored_sha256':sha(SRC.read_bytes())}
 out=Path(a.record);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
