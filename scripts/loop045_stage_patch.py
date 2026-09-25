#!/usr/bin/env python3
"""Temporary stage marks inside borrowed NPUModelRunner.execute_model."""
import argparse,hashlib,json
from pathlib import Path
MODEL=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
BACKUP=Path('/tmp/extreme_loop045_run160_model_runner_v1.py.boundary')
MARKER='# EXTREME_LOOP045_RUN160_STAGE'
def sha(b):return hashlib.sha256(b).hexdigest()
def once(s,old,new):
 assert s.count(old)==1,(old[:100],s.count(old))
 return s.replace(old,new,1)
def install():
 assert not BACKUP.exists() and MARKER not in MODEL.read_text()
 original=MODEL.read_bytes();s=original.decode()
 s=once(s,'        with record_function_or_nullcontext("prepare input"):',
        '        # EXTREME_LOOP045_RUN160_STAGE\n        self._extreme_stage_marks = [("prep_start", time.perf_counter_ns())]\n        with record_function_or_nullcontext("prepare input"):')
 s=once(s,'            update_cos_sin(positions)\n\n        if self.dynamic_eplb:',
        '            update_cos_sin(positions)\n\n        self._extreme_stage_marks.append(("prep_end", time.perf_counter_ns()))\n        if self.dynamic_eplb:')
 s=once(s,'        with (\n            record_function_or_nullcontext("forward"),',
        '        self._extreme_stage_marks.append(("forward_context_start", time.perf_counter_ns()))\n        with (\n            record_function_or_nullcontext("forward"),')
 s=once(s,'            hidden_states = self._model_forward(\n                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs\n            )\n            if _extreme_runtime_inputs is not None:',
        '            self._extreme_stage_marks.append(("model_forward_start", time.perf_counter_ns()))\n            hidden_states = self._model_forward(\n                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs\n            )\n            self._extreme_stage_marks.append(("model_forward_end", time.perf_counter_ns()))\n            if _extreme_runtime_inputs is not None:')
 s=once(s,'        with record_function_or_nullcontext("post process"):',
        '        self._extreme_stage_marks.append(("post_start", time.perf_counter_ns()))\n        with record_function_or_nullcontext("post process"):')
 BACKUP.write_bytes(original);MODEL.write_text(s)
 return {'action':'install','original_sha256':sha(original),'patched_sha256':sha(MODEL.read_bytes())}
def restore():
 assert BACKUP.exists() and MARKER in MODEL.read_text()
 original=BACKUP.read_bytes();MODEL.write_bytes(original);BACKUP.unlink()
 return {'action':'restore','original_sha256':sha(original),'restored_sha256':sha(MODEL.read_bytes())}
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['install','restore']);p.add_argument('--record',required=True);a=p.parse_args()
 result=install() if a.action=='install' else restore()
 out=Path(a.record);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
