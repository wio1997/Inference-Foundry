#!/usr/bin/env python3
"""Reversible DSpark-only Target MTP stash experiment in borrowed framework."""
import argparse, hashlib, json
from pathlib import Path
SOURCE=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/models/deepseek_v4.py')
BASE_SHA='11dd3e983dc1d628bf42a95581f9617861471b1f658739db44c20df886147247'
OLD='''        forward_ctx = get_forward_context()
        if forward_ctx is not None and forward_ctx.flash_comm_v1_enabled:
            h_states_flat = tensor_model_parallel_all_gather(hidden_states.flatten(1), dim=0)
            pad_size = forward_ctx.pad_size
            if pad_size > 0:
                h_states_flat = h_states_flat[:-pad_size]
            num_tokens = h_states_flat.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(h_states_flat)
        else:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
'''
NEW='''        # Extreme experiment: DSpark consumes its own proposer states and never
        # reads the MTP residual buffer. Keep original behavior for all other
        # methods, including MTP. This flag is set only in the fixed E2E run.
        if not self._extreme_skip_mtp_stash:
            forward_ctx = get_forward_context()
            if forward_ctx is not None and forward_ctx.flash_comm_v1_enabled:
                h_states_flat = tensor_model_parallel_all_gather(hidden_states.flatten(1), dim=0)
                pad_size = forward_ctx.pad_size
                if pad_size > 0:
                    h_states_flat = h_states_flat[:-pad_size]
                num_tokens = h_states_flat.shape[0]
                self._mtp_hidden_buffer[:num_tokens].copy_(h_states_flat)
            else:
                num_tokens = hidden_states.shape[0]
                self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
'''
# The module's config is not otherwise retained; use an explicit constructor field.
ANCHOR='''        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
'''
REPLACEMENT='''        self._extreme_skip_mtp_stash = (
            __import__("os").getenv("EXTREME_DSPARK_SKIP_MTP_STASH") == "1"
            and getattr(vllm_config.speculative_config, "method", None) == "dspark"
        )
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
'''

def sha(data): return hashlib.sha256(data).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['install','restore']);p.add_argument('--record',required=True);a=p.parse_args()
 record=Path(a.record);record.parent.mkdir(parents=True,exist_ok=True)
 raw=SOURCE.read_bytes()
 if a.action=='install':
  if sha(raw)!=BASE_SHA: raise SystemExit('borrowed source SHA changed; refusing install')
  txt=raw.decode()
  if txt.count(OLD)!=1 or txt.count(ANCHOR)!=1: raise SystemExit('patch anchors not unique')
  backup=record.with_suffix('.original.py');backup.write_bytes(raw)
  new=txt.replace(ANCHOR,REPLACEMENT).replace(OLD,NEW)
  compile(new,str(SOURCE),'exec')
  SOURCE.write_text(new)
  record.write_text(json.dumps({'source':str(SOURCE),'backup':str(backup),'base_sha':sha(raw),'patched_sha':sha(new.encode())},indent=2))
  print(record.read_text())
 else:
  info=json.loads(record.read_text());original=Path(info['backup']).read_bytes()
  if sha(original)!=info['base_sha'] or sha(original)!=BASE_SHA: raise SystemExit('backup mismatch')
  if sha(raw)!=info['patched_sha']:
   if sha(raw)==BASE_SHA: print('already restored');return
   raise SystemExit('patched source changed; refusing overwrite')
  SOURCE.write_bytes(original);print('restored',sha(SOURCE.read_bytes()))
if __name__=='__main__':main()
