import hashlib,json
from pathlib import Path
r=Path('/data/wio/Inference_Foundry')
base=(r/'evidence/20260926_loop062_nongmm/run259/patch.original.py').read_text()
patched=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/models/deepseek_v4.py').read_text()
runner=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py').read_text()
target=(r/'runtime/target_adapter.py').read_text()
handoff=(r/'bootstrap/vllm_target_handoff.py').read_text()
ds=(r/'bootstrap/vllm_dspark_handoff.py').read_text()
checks={'base_stash_writes':base.count('self._mtp_hidden_buffer[:num_tokens].copy_(')==2,'patch_constructor_guard':'self._extreme_skip_mtp_stash = (' in patched and 'method", None) == "dspark"' in patched,'patch_forward_guard':'if not self._extreme_skip_mtp_stash:' in patched,'runner_only_mtp_rebind':'if self.speculative_config.method == "mtp" and mtp_hidden_states is not None:' in runner,'direct_target_no_buffer':'_mtp_hidden_buffer' not in target and '_mtp_hidden_buffer' not in handoff,'direct_dspark_no_buffer':'_mtp_hidden_buffer' not in ds,'runtime_snapshot_only_diagnostic':'EXTREME_TARGET_SELF_PARITY' in runner}
result={'status':'pass' if all(checks.values()) else 'fail','checks':checks,'base_sha256':hashlib.sha256(base.encode()).hexdigest(),'limit':'Static source consumer audit plus Run260 FULL Graph product gates; not a same-state tensor differential against base. MTP method remains ungated and unchanged.'}
print(json.dumps(result,indent=2))
