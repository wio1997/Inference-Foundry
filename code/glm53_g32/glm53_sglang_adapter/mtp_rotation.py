# SPDX-License-Identifier: Apache-2.0
"""Validate the inherited NextN rotation load; preserve its actual symmetric matrix."""
import hashlib
import json
from pathlib import Path
import torch


def configure_nextn_rotation(model,quant_config=None,model_path=None,expected_file_sha256="d49a897c8c9a899b57890e3943f520dcb58f71490386c2895980634d31afc053"):
    if getattr(model,'_glm53_rotation_configured',False):
        return model._glm53_rotation_evidence
    if model_path is None:
        from sglang.srt.runtime_context import get_model
        model_path=get_model().model_path
    root=Path(model_path)
    description=json.loads((root/'quant_model_description.json').read_text())
    required=description.get('is_rot_used',False)
    if not isinstance(required,bool):raise TypeError('is_rot_used must be a boolean')
    config_description=getattr(quant_config,'quant_description',{}) or {}
    if 'is_rot_used' in config_description and config_description['is_rot_used']!=required:
        raise ValueError('ModelSlim config and checkpoint disagree on is_rot_used')
    evidence={'required':required,'scope':'NextN previous hidden before hnorm','source_key':'rot.weight',
              'native_contract':'previous_hidden @ model.rot_weight','active_contract':'native inherited previous_hidden @ exported_rot.weight'}
    if required:
        path=root/'rot.safetensors'
        if not path.is_file():raise FileNotFoundError('is_rot_used=true requires root rot.safetensors')
        raw=getattr(model.model,'rot_weight',None)
        if raw is None:
            from safetensors.torch import load_file
            tensors=load_file(str(path),device='cpu')
            if 'rot.weight' not in tensors:raise KeyError('rot.safetensors lacks rot.weight')
            raw=tensors['rot.weight']
        hidden=model.config.hidden_size
        if tuple(raw.shape)!=(hidden,hidden) or not torch.is_floating_point(raw):
            raise ValueError(f'Invalid exported rotation shape/dtype: {raw.shape} {raw.dtype}')
        device=model.model.hnorm.weight.device;dtype=model.model.hnorm.weight.dtype
        # The actual task matrix is bitwise symmetric on both hosts. Preserve
        # the native parent operand; no transpose or extra transform is added.
        model.model.rot_weight=raw.to(device=device,dtype=dtype)
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        if expected_file_sha256 is not None and digest.hexdigest()!=expected_file_sha256:
            raise ValueError('Rotation file changed from the parent-verified symmetric export; revalidate its contract')
        evidence.update(path=str(path),file_sha256=digest.hexdigest(),shape=list(raw.shape),dtype=str(raw.dtype),loaded=True,
                        native_direction_unchanged=True,actual_symmetry_evidence='hash_overlay_sanity_2203/2204.json' if expected_file_sha256 else 'synthetic test')
    else:
        # The inherited parent loads root rot unconditionally; honor metadata.
        model.model.rot_weight=None
        evidence['loaded']=False
    model._glm53_rotation_configured=True
    model._glm53_rotation_evidence=evidence
    return evidence
