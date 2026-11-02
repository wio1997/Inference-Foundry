# SPDX-License-Identifier: Apache-2.0
"""Device-free native dependency declarations and expert logical-byte budgets.

No import of torch/native modules, device discovery, weight read or allocation.
A matching declaration is NOT operator availability, ABI correctness, execution
validation, authorization or full-model readiness. Physical NZ/padding, allocator
copies and scratch remain unknown: logical bytes are not an HBM peak estimate.
"""
from dataclasses import asdict,dataclass
import importlib.metadata
import platform


# This profile is bounded to the historical KDA tuple, not a DSA/engine ABI.
KDA_PROFILE=(('torch','2.10.0+cpu'),('torch-npu','2.10.0.post7'),('flash-linear-attention-npu-a3','26.9.1'))
KDA_W8A8_BINDINGS=frozenset((
    'fla_npu.ops.ascendc.causal_conv1d_fn','fla_npu.ops.ascendc.recurrent_kda',
    'torch_npu.npu_dynamic_quant','torch_npu.npu_quant_matmul',
    'torch_npu.npu_grouped_matmul','torch_npu.npu_format_cast'))


@dataclass(frozen=True)
class NativeDependencyDeclaration:
    architecture: str
    packages: tuple[tuple[str,str | None],...]
    bindings: frozenset[str] = frozenset()

    @classmethod
    def from_metadata(cls):
        """Read distribution metadata only; do not infer exported operators."""
        values=[]
        for name,_ in KDA_PROFILE:
            try:version=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:version=None
            values.append((name,version))
        return cls(platform.machine(),tuple(values))

    def report(self):
        if not isinstance(self.packages,tuple) or any(not isinstance(p,tuple) or len(p)!=2 for p in self.packages):
            raise ValueError('package declaration must be an immutable sequence of name/version pairs')
        observed=dict(self.packages)
        if len(observed)!=len(self.packages) or set(observed)!=dict(KDA_PROFILE).keys():
            raise ValueError('package declarations must cover the exact profile without duplicate names')
        if not isinstance(self.bindings,frozenset) or any(not isinstance(n,str) for n in self.bindings):
            raise ValueError('operator binding names must be an immutable explicit set')
        reasons=[]
        if self.architecture!='aarch64':reasons.append('host architecture does not match aarch64 profile')
        for name,wanted in KDA_PROFILE:
            if observed[name]!=wanted:reasons.append(f'{name}: expected {wanted}, observed {observed[name]}')
        missing=sorted(KDA_W8A8_BINDINGS-self.bindings)
        if missing:reasons.append('missing explicit KDA/W8A8 binding declarations: '+', '.join(missing))
        return {'scope':'KDA_W8A8_declarations_only','declaration_matches':not reasons,'reasons':reasons,
                'packages':observed,'architecture':self.architecture,'missing_bindings':missing,
                'native_execution_validated':False,'model_ready':False}

    def require_matching_declaration(self):
        result=self.report()
        if not result['declaration_matches']:raise ValueError('; '.join(result['reasons']))
        return result


@dataclass(frozen=True)
class ExpertResidencyPlan:
    routed_experts: int
    includes_shared: bool
    checkpoint_payload_bytes: int
    resident_logical_bytes: int
    largest_weight_bytes: int

    def to_dict(self):
        result=asdict(self)
        result.update(scope='selected_INT8_experts_only',physical_peak_bytes=None,model_ready=False)
        return result


def plan_expert_residency(manifest,*,max_checkpoint_bytes,max_resident_bytes):
    """Budget all selected INT8 weights/F32 sidecars and resident BF16 scales.

    Existing expert_manifest validates quantization and channel metadata first.
    This function checks geometry/IDs as well, before native payload reads.
    Counts are sums of named logical tensors, including the shared expert once;
    transient duplicates/NZ storage/other layers are explicitly excluded.
    Limits never truncate expert IDs or silently drop shared output.
    """
    for value in (max_checkpoint_bytes,max_resident_bytes):
        if type(value) is not int or value<1:raise ValueError('positive explicit expert byte budgets required')
    if type(manifest.num_experts) is not int or manifest.num_experts<1 or type(manifest.hidden_size) is not int or manifest.hidden_size<1:
        raise ValueError('invalid expert namespace/hidden geometry')
    ids=[s.global_id for s in manifest.routed]
    if any(type(i) is not int or not 0<=i<manifest.num_experts for i in ids) or ids!=sorted(set(ids)):
        raise ValueError('routed expert IDs must be unique, ascending and in range')
    if manifest.shared is not None and manifest.shared.global_id is not None:raise ValueError('shared expert must have no global routed ID')
    checkpoint=resident=largest=0;seen=set();intermediate=None
    for spec in (*manifest.routed,*(() if manifest.shared is None else (manifest.shared,))):
        if len(spec.shapes)!=3 or len(spec.prefixes)!=3:raise ValueError('gate/up/down expert matrices required')
        for shape in spec.shapes:
            if len(shape)!=2 or any(type(d) is not int or d<1 for d in shape):raise ValueError('positive matrix dimensions required')
        gate,up,down=spec.shapes
        if gate!=up or gate[1]!=manifest.hidden_size or down!=(manifest.hidden_size,gate[0]):raise ValueError('expert gate/up/down geometry mismatch')
        if intermediate is None:intermediate=gate[0]
        if intermediate!=gate[0]:raise ValueError('selected expert bank requires identical intermediate sizes')
        for name,(rows,cols) in zip(spec.prefixes,spec.shapes,strict=True):
            if not isinstance(name,str) or not name or name in seen:raise ValueError('unique explicit matrix prefixes required')
            seen.add(name);weight=rows*cols;largest=max(largest,weight)
            checkpoint+=weight+rows*8  # I8 weight, F32 weight_scale and offset.
            resident+=weight+rows*2  # I8 weight, rounded BF16 channel scale.
    if checkpoint>max_checkpoint_bytes:raise ValueError(f'checkpoint expert payload {checkpoint} exceeds budget {max_checkpoint_bytes}')
    if resident>max_resident_bytes:raise ValueError(f'logical resident expert bytes {resident} exceeds budget {max_resident_bytes}')
    return ExpertResidencyPlan(len(ids),manifest.shared is not None,checkpoint,resident,largest)
