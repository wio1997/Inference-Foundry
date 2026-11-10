# SPDX-License-Identifier: Apache-2.0
"""Construct the installed complete reference model, without a request engine.

The only scheduler, sampler and request owner belong to DeepInfer. Configuration
creation reuses EngineArgs solely to run the pinned platform/model/cache config
normalization. get_model runs the real checkpoint loader and quant postload;
the fixed Ascend prefetch offloader is installed before model construction.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

import torch
import torch.distributed as dist

from .reference_tp import bind_worker_tensor_parallel


CHECKPOINT_PINS = {
    'config.json': '33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943',
    'quant_model_weights.safetensors.index.json': 'dfaf0918f4c088c98b011def38475d9b1cb395f35ed5e279abc0c57610e23be4',
}


def verify_installed_reference_sources():
    """Require the exact exported 107 installation, without private path pins."""
    import vllm
    import vllm_ascend
    roots = {'vllm': Path(vllm.__file__).resolve().parent,
             'vllm_ascend': Path(vllm_ascend.__file__).resolve().parent}
    pins = json.loads(Path(__file__).with_name('reference-runtime-pins.json').read_text())
    if platform.machine() != pins['host_architecture']:
        raise ValueError('Host Architecture differs from the audited source installation')
    for package, version in pins['packages'].items():
        if version.startswith('not installed'):
            continue
        if importlib.metadata.version(package) != version:
            raise ValueError('installed dependency version differs from audited snapshot: ' + package)
    for relative, expected in pins['files'].items():
        package, path = relative.split('/', 1)
        actual = roots[package] / path
        if not actual.is_file() or hashlib.sha256(actual.read_bytes()).hexdigest() != expected:
            raise ValueError('installed reference source differs from audited snapshot: ' + relative)


@dataclass(frozen=True)
class LoadedReferenceModel:
    # Keep the root alive: it owns the checkpoint mapper and multimodal wrapper.
    # Only its complete causal language model is called for text computation.
    root_model: object
    model: object
    vllm_config: object
    offloader: object
    device: torch.device

    @contextmanager
    def step_scope(self):
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.offloader.base import get_offloader
        if get_offloader() is not self.offloader:
            raise RuntimeError('reference offloader singleton changed after model construction')
        with set_current_vllm_config(self.vllm_config):
            self.offloader.sync_prev_onload()
            try:
                yield
            finally:
                self.offloader.join_after_forward()


def create_reference_config(model_dir, *, tensor_parallel_size,
                            max_context_tokens, max_step_tokens):
    """Use round599's fixed eager W8A8/prefetch route, no engine allocation."""
    if any(type(n) is not int or n < 1 for n in
           (tensor_parallel_size, max_context_tokens, max_step_tokens)):
        raise ValueError('explicit positive TP/context/step bounds required')
    if max_step_tokens > max_context_tokens:
        raise ValueError('step bound exceeds context bound')
    root = Path(model_dir).resolve(strict=True)
    for name, expected in CHECKPOINT_PINS.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('checkpoint identity differs from pinned reference: ' + name)
    # Must be selected before any offloader is created. Task launcher owns all
    # cache/output directory environment settings; do not write to home here.
    os.environ['VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY'] = '1'
    from vllm.engine.arg_utils import EngineArgs
    args = EngineArgs(
        model=str(root), tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1, data_parallel_size=1,
        distributed_executor_backend='external_launcher', enforce_eager=True,
        offload_backend='prefetch', offload_group_size=2,
        offload_num_in_group=1, offload_prefetch_step=1,
        offload_params={'w13_weight', 'w2_weight'},
        max_model_len=max_context_tokens, max_num_seqs=1,
        max_num_batched_tokens=max_context_tokens,
        enable_chunked_prefill=False, enable_prefix_caching=False,
        block_size=128, mamba_block_size=128,
        mamba_cache_mode='none', additional_config={'enable_dsa_cp': False},
        trust_remote_code=False, disable_log_stats=True)
    config = args.create_engine_config()
    if config.quant_config is None or config.model_config.dtype != torch.bfloat16:
        raise ValueError('pinned reference requires real checkpoint quantization and BF16 activations')
    if config.compilation_config.static_forward_context:
        raise ValueError('new configuration already owns registered model layers')
    return config


@torch.inference_mode()
def load_reference_model(model_dir, device, *, worker_group, local_rank,
                         max_context_tokens, max_step_tokens):
    """All Worker ranks call once; initialization failures are process-fatal.

    The Worker has selected/claimed its NPU and initialized distributed WORLD.
    Ascend platform/distributed initialization is a separate explicit startup
    prerequisite, checked by the final runtime assembly rather than inferred
    from a successful import. No model forward/warmup is performed here.
    """
    device = torch.device(device)
    if device.type != 'npu' or not dist.is_initialized():
        raise ValueError('prepared NPU Worker and initialized distributed world required')
    verify_installed_reference_sources()
    config = create_reference_config(
        model_dir, tensor_parallel_size=dist.get_world_size(worker_group),
        max_context_tokens=max_context_tokens, max_step_tokens=max_step_tokens)
    from vllm.config import set_current_vllm_config
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
    # Match the installed Ascend Worker's model startup without constructing
    # its scheduler: MLA/SFA constructor compatibility and OOT registrations
    # are part of the compute backend contract, not supplied by get_model.
    from vllm_ascend.utils import adapt_patch, register_ascend_customop, enable_custom_op
    from vllm_ascend import ops
    from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
    adapt_patch()
    ops.register_dummy_fusion_op()
    if get_current_hardware_profile().supports(HardwareCapability.ATB_EXTENSIONS):
        from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
        _register_atb_extensions()
    register_ascend_customop(config)
    init_ascend_config(config)
    # AscendWorker.init_device lazily loads the C++ torch library after
    # selecting its device. Registering Python CustomOp classes is separate.
    if not enable_custom_op():
        raise RuntimeError('installed Ascend custom-op extension could not be enabled')
    for op in ('npu_hc_pre_v2', 'npu_hc_post'):
        if not hasattr(torch.ops._C_ascend, op):
            raise RuntimeError('required installed GLM compute op is unregistered: ' + op)
    bind_worker_tensor_parallel(config, worker_group=worker_group, local_rank=local_rank)
    with set_current_vllm_config(config):
        init_ascend_model_parallel(config.parallel_config, config.engram_config)
    from vllm_ascend.distributed.parallel_state import get_mc2_group
    if list(get_mc2_group().ranks) != dist.get_process_group_ranks(worker_group):
        raise ValueError('Ascend MC2 group differs from Worker TP membership/order')
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.offloader.base import set_offloader
    from vllm_ascend.model_executor.offloader import create_offloader
    from vllm_ascend.model_executor.offloader.prefetch import AscendPrefetchOffloader
    from vllm_ascend.models.glm5next.model import Glm5NextForCausalLM
    from vllm_ascend.worker.model_runner_v1 import _torch_cuda_wrapper

    with _torch_cuda_wrapper(), torch.device(device), set_current_vllm_config(config):
        offloader = create_offloader(config.offload_config)
        if not isinstance(offloader, AscendPrefetchOffloader):
            raise TypeError('fixed route requires the Ascend NZ-aware prefetch offloader')
        set_offloader(offloader)
        root_model = get_model(vllm_config=config)
        # Matches NPUModelRunner.load_model: post_init follows get_model and
        # all quant postload work. Never manually run quant transforms twice.
        offloader.post_init()
        root_model.eval()
    model = (root_model if isinstance(root_model, Glm5NextForCausalLM)
             else getattr(root_model, 'language_model', None))
    if not isinstance(model, Glm5NextForCausalLM):
        raise TypeError('checkpoint loader did not construct the pinned complete causal GLM model')
    if len(model.model.layers) != model.config.num_hidden_layers:
        raise ValueError('loaded causal model does not contain every configured layer')
    if model.config.qk_rope_head_dim != 0:
        raise ValueError('this candidate binds the pinned GLM MLA-NoPE checkpoint only')
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    # KDA's MambaBase declares kv_cache as an annotation without initializing
    # an instance value. Publish empty baseline slots before collecting layers;
    # otherwise hasattr(kv_cache) would silently omit every recurrent layer.
    for layer in config.compilation_config.static_forward_context.values():
        if isinstance(layer, AttentionLayerBase) and not hasattr(layer, 'kv_cache'):
            layer.kv_cache = ()
    return LoadedReferenceModel(root_model, model, config, offloader, device)


@dataclass(frozen=True)
class ReferenceWorkerRuntime:
    loaded: LoadedReferenceModel
    cache: object
    backend: object
    adapter: object

    @property
    def runner(self):
        return self.adapter.bridge._runner

    def install(self, worker):
        worker.install_glm_single_request_adapter(self.adapter)
        worker._glm_worker_runtime = self


def prepare_reference_worker_runtime(model_dir, device, *, worker_group,
                                     local_rank, sampler,
                                     max_context_tokens, max_step_tokens):
    """Concrete assembly entry, called by each already initialized Worker rank.

    Driver uses DeepInfer's installed sampler. The returned runtime installs at
    the existing Worker hook; callers then use the original DeepInfer engine's
    from_bounded_glm_worker path and its real scheduler/callback lifecycle.
    """
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.forward_context import get_forward_context
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.models.glm5next.model import Glm5NextForCausalLM
    from .reference_backend import FullModelBackend
    from .reference_cache import ReferenceCacheRuntime
    from .reference_worker_runner import TorchRankAgreement, build_worker_adapter

    loaded = load_reference_model(
        model_dir, device, worker_group=worker_group, local_rank=local_rank,
        max_context_tokens=max_context_tokens, max_step_tokens=max_step_tokens)
    with set_current_vllm_config(loaded.vllm_config):
        cache = ReferenceCacheRuntime(loaded, max_context_tokens=max_context_tokens)
    agreement = TorchRankAgreement(worker_group, device, get_tp_group)

    def metadata_builder(span, request_cache):
        with set_current_vllm_config(loaded.vllm_config):
            return cache.build_metadata(span, request_cache)

    @contextmanager
    def context(metadata, config, *, num_tokens, slot_mapping):
        # Ascend's context manager owns the extra MoE/MLA runtime fields and
        # calls the real base context internally. It lacks a slot_mapping
        # argument, so publish the exact per-prefix maps on that active context.
        with set_ascend_forward_context(
                metadata, config, num_tokens=num_tokens,
                num_actual_tokens=num_tokens, model_instance=loaded.model):
            active = get_forward_context()
            active.slot_mapping = slot_mapping
            yield active

    backend = FullModelBackend(
        loaded.model, model_type=Glm5NextForCausalLM, contracts=cache.contracts,
        cache_factory=cache.allocate, metadata_builder=metadata_builder,
        set_forward_context=context, get_forward_context=get_forward_context,
        step_scope=loaded.step_scope, release_cache=cache.release,
        validate_tp=agreement.validate_tp, agree=agreement,
        is_driver=dist.get_rank(worker_group) == 0,
        max_context_tokens=max_context_tokens, max_step_tokens=max_step_tokens)
    adapter = build_worker_adapter(backend, sampler=sampler, worker_group=worker_group)
    return ReferenceWorkerRuntime(loaded, cache, backend, adapter)
