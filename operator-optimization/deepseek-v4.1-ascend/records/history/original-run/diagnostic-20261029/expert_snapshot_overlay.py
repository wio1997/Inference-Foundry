"""Disabled-by-default, metadata-only draft for isolated MC2 diagnostics.

This does NOT sample tensor values, install itself, or implement an NPU
snapshot backend. Loading this module imports neither torch nor torch_npu.
Install only in an explicitly isolated diagnostic process after review. In
particular a graph capture sees no copy node from this observer. An observation
is host-dispatch metadata for a future experiment, never an NPU completion,
successful snapshot, or proof that graph replay invoked this Python wrapper.
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import sys
from dataclasses import dataclass
from typing import Callable


EXPECTED_OP = "_C_ascend::dispatch_ffn_combine"


@dataclass(frozen=True)
class LayerBinding:
    layer_index: int
    routed_object: object
    slot_label: str  # descriptive only: no device storage allocated


def bind_main_layers(model_root: object, target_layers: frozenset[int]) -> dict[int, LayerBinding]:
    """Bind actual main model objects, rejecting ambiguous or missing layers.

    Caller must supply the independently identified main-model root. This
    routine does not unwrap arbitrary graph wrappers, enumerate parameters,
    or assume names alone distinguish draft from main.
    """
    if not target_layers or any(type(i) is not int or i < 0 for i in target_layers):
        raise ValueError("invalid target layers")
    try:
        layers = model_root.model.layers
    except AttributeError as exc:
        raise ValueError("main model root layout is unknown") from exc
    bindings = {}
    for decoder in layers:
        idx = getattr(decoder, "layer_idx", None)
        if type(idx) is not int or idx not in target_layers:
            continue
        if idx in bindings:
            raise ValueError("duplicate decoder index")
        try:
            mlp = decoder.mlp
            routed = mlp.experts.routed_experts
            if mlp.layer_idx != idx or not isinstance(routed.layer_name, str):
                raise ValueError("layer metadata mismatch")
        except AttributeError as exc:
            raise ValueError("routed expert ownership is unknown") from exc
        bindings[idx] = LayerBinding(idx, routed, f"main-layer-{idx}-UNALLOCATED")
    if set(bindings) != set(target_layers):
        raise ValueError("not all target main layers were bound")
    if len({id(item.routed_object) for item in bindings.values()}) != len(bindings):
        raise ValueError("different layers share one routed object")
    return bindings


def _get_schema_argument(func, args, kwargs, name):
    for index, argument in enumerate(func._schema.arguments):
        if argument.name == name:
            if index < len(args):
                return args[index]
            if name in kwargs:
                return kwargs[name]
            raise ValueError("required op argument absent")
    raise ValueError("expected op argument not in schema")


def _anonymous_rank(secret: bytes, group_name: str, rank: int, global_rank: int) -> str:
    if not secret or not isinstance(group_name, str):
        raise ValueError("anonymous identity unavailable")
    message = f"{len(group_name)}:{group_name}:{rank}:{global_rank}".encode()
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _forward_observed(func, args, kwargs, expected_count, observations):
    """Observe if possible, then call the supplied operation exactly once."""
    try:
        schema = func._schema
        if schema.name == EXPECTED_OP:
            capacity = _get_schema_argument(func, args, kwargs, "max_output_size")
            count_out = _get_schema_argument(func, args, kwargs, "expert_token_nums")
            valid = type(capacity) is int and capacity > 0 and count_out is expected_count
            observations.append(capacity if valid else None)
    except Exception:
        # An unreadable schema might itself have been the target; therefore
        # invalidate this call's observation and still pass it through.
        observations.append(None)
    return func(*args, **kwargs)


def install_metadata_observer(
    comm_class: type,
    *,
    enabled: bool = False,
    main_bindings: dict[int, LayerBinding] | None = None,
    target_mc2_ranks: frozenset[int] = frozenset(),
    anonymous_key: bytes = b"",
    execution_context: Callable[[], str | None] | None = None,
    global_rank: Callable[[], int | None] | None = None,
    emit: Callable[[dict], None] | None = None,
) -> bool:
    """Install process-local metadata observer only when explicitly enabled.

    Execution context must return capture or eager_host_call only when
    independently observed. eager_host_call means this Python call ran; it is
    not proof that compiled regions or other streams are absent. replay and
    unknown modes are refused because Python hooks need not run on replay.

    The observer adds dispatcher overhead and does not capture any count.
    Arbitrary boolean claims cannot enable a device copy: no such backend
    exists in this module. Keep this disabled in the baseline service.
    """
    if not enabled:
        return False
    if not main_bindings or not target_mc2_ranks or not anonymous_key:
        raise ValueError("layer binding, allowed ranks and anonymous key are required")
    if not callable(execution_context) or not callable(global_rank) or not callable(emit):
        raise ValueError("context, rank and output callbacks are required")
    if any(type(i) is not int or i < 0 for i in target_mc2_ranks):
        raise ValueError("invalid rank allowlist")
    original = getattr(comm_class, "fused_experts", None)
    if not callable(original) or getattr(original, "_wio_meta_observer", False):
        raise ValueError("original fused_experts unavailable or already wrapped")
    # Only the explicitly enabled isolated installation imports torch.
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except Exception:
        # No patch has been installed and no target op has run here.
        return False

    @functools.wraps(original)
    def wrapped(comm, fused_experts_input, *args, **kwargs):
        try:
            layer = fused_experts_input.layer
            matching = [binding for binding in main_bindings.values()
                        if binding.routed_object is layer]
            dispatcher = comm.token_dispatcher
            rank = dispatcher.ep_rank_id
            size = dispatcher.ep_world_size
            world_rank = global_rank()
            context = execution_context()
            if (len(matching) != 1 or type(rank) is not int or rank not in target_mc2_ranks
                    or type(size) is not int or not (0 <= rank < size)
                    or type(world_rank) is not int or world_rank < 0
                    or context not in ("capture", "eager_host_call")):
                raise ValueError("unverified call identity")
            binding = matching[0]
            rank_token = _anonymous_rank(
                anonymous_key, dispatcher.moe_all_to_all_group_name, rank, world_rank
            )
        except Exception:
            # Unknown identity is not evidence. Do not disrupt inference.
            return original(comm, fused_experts_input, *args, **kwargs)

        observations = []

        class ObserveCall(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                try:
                    count = comm.expert_token_nums
                except Exception:
                    observations.append(None)
                    return func(*args, **kwargs)
                return _forward_observed(func, args, kwargs, count, observations)

        # These hooks observe host dispatch only. A missing/duplicated op or a
        # different tensor object is treated as unknown, never guessed.
        try:
            mode = ObserveCall()
        except Exception:
            # No mode entered and no target op has run.
            return original(comm, fused_experts_input, *args, **kwargs)
        try:
            mode.__enter__()
        except Exception as exc:
            # Torch may partially mutate mode state before raising. Abort this
            # diagnostic invocation; do not run the target or guess rollback.
            raise RuntimeError("diagnostic mode entry failed; process state unknown") from exc
        try:
            result = original(comm, fused_experts_input, *args, **kwargs)
        except BaseException:
            original_error = sys.exc_info()
            try:
                mode.__exit__(*original_error)
            except Exception:
                # Preserve the original failure. Mode state may be corrupted;
                # the isolated diagnostic process must not be reused.
                add_note = getattr(original_error[1], "add_note", None)
                if callable(add_note):
                    add_note("Diagnostic mode cleanup also failed; retire this process.")
            raise
        try:
            mode.__exit__(None, None, None)
        except Exception as exc:
            # Original already ran: NEVER retry it. Abort and retire process.
            raise RuntimeError("diagnostic mode cleanup failed after target execution") from exc
        capacity = observations[0] if len(observations) == 1 else None
        try:
            emit({"kind": "mc2_host_dispatch_metadata", "layer": binding.layer_index,
                  "anonymous_rank": rank_token, "mc2_world_size": size,
                  "context": context, "actual_capacity": capacity,
                  "capacity_status": "observed" if capacity is not None else "unknown",
                  "slot_label": binding.slot_label, "slot_status": "unallocated",
                  "count_status": "unknown_not_sampled", "completion_status": "unknown"})
        except Exception:
            # Diagnostics cannot turn a successful inference into a failure.
            pass
        return result

    wrapped._wio_meta_observer = True
    comm_class.fused_experts = wrapped
    return True


def request_device_snapshot(*_args, **_kwargs):
    """Hard refusal: stream, copy, event and graph-lifetime proofs are absent."""
    raise RuntimeError("NPU snapshot backend is not implemented or verified")
