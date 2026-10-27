"""Fail-closed 64-layer FX endpoint markers for root-owned vLLM diagnostics.

This file does not import torch/NPU. Root supplies FX graph operators and the
event factory so it can be reviewed and tested without mutating the runtime.
"""
from __future__ import annotations

import re
import os
import json

_LAYER_PATH = re.compile(r"^L\[['\"]self['\"]\]\.layers\.(\d+)(?:\.|$)")


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def build_task_annotation(marker_id):
    """Build bounded ASCII JSON during compile-time FX insertion, never capture."""
    if not isinstance(marker_id, str) or not marker_id or not marker_id.isascii():
        raise ValueError("marker_id must be nonempty ASCII")
    value = json.dumps({"q27_marker_id": marker_id, "q27_marker_schema": 1},
                       ensure_ascii=True, separators=(",", ":"))
    if len(value.encode("utf-8")) > 4096:
        raise ValueError("marker annotation exceeds 4KiB")
    return value


def find_layer_spans(nodes, expected_layers=64):
    """Resolve contiguous, ordered layer intervals only from nn_module_stack."""
    ordinals_by_layer = {i: [] for i in range(expected_layers)}
    ambiguous = []
    for ordinal, node in enumerate(nodes):
        meta = getattr(node, "meta", {})
        stack = meta.get("nn_module_stack", {}) if isinstance(meta, dict) else {}
        layer_ids = set()
        for path in _strings(stack):
            match = _LAYER_PATH.match(path)
            if match:
                layer_ids.add(int(match.group(1)))
        if len(layer_ids) > 1:
            ambiguous.append({"ordinal": ordinal, "layers": sorted(layer_ids)})
        elif len(layer_ids) == 1:
            layer_id = next(iter(layer_ids))
            if layer_id >= expected_layers:
                ambiguous.append({"ordinal": ordinal, "layers": [layer_id], "reason": "out_of_range"})
            else:
                ordinals_by_layer[layer_id].append(ordinal)

    missing = [i for i, ords in ordinals_by_layer.items() if not ords]
    if missing or ambiguous:
        return {"status": "unknown", "reason": "missing_or_ambiguous_layer_source",
                "missing_layers": missing, "ambiguous_nodes": ambiguous[:16],
                "ambiguous_node_count": len(ambiguous)}

    spans = []
    for layer_id, ords in ordinals_by_layer.items():
        lo, hi = min(ords), max(ords)
        if ords != list(range(lo, hi + 1)):
            return {"status": "unknown", "reason": "layer_nodes_not_contiguous",
                    "layer_id": layer_id, "ordinals": ords[:16]}
        spans.append({"layer_id": layer_id, "first_ordinal": lo, "last_ordinal": hi,
                      "node_count": len(ords)})

    for left, right in zip(spans, spans[1:]):
        if left["last_ordinal"] >= right["first_ordinal"]:
            return {"status": "unknown", "reason": "layer_spans_overlap",
                    "left_layer": left["layer_id"], "right_layer": right["layer_id"]}
        if right["first_ordinal"] != left["last_ordinal"] + 1:
            return {"status": "unknown", "reason": "layer_spans_not_globally_contiguous",
                    "left_layer": left["layer_id"], "right_layer": right["layer_id"]}
    return {"status": "exact", "layer_count": expected_layers, "spans": spans}


def insert_layer_endpoint_markers(concrete_graph, graph_module, *, prefix,
                                  create_event_by_name, tagged_event_record_op,
                                  anchor_op, expected_layers=64):
    """Add one event+anchor before/after each exact nn_module_stack layer span."""
    graph = getattr(graph_module, "graph", None)
    nodes = list(graph.nodes) if graph is not None else []
    if not nodes or not callable(getattr(graph, "call_function", None)):
        return {"status": "unknown", "reason": "fx_graph_api_unavailable"}
    span_result = find_layer_spans(nodes, expected_layers)
    if span_result["status"] != "exact":
        return span_result
    info = getattr(concrete_graph, "_aclgraph_cache_info", None)
    tag_names = getattr(info, "tagged_event_names", None)
    if not isinstance(tag_names, list) or not prefix or not callable(create_event_by_name):
        return {"status": "unknown", "reason": "tag_registry_or_factory_missing"}
    if not callable(tagged_event_record_op) or not callable(anchor_op):
        return {"status": "unknown", "reason": "marker_operators_missing"}

    tags = []
    markers = []
    for span in span_result["spans"]:
        layer_id = span["layer_id"]
        for edge in ("pre", "post"):
            marker_id = f"{prefix}:layer{layer_id:02d}:{edge}"
            annotation_json = build_task_annotation(marker_id)
            tag = marker_id + ":event"
            if tag in tag_names:
                return {"status": "unknown", "reason": "tag_collision", "tag": tag}
            tags.append(tag)
            markers.append({"marker_id": marker_id, "tag": tag, "layer_id": layer_id,
                            "edge": edge, "annotation_json": annotation_json,
                            "anchor_enum": "after_add" if edge == "pre" else "after_mul"})
    if len(set(tags)) != 2 * expected_layers:
        return {"status": "unknown", "reason": "duplicate_generated_tags"}

    # Pre-create and retain events before FX mutation/capture; root binds the
    # resulting tag/event refs to the exact outer vLLM graph through its owner.
    created = []
    try:
        for tag in tags:
            create_event_by_name(tag)
            created.append(tag)
    except BaseException:
        return {"status": "unknown", "reason": "event_creation_failed", "created_tags": created}

    original_nodes = list(nodes)
    by_layer = {span["layer_id"]: span for span in span_result["spans"]}
    marker_by_layer = {}
    for marker in markers:
        marker_by_layer.setdefault(marker["layer_id"], {})[marker["edge"]] = marker
    try:
        for layer_id in range(expected_layers):
            span = by_layer[layer_id]
            first = nodes[span["first_ordinal"]]
            last = nodes[span["last_ordinal"]]
            pre = marker_by_layer[layer_id]["pre"]
            post = marker_by_layer[layer_id]["post"]
            with graph.inserting_before(first):
                pre_event = graph.call_function(tagged_event_record_op, args=(pre["tag"], True))
            with graph.inserting_after(pre_event):
                graph.call_function(anchor_op, args=(pre["anchor_enum"], pre["marker_id"], pre["annotation_json"]))
            with graph.inserting_after(last):
                post_event = graph.call_function(tagged_event_record_op, args=(post["tag"], True))
            with graph.inserting_after(post_event):
                graph.call_function(anchor_op, args=(post["anchor_enum"], post["marker_id"], post["annotation_json"]))
        graph.lint()
        graph_module.recompile()
        tag_names.extend(tags)
    except BaseException:
        # The caller must abort this compile attempt and retain every created
        # Event reference until the owning outer graph is known and drained.
        raise

    current = list(graph.nodes)
    inserted = 4 * expected_layers
    if len(current) != len(original_nodes) + inserted:
        raise RuntimeError("unexpected marker node count")
    original_set = set(original_nodes)
    filtered = [node for node in current if node in original_set]
    if filtered != original_nodes:
        raise RuntimeError("original FX node identity/order changed")
    return {"status": "markers_inserted", "layer_count": expected_layers,
            "tagged_event_record_count": 2 * expected_layers,
            "anchor_node_count": 2 * expected_layers,
            "inserted_node_count": inserted, "tags": tags, "markers": markers,
            "spans": span_result["spans"], "original_node_order_preserved": True,
            "output_node_unchanged": current[-1] is original_nodes[-1]}


def make_anchor_impl(extension, result_rows, *, phase_provider, record_function,
                     pid_provider=os.getpid):
    """Build the Composite impl; register it and a no-op Meta impl before capture.

    The implementation queues only the existing public task-anchor callback,
    records an exact marker-named CPU scope, and retains its result object. It
    deliberately does not inspect callback fields or synchronize here.
    """
    allowed = {"after_add", "after_mul"}  # C++ API enum only; marker_id is identity.

    def anchor_impl(anchor_enum, marker_id, annotation_json):
        if anchor_enum not in allowed or not isinstance(marker_id, str) or not marker_id:
            raise ValueError("invalid anchor enum or marker identity")
        if (not isinstance(annotation_json, str)
                or len(annotation_json.encode("utf-8")) > 4096
                or "q27_marker_id" not in annotation_json
                or marker_id not in annotation_json):
            raise ValueError("invalid precomputed task annotation JSON")
        context = phase_provider()
        if not isinstance(context, dict):
            raise TypeError("phase_provider must return explicit context metadata")
        with record_function(marker_id):
            result = extension.anchor(anchor_enum, annotation_json)
        result_rows.append({"pid": pid_provider(), "phase": context.get("phase"),
                            "capture_state": context.get("capture_state"),
                            "anchor_enum": anchor_enum, "marker_id": marker_id,
                            "result": result})
        return None

    return anchor_impl


def register_anchor_op(torch, extension, result_rows, *, phase_provider,
                       record_function, namespace="q27_vllm_layer_diag"):
    """Register the user op before the model/capture; keep the Library alive."""
    lib = torch.library.Library(namespace, "DEF")
    lib.define("anchor(str anchor_enum, str marker_id, str annotation_json) -> ()")
    op = getattr(getattr(torch.ops, namespace), "anchor").default
    from torch.fx.node import has_side_effect
    has_side_effect(op)
    lib.impl("anchor", make_anchor_impl(extension, result_rows,
             phase_provider=phase_provider, record_function=record_function), "CompositeExplicitAutograd")
    lib.impl("anchor", lambda anchor_enum, marker_id, annotation_json: None, "Meta")
    return {"library": lib, "op": op}
