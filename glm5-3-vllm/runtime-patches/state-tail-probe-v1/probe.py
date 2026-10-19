"""GLM5.3 native P/D tail diagnostic; installed only by the gated worker overlay.

No new event waits, device synchronization, kernels, or transfer operations.
Reads happen after native completion; sampled rows never prove a full prefix.
"""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import time


def need(ok, message):
    if not ok:
        raise ValueError(message)


def row_location(shape, stride, spec, block, offset, component):
    """Validate the matched native MLA packed / split Indexer view convention."""
    need(spec["type"] in ("AscendMLAAttentionSpec", "AscendSFAIndexerCacheSpec"), "unsupported spec")
    need(len(shape) == len(stride) == 4 and shape[1] == spec["block_size"] == 128,
         "requires native (blocks,128,heads,width) view")
    need(shape[2] == spec["heads"] == 1 and 0 <= block < shape[0] and 0 <= offset < 128,
         "invalid head/block/offset")
    need(not spec.get("li_c4", False) and spec.get("replication", 1) == 1, "C4/replicated layout unsupported")
    width = spec["head_size"] if component == 0 else spec["scale_dim"]
    if spec["type"] == "AscendMLAAttentionSpec":
        need(spec["packed_c8"] and component == 0, "requires one packed C8 main view")
    else:
        need(component in (0, 1) and (component == 0 or spec["scale_dim"] > 0), "invalid Indexer component")
    need(width > 0 and shape[3] == width and stride[3] == 1 and stride[2] == width,
         "row layout differs from native reshape")
    need(stride[1] == width and stride[0] >= 128 * width, "overlapping/transposed rows")
    return block, offset


def spec_info(spec):
    return {"type": type(spec).__name__, "block_size": spec.block_size,
            "heads": spec.num_kv_heads, "head_size": spec.head_size,
            "scale_dim": getattr(spec, "scale_dim", 0),
            "packed_c8": getattr(spec, "cache_sparse_sfa_c8", False),
            "li_c4": getattr(spec, "cache_sparse_li_c4", False),
            "replication": getattr(spec, "sfa_dcp_replicated_indexer_size", 1)}


def row_digest(dtype, shape, raw):
    header = json.dumps({"dtype": dtype, "shape": list(shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def install(ns, source_pins, prompt_hashes):
    import torch
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

    Runner, AsyncOutput = ns["NPUModelRunner"], ns["AsyncGPUModelRunnerOutput"]
    need(not getattr(Runner, "_glm53_tail_probe", False), "probe already installed")
    root = Path(os.environ["GLM53_STATE_PROBE_OUTPUT"])
    need(str(root).startswith("/out/") and ".." not in root.parts, "probe output must be under fresh /out")
    root.mkdir(parents=True, exist_ok=True)
    for relative, expected in source_pins.items():
        need(hashlib.sha256((Path("/vllm-workspace") / relative).read_bytes()).hexdigest() == expected,
             "runtime source identity changed: " + relative)
    lock = threading.RLock()
    state = {"runner": None, "worker": None, "epoch": 0, "frame": None, "transfers": {}}

    def stream_id():
        return int(torch.npu.current_stream().npu_stream)

    def emit(kind, **fields):
        worker = state["worker"]
        value = {"kind": kind, "process_monotonic_ns": time.monotonic_ns(),
                 "pid": os.getpid(), "tp_rank": getattr(worker or state["runner"], "tp_rank", None),
                 "device_run": True, "sampled_rows_only": True, **fields}
        # One file per worker process; record-level lock also protects against
        # transfer pool callbacks. No device waits or cross-node timestamps.
        with lock, (root / ("worker-%d.jsonl" % os.getpid())).open("a") as f:
            f.write(json.dumps(value, sort_keys=True) + "\n")

    def descriptor(worker, name):
        spec = worker.kv_cache_specs[worker.layer_name_to_spec_index[name]]
        tensors = worker.kv_caches[name]
        tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)
        return spec, tensors

    def sample(worker, name, block, offset, position, family, **origin):
        spec, tensors = descriptor(worker, name)
        info = spec_info(spec)
        expected_count = 1 if info["type"] == "AscendMLAAttentionSpec" else 1 + bool(info["scale_dim"])
        need(len(tensors) == expected_count, "component count mismatch")
        result = []
        for component, tensor in enumerate(tensors):
            row_location(tuple(tensor.shape), tuple(tensor.stride()), info, block, offset, component)
            expected_dtype = spec.dtype if component == 0 else spec.scale_dtype
            need(tensor.dtype == expected_dtype, "cache dtype differs from spec")
            row = tensor[block, offset]
            need(row.is_contiguous(), "non-contiguous logical row")
            # Byte reinterpretation of the real registered view; no packed
            # cache decoding, arithmetic checksum, clone or raw pointer access.
            raw = row.view(torch.uint8).cpu().numpy().tobytes()
            need(0 < len(raw) <= 4096, "row byte bound exceeded")
            result.append({"layer": name, "family": family,
                "group": worker.layer_name_to_group_index[name], "spec": info,
                "component": component, "dtype": str(tensor.dtype), "row_shape": list(row.shape),
                "position": position, "block": block, "offset": offset,
                "tensor_shape": list(tensor.shape), "tensor_stride": list(tensor.stride()),
                "storage_offset": tensor.storage_offset(), "bytes": len(raw),
                "sha256": row_digest(str(tensor.dtype), row.shape, raw), **origin})
        return result

    def frame_for(runner):
        cache = runner.execute_model_state
        if cache is None or runner.input_batch.num_reqs != 1 or not runner.is_kv_producer:
            return None
        req_id = runner.input_batch.req_ids[0]
        req = runner.requests[req_id]
        ids = req.prompt_token_ids
        n = len(ids or [])
        if n not in (256, 257):
            return None
        sha = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        # Dummy/warmup requests must retain native behavior. The functional
        # client uses the frozen prompt hashes; missing captures fail analysis.
        if sha != prompt_hashes[str(n)] or req.sampling_params.max_tokens != 1:
            return None
        need(cache.scheduler_output.num_scheduled_tokens[req_id] == n, "requires one cold unchunked prefill")
        new = [r for r in cache.scheduler_output.scheduled_new_reqs if r.req_id == req_id]
        need(len(new) == 1 and new[0].num_computed_tokens == 0 and runner.speculative_config.num_speculative_tokens == 1,
             "P cold/MTP1 precondition changed")
        return {"req_id": req_id, "n": n, "prompt_sha256": sha, "epoch": state["epoch"],
                "prompt": list(ids), "target_metadata": cache.attn_metadata,
                "target_ids": runner.input_ids.gpu[:n], "target_positions": cache.positions[:n],
                "target_hidden_last": cache.hidden_states[n - 1],
                "block_ids": [list(group) for group in req.block_ids],
                "runnable_hook_bound": getattr(runner.drafter._runnable, "__func__", None) is draft_run,
                "mtp_calls": 0, "mtp_metadata": {}, "scheduled_K": cache.scheduler_output.num_spec_tokens_to_schedule}

    original_init = Runner.__init__
    @functools.wraps(original_init)
    def runner_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        need(self.model_config.hf_text_config.model_type == "glm_moe_dsa", "GLM5.3 only")
        need(self.model_config.hf_text_config.num_hidden_layers == 78 and
             self.model_config.hf_text_config.num_nextn_predict_layers == 1, "unexpected model depth")
        need(self.use_async_scheduling and self.dcp_size == 1 and
             self.parallel_config.prefill_context_parallel_size == 1, "requires native async TP16 without CP")
        need(self.parallel_config.tensor_parallel_size == 16, "TP16 only")
        state["runner"] = self
        emit("runner_identity", proposer_type=type(self.drafter).__name__,
             proposer_source=inspect.getsourcefile(type(self.drafter)),
             proposer_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(type(self.drafter))).read_bytes()).hexdigest(),
             checked_source_sha256=source_pins,
             worker_overlay_sha256=hashlib.sha256(Path(ns["__file__"]).read_bytes()).hexdigest(),
             model_type=self.model_config.hf_text_config.model_type,
             method=self.speculative_config.method, K=self.speculative_config.num_speculative_tokens,
             cuda_event_is_npu_event=torch.cuda.Event is torch.npu.Event,
             runnable_hook_bound=getattr(self.drafter._runnable, "__func__", None) is draft_run)
        need(type(self.drafter).__name__ == "AscendEagleProposer" and
             self.speculative_config.method == "mtp" and self.speculative_config.enforce_eager,
             "this diagnostic only supports the observed GLM Eagle-based eager MTP path")
    Runner.__init__ = runner_init

    original_execute = Runner.execute_model
    @functools.wraps(original_execute)
    def execute(self, *args, **kwargs):
        with lock:
            scheduled = args[0] if args else kwargs["scheduler_output"]
            # Native zero-token KV polling does not overwrite model inputs.
            # Positive work is marked before its original input preparation.
            if scheduled.total_num_scheduled_tokens > 0:
                state["epoch"] += 1
        return original_execute(self, *args, **kwargs)
    Runner.execute_model = execute

    original_sample = Runner.sample_tokens
    @functools.wraps(original_sample)
    def sample_tokens(self, *args, **kwargs):
        with lock:
            frame = frame_for(self)
            state["frame"] = frame
        try:
            output = original_sample(self, *args, **kwargs)
            if frame is not None:
                need(isinstance(output, AsyncOutput), "P did not use native async output")
                output._glm53_tail_frame = frame
            return output
        finally:
            state["frame"] = None
    Runner.sample_tokens = sample_tokens

    original_inputs = AscendSpecDecodeBaseProposer.set_inputs_first_pass
    input_signature = inspect.signature(original_inputs)
    @functools.wraps(original_inputs)
    def first_inputs(self, *args, **kwargs):
        frame = state["frame"]
        bound = input_signature.bind(self, *args, **kwargs).arguments if frame is not None else None
        result = original_inputs(self, *args, **kwargs)
        if frame is not None:
            n = frame["n"]
            frame["proposer_source_hidden_last"] = bound["target_hidden_states"][n - 1]
            frame["proposer_copied_hidden_last"] = self.hidden_states[n - 1]
        return result
    AscendSpecDecodeBaseProposer.set_inputs_first_pass = first_inputs

    original_draft = AscendSpecDecodeBaseProposer._run_merged_draft
    signature = inspect.signature(original_draft)
    @functools.wraps(original_draft)
    def draft_run(self, *args, **kwargs):
        frame = state["frame"]
        if frame is not None:
            bound = signature.bind(self, *args, **kwargs).arguments
            need(bound["batch_size"] == 1 and bound["num_tokens"] == frame["n"], "unexpected MTP batch")
            frame.update(mtp_calls=frame["mtp_calls"] + 1, mtp_stream=stream_id(),
                         mtp_metadata=bound["multi_steps_attn_metadata"][0],
                         mtp_ids=self.input_ids[:frame["n"]], mtp_positions=self._get_positions(frame["n"]))
        result = original_draft(self, *args, **kwargs)
        if frame is not None:
            need(stream_id() == frame["mtp_stream"], "MTP stream changed")
            frame["mtp_returned"] = True
        return result
    AscendSpecDecodeBaseProposer._run_merged_draft = draft_run

    original_async_init = AsyncOutput.__init__
    @functools.wraps(original_async_init)
    def async_init(self, *args, **kwargs):
        self._glm53_constructor_stream = stream_id()
        original_async_init(self, *args, **kwargs)
    AsyncOutput.__init__ = async_init

    original_get = AsyncOutput.get_output
    @functools.wraps(original_get)
    def get_output(self):
        frame = getattr(self, "_glm53_tail_frame", None)
        native_wait_entry_ns = time.monotonic_ns()
        before = bool(self.async_copy_ready_event.query()) if frame is not None else None
        result = original_get(self)  # The original event wait is unchanged.
        native_wait_return_ns = time.monotonic_ns()
        if frame is None:
            return result
        with lock:
            header = {k: frame[k] for k in ("req_id", "n", "prompt_sha256", "mtp_calls", "scheduled_K", "runnable_hook_bound")}
            header.update(event_ready_before_native_wait=before,
                          event_ready_after_native_wait=bool(self.async_copy_ready_event.query()),
                          event_type=type(self.async_copy_ready_event).__module__ + "." + type(self.async_copy_ready_event).__name__,
                          constructor_stream=self._glm53_constructor_stream,
                          mtp_stream=frame.get("mtp_stream"), boundary="native_get_output_returned",
                          native_wait_entry_ns=native_wait_entry_ns, native_wait_return_ns=native_wait_return_ns)
            try:
                need(state["epoch"] == frame["epoch"], "input buffers advanced before observation")
                need(frame["runnable_hook_bound"], "actual request runnable is not the observed eager hook")
                need(header["event_ready_after_native_wait"] and torch.cuda.Event is torch.npu.Event,
                     "native event identity/completion not established")
                if frame["mtp_calls"]:
                    need(frame["mtp_calls"] == 1 and frame["mtp_returned"] and
                         frame["mtp_stream"] == self._glm53_constructor_stream,
                         "MTP stream/call count is not covered by this native copy wait")
                header["sample_start_ns"] = time.monotonic_ns()
                need(frame["target_ids"].cpu().tolist() == frame["prompt"], "target input identity changed")
                need(frame["target_positions"].cpu().tolist() == list(range(frame["n"])), "target position mismatch")
                if frame["mtp_calls"]:
                    need(frame["mtp_positions"].cpu().tolist() == list(range(frame["n"])), "MTP positions mismatch")
                    shifted = frame["mtp_ids"].cpu().tolist()
                    token = result.sampled_token_ids[0][0]
                    need(shifted == frame["prompt"][1:] + [token], "MTP shifted input mismatch")
                    hidden = []
                    for key in ("target_hidden_last", "proposer_source_hidden_last", "proposer_copied_hidden_last"):
                        value = frame[key]
                        need(value.is_contiguous() and value.numel() * value.element_size() <= 65536,
                             "hidden tail observation bound/layout changed")
                        raw = value.view(torch.uint8).cpu().numpy().tobytes()
                        hidden.append({"source": key, "dtype": str(value.dtype), "shape": list(value.shape),
                                       "sha256": row_digest(str(value.dtype), value.shape, raw)})
                    need(len({(r["dtype"], tuple(r["shape"]), r["sha256"]) for r in hidden}) == 1,
                         "target/source/copied hidden tail differs; cannot establish this provenance")
                    header.update(native_first_token=token, target_last_token=frame["prompt"][-1],
                                  mtp_last_token=shifted[-1], mtp_last_position=frame["n"] - 1,
                                  hidden_tail=hidden, hidden_tail_matches=True)
                worker, runner = state["worker"], state["runner"]
                mtp_names = set(runner.drafter.attn_layer_names)
                need(worker is not None and len(worker.kv_caches) <= 128, "cache registry missing/too large")
                rows, refusals = [], []
                for name in sorted(worker.kv_caches):
                    family = "MTP" if name in mtp_names else "target"
                    metadata = frame["mtp_metadata"] if family == "MTP" else frame["target_metadata"]
                    try:
                        need(name in metadata, "no actual per-layer writer metadata")
                        slots = metadata[name].slot_mapping
                        for pos in (frame["n"] - 2, frame["n"] - 1):
                            slot = int(slots[pos].cpu().item())
                            need(slot >= 0 and slot % 128 == pos % 128, "invalid/misaligned native writer slot")
                            rows.extend(sample(worker, name, slot // 128, slot % 128, pos, family,
                                               writer_slot=slot, writer_metadata_observed=True))
                    except (ValueError, AttributeError, KeyError) as exc:
                        refusals.append({"layer": name, "reason": str(exc)})
                need(sum(r["bytes"] for r in rows) <= 1048576, "row capture bound exceeded")
                emit("P_tail", **header, status="observed", mtp_layer_names=sorted(mtp_names),
                     rows=rows, refused_layers=refusals, group_block_ids=frame["block_ids"],
                     scope="Native call/metadata and sampled row bytes after existing completion; not a kernel trace or full-prefix proof")
            except Exception as exc:
                emit("P_tail", **header, status="refused", reason=type(exc).__name__ + ": " + str(exc))
        return result
    AsyncOutput.get_output = get_output

    original_register = MooncakeBaseConnectorWorker.register_kv_caches
    @functools.wraps(original_register)
    def register(self, caches):
        original_register(self, caches)
        state["worker"] = self
        aliases = {}
        def view_info(tensor):
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            alias = aliases.setdefault(key, len(aliases))
            return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                    "dtype": str(tensor.dtype), "storage_offset": tensor.storage_offset(),
                    "storage_alias": alias, "storage_bytes": storage.nbytes()}
        draft_model = state["runner"].drafter.get_model()
        emit("cache_registry", draft_model_type=type(draft_model).__module__ + "." + type(draft_model).__name__,
             layers=[{"name": name,
             "group": self.layer_name_to_group_index[name], "spec": spec_info(descriptor(self, name)[0]),
             "views": [view_info(t) for t in descriptor(self, name)[1]]}
             for name in sorted(caches)])
    MooncakeBaseConnectorWorker.register_kv_caches = register

    original_bucket = MooncakePullRecvingThread._execute_tp_transfer_bucket
    bucket_signature = inspect.signature(original_bucket)
    @functools.wraps(original_bucket)
    def bucket(self, *args, **kwargs):
        bound = bucket_signature.bind(self, *args, **kwargs).arguments
        result = original_bucket(self, *args, **kwargs)
        # The original READ completed successfully; copy CPU mapping only.
        with lock:
            for _, layer_pairs in bound["transfer_entries_by_spec"].items():
                for (local_idx, remote_idx), entries in layer_pairs.items():
                    for req_id, local_ids, remote_ids in entries:
                        state["transfers"].setdefault(req_id, []).append({
                            "layer": self.layer_names[local_idx],
                            "remote_layer": bound["remote_metadata"].layer_names[remote_idx],
                            "local_blocks": list(local_ids), "remote_blocks": list(remote_ids),
                            "remote_tp": bound["remote_tp_rank"], "remote_pp": bound["remote_pp_rank"],
                            "remote_pcp": bound["remote_pcp_rank"]})
        return result
    MooncakePullRecvingThread._execute_tp_transfer_bucket = bucket

    original_receive = MooncakePullRecvingThread._handle_requests
    @functools.wraps(original_receive)
    def receive(self, remote_engine_id, remote_host, remote_port, requests):
        failed = original_receive(self, remote_engine_id, remote_host, remote_port, requests)
        native_receive_return_ns = time.monotonic_ns()
        # Called inside run(), before its finally block publishes finished.
        with lock:
            worker = state["worker"]
            need(worker is not None and self.local_metadata is worker.transfer_metadata, "receiver/cache ownership mismatch")
            for req_id, request in requests.items():
                mappings = state["transfers"].pop(req_id, [])
                n = request.remote_num_prompt_tokens
                header = {"req_id": req_id, "remote_req_id": request.remote_request_id,
                          "remote_engine_id": remote_engine_id, "n": n,
                          "native_receive_return_ns": native_receive_return_ns,
                          "boundary": "native_handle_requests_returned_before_finished_enqueue"}
                if n not in (256, 257):
                    emit("D_tail", **header, status="refused", reason="unexpected prompt length")
                    continue
                rows, refusals = [], []
                try:
                    need(req_id not in failed and request.num_computed_tokens == 0, "failed/non-cold receive")
                    need(request.local_num_prompt_tokens == n and request.num_external_tokens == n,
                         "incomplete logical receive declaration")
                    need(self.dcp_size == self.pcp_size == self.pp_size == 1 and self.tp_size == 16,
                         "unsupported transfer topology")
                    header["sample_start_ns"] = time.monotonic_ns()
                    mtp_names = set(state["runner"].drafter.attn_layer_names)
                    for mapping in mappings:
                        name = mapping["layer"]
                        try:
                            need(name == mapping["remote_layer"] and mapping["remote_pp"] == mapping["remote_pcp"] == 0,
                                 "layer/topology mismatch")
                            for pos in (n - 2, n - 1):
                                index, offset = divmod(pos, 128)
                                need(len(mapping["local_blocks"]) == len(mapping["remote_blocks"]) > index,
                                     "last row absent from actual transfer block pairs")
                                rows.extend(sample(worker, name, mapping["local_blocks"][index], offset, pos,
                                    "MTP" if name in mtp_names else "target",
                                    remote_tp=mapping["remote_tp"], remote_block=mapping["remote_blocks"][index]))
                        except (ValueError, AttributeError, KeyError) as exc:
                            refusals.append({"layer": name, "reason": str(exc)})
                    need(rows and sum(r["bytes"] for r in rows) <= 1048576, "no rows/capture bound exceeded")
                    emit("D_tail", **header, status="observed", rows=rows, refused_layers=refusals,
                         mappings=mappings, scope="Sampled rows after native synchronous READ API; no D query yet; not a full-prefix proof")
                except Exception as exc:
                    emit("D_tail", **header, status="refused", reason=type(exc).__name__ + ": " + str(exc),
                         rows=rows, refused_layers=refusals, mappings=mappings)
        return failed
    MooncakePullRecvingThread._handle_requests = receive
    Runner._glm53_tail_probe = True
