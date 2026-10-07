"""Execute actual streaming-generator ASTs on CPU; no engine/NPU imports.

Protocol serialization and engine outputs are doubles. Both complete generator
bodies, usage selection, GenerationError and existing error conversion are the
installed source. This checks control flow, not full HTTP/Pydantic acceptance.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "pd_error_sources"
CANDIDATE = ROOT / "pd_error_candidate"


class Model(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, key):
        return [] if key == "tool_calls" else None

    def model_dump(self, **kwargs):
        def unpack(value):
            if isinstance(value, Model):
                return {k: unpack(v) for k, v in vars(value).items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            return value
        result = unpack(self)
        if kwargs.get("exclude_none"):
            result = {k: v for k, v in result.items() if v is not None}
        return result

    def model_dump_json(self, **kwargs):
        return json.dumps(self.model_dump(**kwargs), sort_keys=True)


class QuietLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def find_node(path, name, parent=None):
    tree = ast.parse(path.read_text())
    if parent:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent)
    return next(n for n in tree.body if getattr(n, "name", None) == name)


def compile_nodes(nodes, namespace):
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "actual_installed_generator_AST", "exec"), namespace)


class NoneParser:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def parse_delta(self, **kwargs):
        self.calls += 1
        return None


def load_serving(api, patched, parser=False):
    filename = "chat_serving.py" if api == "chat" else "completion_serving.py"
    original_name = "vllm__entrypoints__openai__" + ("chat_completion" if api == "chat" else "completion") + "__serving.py"
    path = CANDIDATE / filename if patched else SOURCES / original_name
    namespace = dict(
        time=SimpleNamespace(time=lambda: 1234), json=json, HTTPStatus=HTTPStatus,
        VLLMServerError=Exception, logger=QuietLogger(), as_list=list,
        ChatCompletionNamedToolChoiceParam=type("NamedChoice", (), {}),
        maybe_filter_parallel_tool_calls=lambda choice, request: choice,
        build_per_request_timing_metrics=lambda *args: None,
    )
    for name in ("ChatCompletionResponseStreamChoice", "ChatCompletionStreamResponse", "DeltaMessage", "CompletionStreamResponse", "CompletionResponseStreamChoice", "UsageInfo", "PromptTokenUsageInfo"):
        namespace[name] = Model
    compile_nodes([
        find_node(SOURCES / "vllm__entrypoints__openai__engine__protocol.py", "GenerationError"),
        find_node(SOURCES / "vllm__entrypoints__serve__utils__api_utils.py", "should_include_usage"),
        find_node(SOURCES / "vllm__entrypoints__openai__chat_completion__serving.py", "_make_prompt_tokens_details"),
    ], namespace)
    base = SOURCES / "vllm__entrypoints__generate__base__serving.py"
    nodes = [find_node(base, name, "GenerateBaseServing") for name in ("_raise_if_error", "_convert_generation_error_to_streaming_response", "create_streaming_error_response")]
    cls_name = "OpenAIServingChat" if api == "chat" else "OpenAIServingCompletion"
    method = "chat_completion_stream_generator" if api == "chat" else "completion_stream_generator"
    nodes.append(find_node(path, method, cls_name))
    compile_nodes(nodes, namespace)
    obj = SimpleNamespace(
        parser_cls=NoneParser if parser else None, model_config=None,
        enable_force_include_usage=False, enable_prompt_tokens_details=True,
        enable_per_request_metrics=False, enable_log_outputs=False,
        enable_log_deltas=False, request_logger=None, system_fingerprint="cpu",
        get_chat_request_role=lambda request: "assistant",
        create_error_response=lambda message, err_type, status_code, param: Model(error=Model(message=str(message), type=err_type, code=int(status_code), param=param)),
        _extract_prompt_text=lambda input: input["prompt"],
        _create_chat_logprobs=lambda **kwargs: Model(content=[]),
        _create_completion_logprobs=lambda **kwargs: Model(tokens=[]),
    )
    for node in nodes:
        setattr(obj, node.name, namespace[node.name].__get__(obj))
    return obj, path


def output(reason=None, text="", ids=(), index=0, logprobs=None):
    return SimpleNamespace(index=index, text=text, token_ids=list(ids), finish_reason=reason, stop_reason=None, logprobs=logprobs)


def result(outputs):
    return SimpleNamespace(prompt="prompt", prompt_token_ids=[10, 11], encoder_prompt_token_ids=None, prompt_logprobs=None, num_cached_tokens=1, num_cache_creation_tokens=0, metrics=None, outputs=outputs)


async def run_case(api, patched, batches, *, n=1, prompts=1, usage=True, continuous=False, parser=False, return_ids=True, logprobs=False, echo=False):
    serving, path = load_serving(api, patched, parser)
    request = SimpleNamespace(n=n, stream_options=SimpleNamespace(include_usage=usage, continuous_usage_stats=continuous), tool_choice=None, tools=None, return_prompt_text=False, return_token_ids=return_ids, echo=echo, logprobs=(True if api == "chat" else 1) if logprobs else (False if api == "chat" else None), top_logprobs=1 if logprobs else None, logprob_token_ids=None, return_tokens_as_token_ids=False, include_reasoning=True, max_tokens=8)
    metadata = SimpleNamespace(final_usage_info=None)
    async def items():
        for prompt_idx, outputs in batches:
            res = result(outputs)
            yield res if api == "chat" else (prompt_idx, res)
    if api == "chat":
        generator = serving.chat_completion_stream_generator(request, items(), "req", "model", [], object(), metadata)
    else:
        generator = serving.completion_stream_generator(request, [{"prompt": "prompt"}] * prompts, items(), "req", 1234, "model", prompts, object(), metadata)
    values = [line async for line in generator]
    assert values[-1] == "data: [DONE]\n\n"
    events = [json.loads(line[6:]) for line in values[:-1]]
    return {"events": events, "metadata": metadata.final_usage_info.model_dump() if metadata.final_usage_info else None}, hashlib.sha256(path.read_bytes()).hexdigest()


def choices(data):
    return [choice for event in data["events"] for choice in event.get("choices", [])]


def errors(data):
    return [event["error"] for event in data["events"] if "error" in event]


async def main():
    checks = []
    def record(api, label, expected, value):
        checks.append(dict(api=api, case=label, expected=expected, observed=value, passed=True))
    for api in ("chat", "completion"):
        failed = [(0, [output("error")])]
        before, original_hash = await run_case(api, False, failed)
        after, patched_hash = await run_case(api, True, failed)
        assert not errors(before), before
        assert len(errors(after)) == 1 and errors(after)[0]["type"] == "InternalServerError" and errors(after)[0]["code"] == 500, after
        assert not any(choice.get("token_ids") for choice in choices(after))
        assert not any(event.get("usage") for event in after["events"])
        record(api, "first_empty_error", "original drops error; patch emits canonical 500 and DONE", {"original": before, "patched": after})
        variants = [
            ("no_usage_error", dict(usage=False)),
            ("continuous_usage_error", dict(continuous=True)),
            ("logprobs_error_without_logprobs", dict(logprobs=True)),
            ("no_token_ids_error", dict(return_ids=False)),
            ("echo_error", dict(echo=True)),
        ]
        if api == "chat":
            variants.append(("parser_error", dict(parser=True)))
        for label, kwargs in variants:
            data, _ = await run_case(api, True, failed, **kwargs)
            assert len(errors(data)) == 1 and errors(data)[0]["code"] == 500, (label, data)
            record(api, label, "canonical internal error", errors(data))
        data, _ = await run_case(api, True, [(0, [output()]), *failed])
        assert len(errors(data)) == 1 and errors(data)[0]["code"] == 500
        record(api, "prefill_then_empty_error", "error survives empty prefill suppression", errors(data))
        for reason in ("stop", "length"):
            before, _ = await run_case(api, False, [(0, [output(reason)])])
            after, _ = await run_case(api, True, [(0, [output(reason)])])
            assert not any(c.get("finish_reason") for c in choices(before))
            assert [c["finish_reason"] for c in choices(after) if c.get("finish_reason")] == [reason]
            assert not errors(after)
            record(api, "empty_terminal_" + reason, "zero-token normal terminal emitted once", reason)
        normal = [(0, [output()]), (0, [output(text="a", ids=[1])]), (0, [output("stop", text="b", ids=[2])])]
        for label, kwargs in [("normal_parity", {}), ("normal_without_usage", {"usage": False}), ("normal_continuous_usage", {"continuous": True}), ("normal_echo", {"echo": True}), ("normal_no_token_ids", {"return_ids": False})]:
            before, _ = await run_case(api, False, normal, **kwargs)
            after, _ = await run_case(api, True, normal, **kwargs)
            assert before == after, (label, before, after)
            record(api, label, "all SSE and usage unchanged", True)
        data, _ = await run_case(api, True, [*normal[:2], *failed])
        assert len(errors(data)) == 1 and errors(data)[0]["code"] == 500
        assert [t for c in choices(data) for t in (c.get("token_ids") or [])] == [1]
        record(api, "error_after_token", "keep prior token, then canonical error", True)
        unfinished = [(0, [output()])]
        before, _ = await run_case(api, False, unfinished)
        after, _ = await run_case(api, True, unfinished)
        assert before == after
        record(api, "unfinished_empty_parity", "empty prefill still suppressed", True)
        two = [(0, [output("stop", text="a", ids=[1], index=0), output("length", text="b", ids=[2], index=1)])]
        before, _ = await run_case(api, False, two, n=2)
        after, _ = await run_case(api, True, two, n=2)
        assert before == after
        record(api, "two_choice_normal_parity", "indices, tokens, terminals and aggregate usage unchanged", True)
        data, _ = await run_case(api, True, [(0, [output("stop", ids=[1], index=0), output("error", index=1)])], n=2)
        assert len(errors(data)) == 1 and errors(data)[0]["code"] == 500
        record(api, "second_choice_empty_error", "error survives n=2", True)
        if api == "chat":
            for reason in ("error", "stop"):
                data, _ = await run_case(api, True, [(0, [output(reason)])], parser=True)
                assert (len(errors(data)) == 1) if reason == "error" else any(c.get("finish_reason") == reason for c in choices(data))
                record(api, "parser_none_delta_" + reason, "terminal survives parser None delta", True)
        else:
            data, _ = await run_case(api, True, [(1, [output("error")])], prompts=2)
            assert len(errors(data)) == 1 and errors(data)[0]["code"] == 500
            record(api, "second_prompt_empty_error", "error survives multi-prompt index offset", True)
    out = dict(type="CPU_ACTUAL_GENERATOR_AST", checks=checks, passed=len(checks), performance_claim=False, full_API_accepted=False, limitations=["Protocol serialization and engine RequestOutput are CPU doubles; full HTTP/Pydantic/PD E2E remains unaccepted.", "No NPU execution or latency claim."])
    (ROOT / "pd_empty_terminal_CPU.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(dict(passed=len(checks), APIs=["chat", "completion"], original_error_drop_reproduced=True, performance_claim=False)))


if __name__ == "__main__":
    asyncio.run(main())
