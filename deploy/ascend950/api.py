"""Loopback API adapter; never substitutes mock output for model inference."""
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from deepseek_recipe import (
    InferenceChunk, InferenceFinishReason, ModelVersion, PromptUsage, Tokenizer,
)
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src/deepseek-recipe/server-py"))
from main import PreparedRequest, RequestError, conversion_options, create_app

BACKEND = os.getenv("DEEPINFER_BACKEND_URL", "http://127.0.0.1:18111")
MODEL = Path(os.getenv("DEEPINFER_HF_MODEL_DIR", "/models/DeepSeek-V4.1-Flash-1017"))
CONTEXT = int(os.getenv("DEEPINFER_DEMO_CONTEXT", "4096"))
TOKENIZER_FILE = Path(os.getenv("DEEPINFER_TOKENIZER_FILE", str(MODEL / "tokenizer.json")))


def rpc(path, payload=None):
    request = urllib.request.Request(
        BACKEND + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


async def infer(request: PreparedRequest):
    if request.images:
        raise RequestError("This initial inference check accepts text only.", 400)
    if request.stream:
        raise RequestError("This initial check supports stream=false only.", 400)
    options = request.inference_options
    if options.output_constraint is not None or options.logprobs:
        raise RequestError("Constrained decoding and logprobs are not wired in this adapter.", 400)
    if not TOKENIZER_FILE.is_file():
        raise RequestError("Tokenizer download is incomplete; inference is not ready.", 503)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    ids = tokenizer.encode(request.prompt)
    try:
        maximum = request.conversation_request.resolve_max_tokens(len(ids), CONTEXT)
    except ValueError as error:
        raise RequestError(str(error), 400) from error
    payload = {
        "prompt_token_ids": ids, "max_tokens": maximum,
        "temperature": options.temperature, "top_p": options.top_p,
        "seed": options.seed,
    }
    try:
        result = await asyncio.to_thread(rpc, "/generate", payload)
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        raise RequestError("DeepInfer backend is unavailable; no model answer was generated.", 503) from error
    if "error" in result:
        raise RequestError(result["error"], 503)
    model_config = json.loads((MODEL / "config.json").read_text())
    tokenizer_config = json.loads((MODEL / "tokenizer_config.json").read_text())
    eos_ids = model_config["eos_token_id"]
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
    ended_with_eos = bool(result["token_ids"]) and result["token_ids"][-1] in eos_ids
    text = result["text"]
    eos_token = tokenizer_config["eos_token"]
    eos_text = eos_token["content"] if isinstance(eos_token, dict) else eos_token
    if ended_with_eos and eos_text and text.endswith(eos_text):
        text = text[:-len(eos_text)]
    yield InferenceChunk.ready(prompt_usage=PromptUsage(prompt_tokens=len(ids)))
    yield InferenceChunk.text(text, token_count=result["output_tokens"])
    reason = InferenceFinishReason.Length if not ended_with_eos and result["output_tokens"] >= maximum else InferenceFinishReason.Stop
    yield InferenceChunk.finish(reason)


app = create_app(
    infer=infer,
    options=conversion_options().with_model_alias("DeepSeek-V4.1-Flash-1017", ModelVersion.DeepseekV41Flash),
)


@app.middleware("http")
async def reject_streaming(request, call_next):
    if request.method == "POST":
        try:
            body = json.loads(await request.body())
        except (ValueError, UnicodeDecodeError):
            body = {}
        if isinstance(body, dict) and body.get("stream") is True:
            return JSONResponse({"error": {"message": "This functional check supports stream=false only."}}, status_code=400)
    return await call_next(request)


@app.get("/healthz")
async def health():
    return {"api": "ready", "backend": "DeepInfer", "mock": False}


@app.get("/readyz")
async def ready():
    try:
        result = await asyncio.to_thread(rpc, "/readyz")
        return JSONResponse(result, status_code=200 if result.get("ready") else 503)
    except (OSError, urllib.error.URLError, TimeoutError):
        return JSONResponse({"ready": False, "reason": "engine not started", "mock": False}, status_code=503)
