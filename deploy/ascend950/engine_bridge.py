"""Small synchronized EP8/DP8 inference harness, for functional checks only.

Each HTTP request is mirrored to the eight DP ranks; rank 0 supplies the answer.
This is deliberately not a throughput benchmark or a production DP scheduler.
"""
import hashlib
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TILELANG_CACHE_DIR", str(ROOT / "runtime-20261021/tilelang-cache"))
sys.path[:0] = [str(ROOT / "src/DeepInfer/examples"), str(ROOT / "src/DeepInfer")]
DEVICES = 8
MODEL = os.getenv("DEEPINFER_HF_MODEL_DIR", "/models/DeepSeek-V4.1-Flash-1017")


def worker(rank, inbox, results):
    llm = None
    initialized = time.monotonic()
    print(f"rank {rank}: model initialization started", flush=True)
    try:
        from _config import create_llm, setup_common_env
        setup_common_env(model_name=MODEL, sampler_force_topk=4096, set_override_experts=False, enable_eplb=False)
        llm = create_llm(
            MODEL, rank, ep_size=DEVICES, cp_size=1, data_parallel_size=DEVICES,
            max_model_len=4096, max_num_batched_tokens=256, max_num_seqs=1,
            gpu_memory_utilization=0.85, lookahead_tokens=0,
        )
        from deepinfer.sampling_params import SamplingParams
        print(f"rank {rank}: model ready after {time.monotonic()-initialized:.3f}s", flush=True)
        results.put({"kind": "ready", "rank": rank, "elapsed_seconds": time.monotonic()-initialized})
        while True:
            job = inbox.get()
            if job is None:
                break
            params = SamplingParams(
                max_tokens=job["max_tokens"], temperature=job["temperature"],
                top_p=job["top_p"], seq_seed=job["seed"] or 0,
            )
            generated_at = time.monotonic()
            output = llm.generate(prompt_token_ids=[job["prompt_token_ids"]], sampling_params=params, use_tqdm=False)[0]
            ids = [token.token_id for token in output]
            response = {"kind": "result", "rank": rank, "job": job["job"],
                        "elapsed_seconds": time.monotonic()-generated_at,
                        "token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
            print(f"rank {rank}: job {job['job']} generated {len(ids)} tokens in {response['elapsed_seconds']:.3f}s", flush=True)
            if rank == 0:
                response.update(text=llm.get_tokenizer().decode(ids), token_ids=ids, output_tokens=len(ids))
            results.put(response)
    except BaseException:
        results.put({"kind": "error", "rank": rank, "error": traceback.format_exc()})
    finally:
        if llm is not None:
            llm.close()


def serve():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    inboxes = [ctx.Queue() for _ in range(DEVICES)]
    workers = [ctx.Process(target=worker, args=(rank, inboxes[rank], results)) for rank in range(DEVICES)]
    for process in workers:
        process.start()
    def receive_event(timeout):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            try:
                return results.get(timeout=min(5,deadline-time.monotonic()))
            except queue.Empty:
                exited = [(rank, process.exitcode) for rank,process in enumerate(workers) if process.exitcode is not None]
                if exited:
                    raise RuntimeError(f"Worker exited unexpectedly: {exited}")
        raise TimeoutError(f"No worker result within {timeout} seconds")

    ready = set()
    try:
        while len(ready) < DEVICES:
            try:
                event = receive_event(3600)
            except queue.Empty:
                raise RuntimeError("Model initialization timed out")
            if event["kind"] == "error":
                raise RuntimeError(event["error"])
            ready.add(event["rank"])
    except BaseException:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join(timeout=5)
        raise
    lock = threading.Lock()
    state = {"ready": True, "next_job": 0}

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, data):
            raw = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != "/readyz":
                return self.respond(404, {"error": "unknown endpoint"})
            live = all(process.is_alive() for process in workers)
            return self.respond(200 if state["ready"] and live else 503, {"ready": state["ready"] and live, "ranks": DEVICES, "mirrored_requests": True})

        def do_POST(self):
            if self.path != "/generate":
                return self.respond(404, {"error": "unknown endpoint"})
            if not lock.acquire(blocking=False):
                return self.respond(503, {"error": "single-request validation harness is busy"})
            try:
                if not state["ready"]:
                    return self.respond(503, {"error": "backend failed; restart required"})
                size = int(self.headers.get("Content-Length", 0))
                if size <= 0 or size > 65536:
                    return self.respond(400, {"error": "invalid request size"})
                job = json.loads(self.rfile.read(size))
                if not 1 <= job["max_tokens"] <= 4096 or len(job["prompt_token_ids"]) + job["max_tokens"] > 4096:
                    return self.respond(400, {"error": "4096-token check context exceeded"})
                job["job"] = state["next_job"]
                state["next_job"] += 1
                for inbox in inboxes:
                    inbox.put(job)
                seen, answer, rank_results = set(), None, []
                while len(seen) < DEVICES:
                    event = receive_event(600)
                    if event["kind"] == "error":
                        raise RuntimeError(event["error"])
                    if event.get("job") != job["job"]:
                        raise RuntimeError("worker result mismatch")
                    seen.add(event["rank"])
                    rank_results.append({key:event[key] for key in ("rank","elapsed_seconds","token_sha256")})
                    if event["rank"] == 0:
                        answer = event
                answer["rank_results"] = sorted(rank_results,key=lambda item:item["rank"])
                answer["all_rank_tokens_equal"] = len({item["token_sha256"] for item in rank_results}) == 1
                evidence_dir = ROOT / "runtime-20261021/generations"
                evidence_dir.mkdir(parents=True, exist_ok=True)
                (evidence_dir / f"job-{job['job']:03d}.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2))
                self.respond(200, answer)
            except BaseException as error:
                state["ready"] = False
                try:
                    self.respond(503, {"error": str(error)})
                finally:
                    threading.Thread(target=server.shutdown, daemon=True).start()
            finally:
                lock.release()

    try:
        print("DeepInfer EP8/DP8 functional backend ready on loopback:18111", flush=True)
        server = ThreadingHTTPServer(("127.0.0.1", 18111), Handler)
        server.serve_forever()
    finally:
        for inbox in inboxes:
            inbox.put(None)
        for process in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()


if __name__ == "__main__":
    try:
        serve()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
