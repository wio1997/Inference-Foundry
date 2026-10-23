"""Frozen token-ID mapping and DP routing; AISBench owns scheduling and timing."""
import json
from pathlib import Path
from ais_bench_prefix_cache.models.vllm_prefix_cache_api import VLLMPrefixCacheAPI
from ais_bench.benchmark.registry import MODELS

@MODELS.register_module()
class E26MappedAPI(VLLMPrefixCacheAPI):
    def __init__(self, corpus, framework, run_id, cache_namespace="", **kwargs):
        self.rows = {str(r["request_index"]): r for r in
                     (json.loads(x) for x in Path(corpus).read_text().splitlines() if x.strip())}
        self.framework, self.run_id = framework, run_id
        self.cache_namespace = cache_namespace or run_id
        super().__init__(**kwargs)

    async def get_request_body(self, input_data, max_out_len, output, **args):
        key = str(input_data).strip()
        row = self.rows[key]
        rank = row["request_index"] % 8
        body = await super().get_request_body(row["input_token_ids"], max_out_len, output, dp_rank=rank)
        output.input = key
        body.update(request_id=self.run_id+"-"+key, user=key, return_token_ids=True,
                    seed=3000, top_k=1, top_p=1.0, temperature=0.0, ignore_eos=True)
        if self.framework == "deepinfer":
            body["assigned_dp_rank"] = rank
        else:
            body["cache_salt"] = self.cache_namespace+"-"+key
        output.extra_details_data.update(logical_id=row["logical_id"], owner_rank=rank,
                                         request_id=body["request_id"], input_ids=row["input_token_ids"])
        return body

    async def parse_stream_response(self, data, output):
        output.extra_details_data.setdefault("first_sse_payload", data)
        for choice in data.get("choices", []):
            output.extra_details_data.setdefault("native_output_token_ids", []).extend(choice.get("token_ids") or [])
            if choice.get("finish_reason"):
                output.extra_details_data["finish_reason"] = choice["finish_reason"]
        if data.get("usage"):
            output.extra_details_data["server_usage"] = data["usage"]
        await super().parse_stream_response(data, output)
