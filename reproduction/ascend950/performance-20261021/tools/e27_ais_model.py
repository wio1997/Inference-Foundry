"""E27 request mapping; AISBench retains its own scheduler and metric clock."""
from e26_ais_model import E26MappedAPI
from ais_bench.benchmark.registry import MODELS


@MODELS.register_module()
class E27MappedAPI(E26MappedAPI):
    def __init__(self, cache_action="", **kwargs):
        self.cache_action = cache_action
        super().__init__(**kwargs)

    async def get_request_body(self, input_data, max_out_len, output, **args):
        body = await super().get_request_body(input_data, max_out_len, output, **args)
        if self.framework == "deepinfer" and self.cache_action:
            key = str(input_data).strip()
            body["user"] = "e27cache:" + self.cache_action + ":" + self.cache_namespace + ":" + key
            output.extra_details_data["cache_action"] = self.cache_action
        return body
