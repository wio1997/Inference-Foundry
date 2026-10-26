"""AISBench adapter preserving native cache usage without editing installed sources."""
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.registry import MODELS

@MODELS.register_module()
class CacheReportAPIChat(VLLMCustomAPIChat):
    @staticmethod
    def _capture(chunk, output):
        usage = chunk.get('usage')
        if usage:
            if output.time_points and isinstance(usage.get('completion_tokens'), int):
                output.extra_details_data.setdefault('usage_events', []).append(dict(t=output.time_points[-1], completion_tokens=usage['completion_tokens']))
            output.extra_details_data['native_usage'] = usage
        details = (chunk.get('sglext') or {}).get('cached_tokens_details')
        if details is not None:
            output.extra_details_data['cached_tokens_details'] = details

    async def parse_stream_response(self, json_content, output):
        await super().parse_stream_response(json_content, output)
        self._capture(json_content, output)

    async def parse_text_response(self, json_content, output):
        await super().parse_text_response(json_content, output)
        self._capture(json_content, output)
