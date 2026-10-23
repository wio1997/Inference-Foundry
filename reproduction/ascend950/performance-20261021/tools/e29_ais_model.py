"""Same AISBench clock/routing; permit complete 80K token-ID SSE events."""
import aiohttp

from e27_ais_model import E27MappedAPI
from ais_bench.benchmark.models.api_models.base_api import AIOHTTP_TIMEOUT
from ais_bench.benchmark.registry import MODELS

READ_BUFFER_BYTES = 2 * 1024 * 1024


class BufferedSession:
    """Set a per-response buffer without changing a shared session globally."""

    def __init__(self, session):
        self.session = session

    def post(self, *args, **kwargs):
        kwargs["read_bufsize"] = READ_BUFFER_BYTES
        return self.session.post(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.session, name)


@MODELS.register_module()
class E29MappedAPI(E27MappedAPI):
    async def generate(self, input_data, max_out_len, output, session=None, **args):
        if session is None:
            async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as owned:
                return await super().generate(input_data, max_out_len, output,
                                              session=BufferedSession(owned), **args)
        return await super().generate(input_data, max_out_len, output,
                                      session=BufferedSession(session), **args)
