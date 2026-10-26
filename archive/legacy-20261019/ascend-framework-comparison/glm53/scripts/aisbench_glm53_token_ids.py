"""Project API-body/audit adapter over the frozen, unchanged AISBench client."""
import hashlib
import struct
import time

from ais_bench.benchmark.models.api_models.vllm_custom_api import VLLMCustomAPI


class GLM53TokenIDAPI(VLLMCustomAPI):
    def __init__(self, *args, run_namespace, expected_input_tokens, expected_output_tokens, **kwargs):
        assert run_namespace.startswith('glm53-')
        assert expected_input_tokens > 0 and expected_output_tokens > 0
        self.run_namespace = run_namespace
        self.expected_input_tokens = expected_input_tokens
        self.expected_output_tokens = expected_output_tokens
        # In this AISBench revision range(self.retry) is total attempts.
        # retry=1 therefore means one request, including failure; retry=0 sends none.
        kwargs['retry'] = 1
        super().__init__(*args, **kwargs)

    async def get_request_body(self, input_data, max_out_len, output, **args):
        assert isinstance(input_data, list) and all(type(x) is int and x >= 0 for x in input_data)
        assert len(input_data) == self.expected_input_tokens
        assert max_out_len == self.expected_output_tokens and output.uuid
        body = await super().get_request_body(input_data, max_out_len, output, **args)
        body.update(request_id=output.uuid, cache_salt=self.run_namespace,
                    min_tokens=max_out_len, ignore_eos=True, temperature=0, seed=20261017,
                    return_token_ids=False, skip_special_tokens=False)
        assert body['prompt'] is input_data and body['max_tokens'] == max_out_len
        output.extra_details_data['glm53_request'] = {
            'request_id':output.uuid, 'namespace':self.run_namespace,
            'input_tokens':len(input_data),
            'input_sha256_le_u32':hashlib.sha256(struct.pack('<'+'I'*len(input_data),*input_data)).hexdigest(),
            'output_target':max_out_len, 'native_attempt_setting':self.retry,
        }
        return body

    async def parse_stream_response(self, api_response, output):
        # Native time_points are kept untouched, including empty/usage events.
        # The contract's first valid output must be measured independently.
        now = time.perf_counter()
        choices = api_response.get('choices', [])
        valid_text = any(bool(choice.get('text')) for choice in choices)
        if valid_text:
            output.extra_perf_data.setdefault('glm53_valid_output_chunk_times', []).append(now)
        if api_response.get('usage') is not None:
            output.extra_details_data['glm53_service_usage'] = api_response['usage']
        for choice in choices:
            if choice.get('finish_reason') is not None:
                output.extra_details_data['glm53_finish_reason'] = choice['finish_reason']
        await super().parse_stream_response(api_response, output)
