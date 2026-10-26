from ais_bench.benchmark.models import VLLMCustomAPIChatStream
models = [dict(
    attr="service", type=VLLMCustomAPIChatStream,
    abbr="vllm-api-stream-chat", path='/data/tiankuan/aisbench/GLM-5.3-w8a8', model='glm-53',
    api_key="", request_rate=0.0, retry=2,
    host_ip='172.16.10.168', host_port=9300, url="",
    max_out_len=600, batch_size=32,
    generation_kwargs=dict(temperature=0, ignore_eos=True),
)]
