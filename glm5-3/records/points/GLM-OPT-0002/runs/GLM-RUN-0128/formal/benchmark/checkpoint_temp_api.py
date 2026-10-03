from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="/data/tiankuan/wio/GLM-5.2-w8a8",
        model="glm-52",
        stream=True,
        request_rate=0,
        retry=1,
        host_ip="127.0.0.1",
        host_port=8000,
        url="",
        max_out_len=61440,
        batch_size=2,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True
        )
    )
]
