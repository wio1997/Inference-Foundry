"""Explicit native --worker-cls entrypoint for coupled DP metadata prototype."""
from vllm_ascend.worker.worker import NPUWorker
from coupled_dp_metadata import install_runner_control

class CoupledMetadataWorker(NPUWorker):
    def init_device(self):
        super().init_device()
        architectures=getattr(self.vllm_config.model_config.hf_config,"architectures",[])
        if "GlmMoeDsaForCausalLM" not in architectures:
            raise RuntimeError("GLM coupled metadata prototype requires GlmMoeDsaForCausalLM")
        install_runner_control(self.model_runner)
