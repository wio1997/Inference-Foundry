"""Task control composition for native coupled DP and atomic MQ binding."""
from atomic_mq_worker import AtomicMQWorker
from coupled_dp_metadata_v2 import install_runner_control

class CoupledAtomicMQWorker(AtomicMQWorker):
    def init_device(self):
        super().init_device()
        architectures = getattr(self.vllm_config.model_config.hf_config, "architectures", [])
        if "GlmMoeDsaForCausalLM" not in architectures:
            raise RuntimeError("GLM coupled metadata control requires GlmMoeDsaForCausalLM")
        install_runner_control(self.model_runner)
