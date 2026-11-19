from layer_probe_common_g1145 import install,install_native_prefix_guard
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Attention,DeepseekV41Model
install(DeepseekV41Attention,'vllm','hidden_states','3575e46a01ab20d839aa7ac1272307dde89802a7e722b7dd55b35e8d2dad03b1','layer_idx')
class ProbeWorkerExtension:
    pass

install_native_prefix_guard(DeepseekV41Model)
