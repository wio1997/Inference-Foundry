from __future__ import annotations

from pathlib import Path
import json

from deepinfer import AttrDict


def load_model_config(model: str) -> AttrDict:
    """Load a V4.1 runtime config or a GLM checkpoint-only config (AttrDict).

    `model` 必须指向 HF 目录；配置与权重来自同一目录。

    返回的 model_cfg 至少包含：
    - model_cfg.architecture：HF architectures 中声明且通过注册表校验的架构名
    - model_cfg.gpt：模型结构（num_attention_heads / hidden_size / vocab_size / kv_lora_rank ...）
    - model_cfg.vl_aligner：V4.1 HF 视觉/投影结构；GLM is parse-only
    """
    # The GLM branch parses only the checkpoint contract. Registration,
    # hybrid state and execution are separate requirements, not implied here.
    model_path = Path(model)
    hf = json.loads((model_path / "config.json").read_text())
    if hf.get("model_type") == "glm5_next":
        from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig

        glm = Glm5NextConfig.from_checkpoint(model_path)
        if glm.quantization_method != "W8A8_DYNAMIC":
            raise ValueError("GLM-5.3-Flash requires the validated W8A8_DYNAMIC checkpoint")
        return AttrDict(
            architecture=glm.architecture,
            gpt=AttrDict(glm.hf_config["text_config"]),
            glm5_next_config=glm,
            runtime_ready=False,
        )

    # HF 配置解析只处理数据，不导入模型实现。
    from deepinfer.hugging_face import load_hf_model_config

    model_cfg = load_hf_model_config(Path(model) / "config.json")
    assert "vl_aligner" in model_cfg, "Only multimodal HF models with vl_aligner are supported"
    return model_cfg
