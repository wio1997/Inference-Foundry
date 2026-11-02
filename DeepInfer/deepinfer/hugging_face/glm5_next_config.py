"""Device-free GLM-5.3-Flash configuration contract for backend bring-up.

This module parses checkpoint data. It does not register a runnable model,
allocate caches, import torch, or establish 910C operator compatibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Glm5NextLayerSpec:
    layer_id: int
    attention: str
    mlp: str


@dataclass(frozen=True)
class Glm5NextConfig:
    architecture: str
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    dtype: str
    layers: tuple[Glm5NextLayerSpec, ...]
    linear_attn_config: dict[str, Any]
    mla_dimensions: dict[str, int]
    moe_config: dict[str, Any]
    mhc_config: dict[str, Any]
    indexer_config: dict[str, Any]
    num_nextn_predict_layers: int
    vision_config: dict[str, Any] | None
    hf_config: dict[str, Any]
    quantization_method: str | None = None

    @classmethod
    def from_hf_dict(cls, hf: dict[str, Any]) -> Glm5NextConfig:
        if hf.get("model_type") != "glm5_next":
            raise ValueError("Expected model_type=glm5_next")
        architectures = hf.get("architectures")
        if architectures != ["Glm5NextForConditionalGeneration"]:
            raise ValueError(f"Unsupported GLM architecture declaration: {architectures!r}")
        text = hf["text_config"]
        if text.get("model_type") != "glm5_next_text":
            raise ValueError("Expected text_config.model_type=glm5_next_text")
        n_layers = _positive_int(text, "num_hidden_layers")
        attention = text["layer_types"]
        mlp = text["mlp_layer_types"]
        if len(attention) != n_layers or len(mlp) != n_layers:
            raise ValueError("Layer arrays must describe every backbone layer")
        if set(attention) - {"linear_attention", "deepseek_sparse_attention"}:
            raise ValueError("Unknown GLM attention type")
        if set(mlp) - {"dense", "sparse"}:
            raise ValueError("Unknown GLM MLP type")
        linear = dict(text["linear_attn_config"])
        kda = [i for i, kind in enumerate(attention) if kind == "linear_attention"]
        dsa = [i for i, kind in enumerate(attention) if kind == "deepseek_sparse_attention"]
        if linear.get("kda_layers") != kda or linear.get("full_attn_layers") != dsa:
            raise ValueError("KDA/DSA layer IDs disagree with per-layer attention types")
        for name in ("num_heads", "head_dim", "short_conv_kernel_size"):
            _positive_int(linear, name)
        dimensions = {
            name: _positive_int(text, name)
            for name in ("num_attention_heads", "num_key_value_heads", "kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "v_head_dim")
        }
        rope = text["qk_rope_head_dim"]
        if type(rope) is not int or rope < 0:
            raise ValueError("qk_rope_head_dim must be a non-negative integer")
        dimensions["qk_rope_head_dim"] = rope
        moe = {
            key: text[key]
            for key in (
                "n_routed_experts",
                "n_shared_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "routed_scaling_factor",
                "scoring_func",
                "norm_topk_prob",
                "n_group",
                "topk_group",
                "topk_method",
            )
        }
        experts = _positive_int(moe, "n_routed_experts")
        if _positive_int(moe, "num_experts_per_tok") > experts:
            raise ValueError("MoE top-k exceeds the number of routed experts")
        mhc = {key: text[key] for key in ("mhc", "hc_mult", "hc_eps", "hc_sinkhorn_iters")}
        indexer = {key: value for key, value in text.items() if key.startswith("index_") or key.startswith("indexer_")}
        # Keep the original dimensions and extra fields; they are not V4.1
        # compression ratios, SWA cache shapes or inferred operator capabilities.
        original = json.loads(json.dumps(hf))
        return cls(
            architecture=architectures[0],
            model_type=hf["model_type"],
            hidden_size=_positive_int(text, "hidden_size"),
            num_hidden_layers=n_layers,
            vocab_size=_positive_int(text, "vocab_size"),
            dtype=text["dtype"],
            layers=tuple(Glm5NextLayerSpec(i, attention[i], mlp[i]) for i in range(n_layers)),
            linear_attn_config=linear,
            mla_dimensions=dimensions,
            moe_config=moe,
            mhc_config=mhc,
            indexer_config=indexer,
            num_nextn_predict_layers=text.get("num_nextn_predict_layers", 0),
            vision_config=original.get("vision_config"),
            hf_config=original,
        )

    @classmethod
    def from_checkpoint(cls, directory: str | Path) -> Glm5NextConfig:
        from dataclasses import replace

        directory = Path(directory)
        config = cls.from_hf_dict(json.loads((directory / "config.json").read_text()))
        quant_path = directory / "quant_model_description.json"
        if quant_path.is_file():
            quant = json.loads(quant_path.read_text())
            methods = {value for key, value in quant.items() if key.endswith(".weight") and isinstance(value, str) and value != "FLOAT"}
            if methods != {"W8A8_DYNAMIC"}:
                raise ValueError(f"Unimplemented GLM quantization contract: {sorted(methods)!r}")
            config = replace(config, quantization_method="W8A8_DYNAMIC")
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_int(data: dict[str, Any], key: str) -> int:
    value = data[key]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value
