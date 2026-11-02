"""Vision-language aligner；视觉前向直接调用 aligner.layers(x)。"""

from __future__ import annotations

from torch import nn

from deepinfer import AttrDict


class MlpProjector(nn.Module):
    def __init__(self, cfg: AttrDict) -> None:
        super().__init__()
        self.cfg = cfg
        assert cfg.projector_type == "downsample_mlp_gelu", f"仅支持 downsample_mlp_gelu aligner，got {cfg.projector_type}"

        # 首层入维是 downsample 后的 channel-concat 宽度（batch_downsample 出 [N, input_dim * d * d]）
        modules = [nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.hidden_size)]
        for _ in range(1, cfg.depth - 1):
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.hidden_size, cfg.hidden_size))
        modules.append(nn.GELU())
        modules.append(nn.Linear(cfg.hidden_size, cfg.hidden_size))
        self.layers = nn.Sequential(*modules)
