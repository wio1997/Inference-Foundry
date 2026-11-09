"""图像相关特殊 token 的偏移量定义。

所有图像特殊 token ID 均基于 vocab_size 动态计算:
    token_id = vocab_size + ImageTokenOffset.XXX

各语言中对应的 enum/常量:
    Python: deepinfer.multimodal_tokens.ImageTokenOffset (IntEnum)
    C++:    MMTokenOffset (enum class, sequence.hpp)
"""

from enum import IntEnum

# try these tokens in the tokenizer to get the correct image token
CANDIDATE_IMAGE_PLACEHOLDER_TOKENS: tuple[str, ...] = ("<image>", "<｜deepseek_image｜>")


class ImageTokenOffset(IntEnum):
    """图像特殊 token 相对于 vocab_size 的偏移量。

    实际 token_id = vocab_size + offset
    """

    IMAGE_START = 0  # <image_start>
    IMAGE_PAD = 1  # <image_pad>
    IMAGE = 2  # <image> 实际图像 token
    IMAGE_NEW_LINE = 3  # <image_new_line>
    IMAGE_END = 4  # <image_end>
    NUM_SPECIAL = 5  # 特殊图像 token 数量；用于合法性检查: token_id < vocab_size + NUM_SPECIAL


class TokenType(IntEnum):
    """multi-modal token 类型（mm_token_type 张量取值），与训练侧 processor 逐值一致。"""

    # wire format 不可改值：deepinfer/ve/processor 产出，MoE gate 按 >= IMAGE 判图像类，replace_input_embeds kernel 按值替换 embeds
    UNDEFINED = 0  # 通常 text，以及非固定分辨率的 token
    TEXT = 1
    IMAGE = 2
    IMAGE_START = 3
    IMAGE_END = 4
    IMAGE_NEW_LINE = 5
    IMAGE_PAD = 6
