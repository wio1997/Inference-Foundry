from types import SimpleNamespace

import numpy as np

from deepinfer.ve.multimodal_processor import MultimodalProcessor
from deepinfer.ve.processor.common import ImageInput, TextLengthInput

IMAGE_TOKEN_ID = 100
IMAGE_TOKEN = 1001
IMAGE_PAD_TOKEN = 1002


def _processor(tokens_per_image: dict[tuple[int, int], int]) -> MultimodalProcessor:
    """A bare processor without workers: tokens per image are looked up from a (width, height) table instead of the real estimate_num_tokens."""
    processor = object.__new__(MultimodalProcessor)
    processor.mm_config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
    processor.mm_image_token = IMAGE_TOKEN
    processor.mm_image_pad_token = IMAGE_PAD_TOKEN

    def estimate_num_tokens(image: ImageInput) -> int:
        return tokens_per_image[(image.width, image.height)]

    processor.mm_processor = SimpleNamespace(image_processor=SimpleNamespace(estimate_num_tokens=estimate_num_tokens))
    return processor


def _seg_spec(segments: list) -> list[tuple]:
    """Type + key-value summary of a segment: text segments by .length, image segments with (w, h, whether binary was stripped)."""
    spec = []
    for segment in segments:
        if isinstance(segment, TextLengthInput):
            spec.append(("TextLengthInput", segment.length))
        elif isinstance(segment, ImageInput):
            spec.append(("ImageInput", segment.width, segment.height, segment.binary is None))
        else:
            spec.append((type(segment).__name__,))
    return spec


def test_overwrite_expands_placeholders_and_builds_segments() -> None:
    processor = _processor({(10, 20): 3, (30, 40): 2})
    token_ids = np.array([1, IMAGE_TOKEN_ID, 2, IMAGE_TOKEN_ID, 3], dtype=np.int32)
    multi_modal_data = {"image_sizes": [(10, 20), (30, 40)], "image_bytes": [b"image-0", b"image-1"]}

    result = processor.overwrite_prompt(token_ids, multi_modal_data)

    assert result is not token_ids  # always copied; never mutates the caller's data in place
    assert token_ids.tolist() == [1, IMAGE_TOKEN_ID, 2, IMAGE_TOKEN_ID, 3]
    assert result.dtype == np.int32
    assert result.tolist() == [1, IMAGE_TOKEN, IMAGE_PAD_TOKEN, IMAGE_PAD_TOKEN, 2, IMAGE_TOKEN, IMAGE_PAD_TOKEN, 3]
    # the raw image keys have moved into segment_input
    assert "image_sizes" not in multi_modal_data and "image_bytes" not in multi_modal_data
    segment_input = multi_modal_data["segment_input"]
    assert segment_input.meta["image_spans"] == [(1, 3), (5, 2)]
    assert _seg_spec(segment_input.segments) == [
        ("TextLengthInput", 1),
        ("ImageInput", 10, 20, False),
        ("TextLengthInput", 1),
        ("ImageInput", 30, 40, False),
        ("TextLengthInput", 1),
    ]
    image_segments = [segment for segment in segment_input.segments if isinstance(segment, ImageInput)]
    assert [segment.binary for segment in image_segments] == [b"image-0", b"image-1"]


def test_overwrite_text_only_prompt_keeps_tokens() -> None:
    processor = _processor({})
    token_ids = np.array([1, 2, 3], dtype=np.int32)
    multi_modal_data = {"image_sizes": [], "image_bytes": []}

    result = processor.overwrite_prompt(token_ids, multi_modal_data)

    assert result is not token_ids
    assert result.tolist() == [1, 2, 3]
    segment_input = multi_modal_data["segment_input"]
    assert segment_input.meta["image_spans"] == []
    assert _seg_spec(segment_input.segments) == [("TextLengthInput", 3)]


def test_overwrite_adjacent_and_leading_images() -> None:
    # the first token is an image and two images are adjacent: no text segment between them, so the spans are contiguous
    processor = _processor({(10, 20): 2, (30, 40): 3})
    token_ids = np.array([IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 5], dtype=np.int32)
    multi_modal_data = {"image_sizes": [(10, 20), (30, 40)], "image_bytes": [b"a", b"b"]}

    result = processor.overwrite_prompt(token_ids, multi_modal_data)

    assert result.tolist() == [IMAGE_TOKEN, IMAGE_PAD_TOKEN, IMAGE_TOKEN, IMAGE_PAD_TOKEN, IMAGE_PAD_TOKEN, 5]
    segment_input = multi_modal_data["segment_input"]
    assert segment_input.meta["image_spans"] == [(0, 2), (2, 3)]
    assert _seg_spec(segment_input.segments) == [
        ("ImageInput", 10, 20, False),
        ("ImageInput", 30, 40, False),
        ("TextLengthInput", 1),
    ]


def test_overwrite_overflow_placeholder_stays_text() -> None:
    # placeholder count (2) > image count (1): surplus placeholders stay as ordinary tokens — not expanded and no image is built
    processor = _processor({(10, 20): 2})
    token_ids = np.array([1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2], dtype=np.int32)
    multi_modal_data = {"image_sizes": [(10, 20)], "image_bytes": [b"a"]}

    result = processor.overwrite_prompt(token_ids, multi_modal_data)

    assert result.tolist() == [1, IMAGE_TOKEN, IMAGE_PAD_TOKEN, IMAGE_TOKEN_ID, 2]
    segment_input = multi_modal_data["segment_input"]
    assert segment_input.meta["image_spans"] == [(1, 2)]
    assert _seg_spec(segment_input.segments) == [
        ("TextLengthInput", 1),
        ("ImageInput", 10, 20, False),
        ("TextLengthInput", 2),
    ]


def test_overwrite_without_image_data_keeps_sizes_only() -> None:
    # only image_sizes are given without pixels (the VE side fetches images separately): still expanded by size, with binary = None
    processor = _processor({(10, 20): 2})
    token_ids = np.array([1, IMAGE_TOKEN_ID], dtype=np.int32)
    multi_modal_data = {"image_sizes": [(10, 20)]}

    result = processor.overwrite_prompt(token_ids, multi_modal_data)

    assert result.tolist() == [1, IMAGE_TOKEN, IMAGE_PAD_TOKEN]
    assert _seg_spec(multi_modal_data["segment_input"].segments) == [("TextLengthInput", 1), ("ImageInput", 10, 20, True)]


def test_add_strips_fully_cached_image_binary() -> None:
    processor = _processor({(10, 20): 3, (30, 40): 2})
    token_ids = np.array([1, IMAGE_TOKEN_ID, 2, IMAGE_TOKEN_ID, 3], dtype=np.int32)
    multi_modal_data = {"image_sizes": [(10, 20), (30, 40)], "image_bytes": [b"cached", b"uncached"]}
    processor.overwrite_prompt(token_ids, multi_modal_data)

    # expanded sequence [1, I, P, P, 2, I, P, 3]: the first 5 tokens hit the cache and cover the first image entirely
    processor._compute_and_strip_cached_images(multi_modal_data, num_cached_tokens=5)

    assert multi_modal_data["_num_cached_images"] == 1
    image_segments = [segment for segment in multi_modal_data["segment_input"].segments if isinstance(segment, ImageInput)]
    assert [segment.binary for segment in image_segments] == [None, b"uncached"]


def test_add_does_not_strip_partially_cached_image() -> None:
    processor = _processor({(10, 20): 3})
    token_ids = np.array([1, IMAGE_TOKEN_ID, 2], dtype=np.int32)
    multi_modal_data = {"image_sizes": [(10, 20)], "image_bytes": [b"partial"]}
    processor.overwrite_prompt(token_ids, multi_modal_data)

    # the cache boundary falls inside the image (the 2nd token), so the image is not counted as cached
    processor._compute_and_strip_cached_images(multi_modal_data, num_cached_tokens=2)

    assert "_num_cached_images" not in multi_modal_data
    image_segments = [segment for segment in multi_modal_data["segment_input"].segments if isinstance(segment, ImageInput)]
    assert [segment.binary for segment in image_segments] == [b"partial"]
