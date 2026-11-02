import struct

import numpy as np
import pytest
import torch  # noqa: F401  Load torch libraries before the C++ extension.

from deepinfer._C.engine import Sequence, encode_compressed
from deepinfer.sampling_params import SamplingParams


def sequence_kwargs() -> dict[str, object]:
    sampling_params = SamplingParams(
        max_tokens=4,
        temperature=0.7,
        top_k=8,
        ignore_eos=True,
        prompt_logprobs=True,
    )
    sampling_params.eos_id = 2
    return {
        "seq_id": 7,
        "multi_modal_data": None,
        "block_size": 16,
        "request_id": "request",
        "user_id": "user",
        "sampling_params": sampling_params,
        "arrival_time": 1.5,
        "golden_token_ids": [9, 10],
    }


def test_sequence_from_token_array() -> None:
    # extra_tokens are concatenated on the Python side
    prompt = np.asarray([1, 2, 3], dtype=np.int32)
    seq = Sequence(
        prompt_token_ids=prompt,
        compressed_prompt_token_ids=prompt,
        **sequence_kwargs(),
    )

    assert seq.data.get_token_ids() == [1, 2, 3]
    assert seq.compressed_data.get_token_ids() == [1, 2, 3]


def test_sequence_from_token_array_compressed() -> None:
    # the compressed array is computed on the Python side (CompressedTokenizer.encode) and passed in:
    # in-table ids map through the lookup (×2), out-of-table ids (image token 100) get the -1 sentinel
    seq = Sequence(
        prompt_token_ids=np.asarray([1, 2, 3, 100], dtype=np.int32),
        compressed_prompt_token_ids=np.asarray([2, 4, 6, -1], dtype=np.int32),
        **sequence_kwargs(),
    )

    assert seq.data.get_token_ids() == [1, 2, 3, 100]
    assert seq.compressed_data.get_token_ids() == [2, 4, 6, -1]


def test_sequence_constructor_rejects_nonconforming_arrays() -> None:
    # construction contract: token inputs accept only int32, C-contiguous ndarrays (noconvert + check_),
    # non-int32 / non-contiguous / list inputs always raise TypeError (no silent conversion)
    src = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int32)
    strided = src[::2]  # not C-contiguous
    assert not strided.flags.c_contiguous
    for bad in (
        strided,  # non-contiguous
        np.asarray([1, 2, 3], dtype=np.int64),  # wrong dtype
        [1, 2, 3],  # a list is not an ndarray
    ):
        with pytest.raises(TypeError):
            Sequence(
                prompt_token_ids=bad,
                compressed_prompt_token_ids=np.asarray([1, 3, 5], dtype=np.int32),
                **sequence_kwargs(),
            )
    # the compressed array is equally strict
    with pytest.raises(TypeError):
        Sequence(
            prompt_token_ids=np.asarray([1, 2, 3], dtype=np.int32),
            compressed_prompt_token_ids=np.asarray([1, 2, 3], dtype=np.int64),
            **sequence_kwargs(),
        )
    # valid inputs are unaffected (int32, C-contiguous, including frombuffer views)
    ok = Sequence(
        prompt_token_ids=src,
        compressed_prompt_token_ids=np.frombuffer(np.asarray([1, 2, 3, 4, 5, 6], np.int32).tobytes(), np.int32),
        **sequence_kwargs(),
    )
    assert ok.data.get_token_ids() == [1, 2, 3, 4, 5, 6]


def test_sequence_from_token_array_frombuffer_view() -> None:
    prompt = np.frombuffer(struct.pack("3i", 1, 2, 3), dtype=np.int32)
    seq = Sequence(
        prompt_token_ids=prompt,
        compressed_prompt_token_ids=prompt,
        **sequence_kwargs(),
    )
    assert seq.data.get_token_ids() == [1, 2, 3]


def test_sequence_from_token_array_empty_prompt() -> None:
    prompt = np.empty(0, dtype=np.int32)
    seq = Sequence(
        prompt_token_ids=prompt,
        compressed_prompt_token_ids=prompt,
        **sequence_kwargs(),
    )
    assert seq.get_prompt_len() == 0
    assert seq.data.get_token_ids() == []


LOOKUP_TABLE = np.asarray([10, 20, 30, 40], dtype=np.int32)


def test_encode_compressed_maps_in_range() -> None:
    out = encode_compressed(np.asarray([0, 3, 1], dtype=np.int32), LOOKUP_TABLE, -1)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, np.asarray([10, 40, 20], dtype=np.int32))


def test_encode_compressed_out_of_range_and_negative_to_sentinel() -> None:
    out = encode_compressed(np.asarray([4, -1, 100], dtype=np.int32), LOOKUP_TABLE, -7)
    np.testing.assert_array_equal(out, np.asarray([-7, -7, -7], dtype=np.int32))


def test_encode_compressed_empty_input() -> None:
    out = encode_compressed(np.empty(0, dtype=np.int32), LOOKUP_TABLE, -1)
    assert out.shape == (0,)
    assert out.dtype == np.int32


def test_encode_compressed_output_is_fresh_array() -> None:
    src = np.asarray([1, 2], dtype=np.int32)
    out = encode_compressed(src, LOOKUP_TABLE, -1)
    out[0] = 123
    np.testing.assert_array_equal(src, np.asarray([1, 2], dtype=np.int32))


@pytest.mark.parametrize(
    "bad",
    [
        [1, 2, 3],  # a list is not an ndarray
        np.asarray([1, 2, 3], dtype=np.int64),  # wrong dtype
        np.asarray([1, 2, 3, 4], dtype=np.int32)[::2],  # not C-contiguous
    ],
)
def test_encode_compressed_rejects_nonconforming_token_ids(bad: object) -> None:
    with pytest.raises(TypeError):
        encode_compressed(bad, LOOKUP_TABLE, -1)


def test_encode_compressed_rejects_2d_token_ids() -> None:
    with pytest.raises(ValueError):
        encode_compressed(np.zeros((2, 2), dtype=np.int32), LOOKUP_TABLE, -1)


def test_encode_compressed_rejects_bad_lookup_table() -> None:
    with pytest.raises(TypeError):
        encode_compressed(np.asarray([1], dtype=np.int32), np.asarray([1], dtype=np.int64), -1)
