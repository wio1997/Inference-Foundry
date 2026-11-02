"""
Test batch invariance of inline VE (ViT + downsample + aligner)

A batch-invariance test that runs only image encoding, not the LLM:
for each image size, generate one fixed random image and use a batch containing just that image as the baseline;
then run two kinds of same-batch tests in which every image's embeds must be bit-identical
(torch.equal) to that image's baseline:
1. single-shape repeat: one batch runs n identical images;
2. shape mix: one batch runs a random mix of images of several shapes.

The 8-GPU DP only parallelizes different image sizes: image encoding is pure single-GPU compute and only VE-side weights are loaded,
so the single-shape test splits image sizes disjointly across ranks by [dp_rank::dp_size],
and the mixed test gives each rank a random combination with a different seed.
"""

from __future__ import annotations

import random
from functools import partial

import numpy as np
import pytest
import torch

from examples._config import DEFAULT_MODEL_NAME, resolve_model_path, setup_cuda_visible_devices
from tests.models._runner import run_model_test

# NOTE: deepinfer-related modules (including examples._prompts, which imports deepinfer) must be loaded
# only after the subprocess's _setup_subprocess_env; names used at runtime are imported lazily inside functions.

# ========================= Config =========================
# Raw image sizes (width, height). Before encoding, compute_image_layout maps them to the best shape,
# and fixed images are generated at the best shape: the CPU resize is then an identity transform and what is compared is batch invariance of the pure GPU path.
# (1, 1) is the mandatory minimum raw size (enlarged by the min_pixels constraint to the smallest best shape).
IMAGE_SIZES: list[tuple[int, int]] = [
    (1, 1),
    (32, 32),
    (256, 256),
    (640, 480),
    (480, 640),
    (512, 1024),
    (1536, 512),
    (1024, 1024),
    (2048, 2048),
]
# images per batch
N_IMAGES_LIST: list[int] = [1, 2, 64, 256]
# images per batch for the mixed-shape test (n=1 has no mixing; a fixed "exactly one per shape" set is also tested)
MIXED_N_IMAGES_LIST: list[int] = [8, 64, 256]
IMAGE_RNG_SEED = 20260722


def _encode_batch(tuner: object, image_processor: object, raw_images: list[tuple[bytes, int, int]]) -> list[torch.Tensor]:
    """Encode one batch of images and return the embeds per image."""
    from examples._ve_helper import build_encode_inputs, encode_images

    inputs = build_encode_inputs(tuner, image_processor, raw_images)
    return list(encode_images(tuner, inputs).split(inputs.num_downsampled))


# ========================= Subprocess entry =========================
def _setup_subprocess_env(model_name: str, dp_rank: int) -> None:
    """Subprocess env setup. Must be called before importing any deepinfer module."""
    setup_cuda_visible_devices(dp_rank, 1, torch.cuda.device_count())


def _run_ve_batch_invariant(model_name: str, dp_size: int, dp_rank: int) -> None:
    _setup_subprocess_env(model_name, dp_rank)

    # _prompts imports deepinfer internally, so it must be loaded after _setup_subprocess_env
    from deepinfer.ve.processor.image_processor import compute_image_layout
    from examples._prompts import random_bmp_bytes
    from examples._ve_helper import load_ve_only

    tuner, image_processor = load_ve_only(resolve_model_path(model_name))

    # ========================= Build fixed images (one per size) =========================
    # Dedupe the full IMAGE_SIZES by best shape first (different raw sizes may map to the same best shape,
    # e.g. all enlarged by min_pixels; a fixed image is determined only by best shape + seed) to get a deduped list
    # identical on every rank, then split it across ranks by [dp_rank::dp_size] for parallel testing.
    best_shapes: dict[tuple[int, int], tuple[bytes, int, int]] = {}  # best shape -> raw_image
    for orig_w, orig_h in IMAGE_SIZES:
        layout = compute_image_layout(
            orig_w,
            orig_h,
            patch_size=image_processor.patch_size,
            downsample_ratio=image_processor.downsample_ratio,
            max_n_token=image_processor.max_n_token,
            min_pixels=image_processor.min_pixels,
            max_wh_ratio=image_processor.max_wh_ratio,
        )
        best_w, best_h = layout.best_width, layout.best_height
        if (best_w, best_h) in best_shapes:
            continue
        seed = IMAGE_RNG_SEED + len(best_shapes)  # index within the deduped list, identical on every rank
        bmp = random_bmp_bytes(np.random.default_rng(seed), best_w, best_h)
        best_shapes[(best_w, best_h)] = (bmp, best_w, best_h)
        if dp_rank == 0:
            print(f"size {(orig_w, orig_h)} -> best shape {(best_w, best_h)}, n_llm = {(layout.n_llm_w, layout.n_llm_h)}", flush=True)

    shape_list = list(best_shapes.keys())
    my_shapes = shape_list[dp_rank::dp_size]
    print(f"[Rank {dp_rank}] testing best shapes: {my_shapes}", flush=True)

    # ========================= Baselines (one image per batch per shape) =========================
    # Every rank computes baselines for all shapes (single-image batches, negligible cost): the single-shape repeat test only uses
    # this rank's shapes, but the mixed test's random combinations may hit any shape.
    baselines = {shape: _encode_batch(tuner, image_processor, [best_shapes[shape]])[0] for shape in shape_list}

    # ========================= Run & compare =========================
    failures: list[str] = []

    def check_batch(shapes: list[tuple[int, int]], tag: str) -> None:
        """Run one batch of the fixed images for shapes and compare each image bitwise against its shape's baseline."""
        n = len(shapes)
        results = _encode_batch(tuner, image_processor, [best_shapes[s] for s in shapes])
        assert len(results) == n
        for img_idx, embeds in enumerate(results):
            ref_embeds = baselines[shapes[img_idx]]
            what = f"[Rank {dp_rank}] {tag}, image {img_idx} (best shape {shapes[img_idx]})"
            if embeds.shape != ref_embeds.shape:
                failures.append(f"{what}: embeds shape mismatch {tuple(embeds.shape)} != {tuple(ref_embeds.shape)}")
                continue
            if not torch.equal(embeds, ref_embeds):
                ne = embeds != ref_embeds
                maxdiff = (embeds.float() - ref_embeds.float()).abs().max().item()
                failures.append(f"{what}: embeds not bitwise equal, {int(ne.sum())}/{ne.numel()} elems differ, maxdiff={maxdiff:.9e}")
        torch.cuda.empty_cache()  # free this batch's activation memory so large n do not accumulate
        print(f"[Rank {dp_rank}] {tag}: checked {n} images", flush=True)

    # ---- single-shape repeat: one batch runs n identical images ----
    for shape in my_shapes:
        for n in N_IMAGES_LIST:
            check_batch([shape] * n, f"single-shape {shape}, n={n}")

    # ---- shape mix: one batch runs a random mix of shapes ----
    # Each rank uses a different seed so the union covers more mixing patterns; fixed seeds keep it reproducible.
    # Test a fixed "exactly one per shape" set (random order) first, then the random mixes for each n.
    rng = random.Random(IMAGE_RNG_SEED + 1000 + dp_rank)
    all_shapes_once = shape_list.copy()
    rng.shuffle(all_shapes_once)
    check_batch(all_shapes_once, "mixed all-shapes-once")
    for n in MIXED_N_IMAGES_LIST:
        check_batch([rng.choice(shape_list) for _ in range(n)], f"mixed n={n}")

    assert not failures, "VE batch invariance failed:\n" + "\n".join(failures)
    print(f"[Rank {dp_rank}] All VE batch invariance checks passed!", flush=True)


# ========================= pytest entry =========================

DP_SIZE = 8  # NOTE only for parallel shape testing; the full case also runs on a single GPU
MODEL_LIST = [DEFAULT_MODEL_NAME]


@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_ve_batch_invariant(model_name: str) -> None:
    run_model_test(partial(_run_ve_batch_invariant, model_name), DP_SIZE)
