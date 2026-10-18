#!/usr/bin/env python3
"""Read source identity only. No framework imports, NPU access or live hooks."""
import argparse
import hashlib
import json
from pathlib import Path


def inspect(root, manifest):
    rows = []
    for expected in manifest:
        relative = Path(expected["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe source path")
        path = root / relative
        sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        rows.append({"path": str(relative), "expected_sha256": expected["sha256"],
                     "actual_sha256": sha, "matched": sha == expected["sha256"]})
    return {"source_identity_matches": bool(rows) and all(r["matched"] for r in rows),
            "files": rows, "actual_proposer_object_type": None,
            "device_run": False, "capture_ready": False,
            "scope": "File bytes only; a Mac fixture is not an image observation. No model state or completion event inspected."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/vllm-workspace"))
    args = parser.parse_args()
    # Embedded to keep the existing collector's single frozen script interface.
    result = inspect(args.root, EXPECTED)
    print("GLM53_RESULT_JSON_BEGIN\n" + json.dumps(result, indent=2) + "\nGLM53_RESULT_JSON_END")
    return 0 if result["source_identity_matches"] else 2


EXPECTED = [
    {
        "path": "vllm/vllm/v1/worker/gpu_model_runner.py",
        "sha256": "87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51"
    },
    {
        "path": "vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
        "sha256": "568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c"
    },
    {
        "path": "vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py",
        "sha256": "ee50f5a0f41c5569ee6ef2af7018bd84aaea943442b2073889fb1e43738b1697"
    },
    {
        "path": "vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py",
        "sha256": "42e6dc76ae5cc59cb6d15ac0d428e62689c2a500289f4b625e847abb4b95f91c"
    },
    {
        "path": "vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py",
        "sha256": "107b0da0ffa228773441631d610bcad5366114c734776dcba3285e90536a67eb"
    },
    {
        "path": "vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_scheduler.py",
        "sha256": "984878c2d8d41a702640f77d466079f9c42147fd626e21f89d58894c05dc812e"
    },
    {
        "path": "vllm-ascend/vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py",
        "sha256": "34948cb9796b7018295edf249e67600c49de5945febed1de52fe3b1ce55725cf"
    },
    {
        "path": "vllm-ascend/vllm_ascend/spec_decode/__init__.py",
        "sha256": "a3b72c62d683a4e58916c976fdac1c6cd2aa87c670d87aa1b43c4a9bfc5ba9a2"
    },
    {
        "path": "vllm/vllm/config/speculative.py",
        "sha256": "9c642a87264339c0dbc1d0b57330572149df0f9d5e16c6893b9895e50a619e19"
    },
    {
        "path": "vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py",
        "sha256": "aa5994f69f3050dd0a7cd374676857ddb787153083d7f6f9b3566cd3e4dc3474"
    },
    {
        "path": "vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py",
        "sha256": "458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31"
    },
    {
        "path": "vllm/vllm/model_executor/models/deepseek_mtp.py",
        "sha256": "d8b7989105c2b24bd83965e45d49758eec8346ac5e449f95a6995c531ef2f8f5"
    },
    {
        "path": "vllm/vllm/v1/spec_decode/llm_base_proposer.py",
        "sha256": "5bd11c41f0a8cf1272b5880d9878043f876bee6c7e04fb781e613f92930b15ab"
    },
    {
        "path": "vllm-ascend/vllm_ascend/models/__init__.py",
        "sha256": "c54c54965a53a33ec686590fe4e4898cff8ebb88fd2155b7d9b95740686eafaa"
    },
    {
        "path": "vllm-ascend/vllm_ascend/models/deepseek_mtp.py",
        "sha256": "4ed40eae69848d1fac7c901a1c73b0532a71473a045af1aa79508f9c74be0a2b"
    },
    {
        "path": "vllm/vllm/v1/executor/multiproc_executor.py",
        "sha256": "3ed0149a8a8e1550c4f062af40e645d1b734e70f99981bc5bf15f0360507648a"
    },
    {
        "path": "vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py",
        "sha256": "e1cc76d065a7e60aacddb190ae795a480b65cd02ba71a1fc699a02a85c1a3de2"
    },
    {
        "path": "vllm-ascend/vllm_ascend/attention/sfa_v1.py",
        "sha256": "9b94f67f30364d40caf4a7a67af0f0365d1ba0b5048939873ab0464e95b096dc"
    }
]

if __name__ == "__main__":
    raise SystemExit(main())
