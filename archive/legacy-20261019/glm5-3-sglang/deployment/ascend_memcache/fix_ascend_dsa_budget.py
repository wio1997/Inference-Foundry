#!/usr/bin/env python3
"""Pinned-source fix: account for Ascend MLA's unquantized index buffer."""
import argparse
import ast
import hashlib
from pathlib import Path

ORIGINAL_SHA = "2a38f9aeaad230e4b511861738b972ed9bbde507e72dfa579229a6a9cb197fb3"
OLD = """                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // DSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    DSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
"""
NEW = """                # Ascend's NPUMLATokenToKVPool stores raw index_head_dim
                # elements with the main KV dtype, without FP8 scale packing.
                # Keep the GPU/HiSparse packed-index calculation unchanged.
                if (
                    kvc.server_args.attention_backend == "ascend"
                    and not kvc.server_args.enable_hisparse
                ):
                    indexer_size_per_token = index_head_dim
                    element_size = kv_size
                else:
                    indexer_size_per_token = (
                        index_head_dim
                        + index_head_dim // DSATokenToKVPool.quant_block_size * 4
                    )
                    element_size = torch._utils._element_size(
                        DSATokenToKVPool.index_k_with_scale_buffer_dtype
                    )
"""

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    path = Path(args.source)
    raw = path.read_bytes()
    text = raw.decode()
    # Idempotence is verified by reconstructing the exact pinned original.
    if text.count(NEW) == 1:
        original = text.replace(NEW, OLD, 1).encode()
        if digest(original) != ORIGINAL_SHA:
            raise SystemExit('Unknown source around existing budget patch')
        print('ASCEND_DSA_BUDGET_ALREADY_APPLIED sha256=' + digest(raw))
        return
    if digest(raw) != ORIGINAL_SHA or text.count(OLD) != 1:
        raise SystemExit('Source differs from pinned pool_configurator; refusing patch')
    patched = text.replace(OLD, NEW, 1)
    ast.parse(patched)
    out = patched.encode()
    if not args.dry_run:
        backup = path.with_name(path.name + '.before_ascend_dsa_budget')
        if backup.exists() and digest(backup.read_bytes()) != ORIGINAL_SHA:
            raise SystemExit('Backup differs from pinned original')
        if not backup.exists():
            backup.write_bytes(raw)
        path.write_bytes(out)
    print('ASCEND_DSA_BUDGET_' + ('DRY_RUN' if args.dry_run else 'APPLIED')
          + ' sha256=' + digest(out))

if __name__ == '__main__':
    main()
