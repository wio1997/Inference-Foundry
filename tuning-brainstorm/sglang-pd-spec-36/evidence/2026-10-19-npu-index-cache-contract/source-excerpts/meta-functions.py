# Source excerpts for review, not a runtime registration module.
# From installed torch_npu 2.10.0.post6; original function bodies preserved.
# See native-interface-index.json for full-file hash and provenance.

# Installed source lines 2322-2366
def npu_lightning_indexer_forward(query, key, weights, *, actual_seq_lengths_query=None,
    actual_seq_lengths_key=None, block_table=None, layout_query="BSND", layout_key="BSND", sparse_count=2048, sparse_mode=3,
    pre_tokens=9223372036854775807, next_tokens=9223372036854775807, return_value=False):
    require_param = {"query": query, "key": key, "weights": weights}

    for item_name, item in require_param.items():
        torch._check(
            item is not None,
            lambda: item_name + " should not be None, but the actual value is None" + ops_error(ErrCode.VALUE),
        )

    torch._check(
        query.numel() > 0,
        lambda: "Input query should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        key.numel() > 0,
        lambda: "Input key should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        sparse_count > 0,
        lambda: "sparse_count should be greater than 0, but got " + str(sparse_count) +
            ops_error(ErrCode.VALUE),
    )

    if layout_query == "BSND":
        sparse_indices_out = torch.empty([query.size(0), query.size(1), key.size(2), sparse_count], dtype=torch.int32, device='meta')
    else:
        if layout_key == "TND":
            n_dim_idx = 1
        else:
            n_dim_idx = 2
        sparse_indices_out = torch.empty([query.size(0), key.size(n_dim_idx), sparse_count], dtype=torch.int32, device='meta')
    if return_value:
        if layout_query == "BSND":
            sparse_values_out = torch.empty([query.size(0), query.size(1), key.size(2), sparse_count], dtype=query.dtype, device='meta')
        else:
            if layout_key == "TND":
                n_dim_idx = 1
            else:
                n_dim_idx = 2
            sparse_values_out = torch.empty([query.size(0), key.size(n_dim_idx), sparse_count], dtype=query.dtype, device='meta')
    else:
        sparse_values_out = torch.empty([0], dtype=query.dtype, device='meta')
    return (sparse_indices_out, sparse_values_out)

# Installed source lines 2060-2099
def npu_quant_lightning_indexer_forward(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode, *, actual_seq_lengths_query=None,
                                        actual_seq_lengths_key=None, block_table=None, layout_query="BSND", layout_key="BSND", sparse_count=2048, sparse_mode=3,
                                        pre_tokens=9223372036854775807, next_tokens=9223372036854775807, query_dtype=None, key_dtype=None):
    require_param = {"query": query, "key": key, "weights": weights, "query_dequant_scale": query_dequant_scale, "key_dequant_scale": key_dequant_scale}

    for item_name, item in require_param.items():
        torch._check(
            item is not None,
            lambda: item_name + " should not be None, but the actual value is None" + ops_error(ErrCode.VALUE),
        )

    torch._check(
        query.numel() > 0,
        lambda: "Input query should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        key.numel() > 0,
        lambda: "Input key should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        sparse_count > 0,
        lambda: "sparse_count should be greater than 0, but got " + str(sparse_count) +
            ops_error(ErrCode.VALUE),
    )

    if layout_key == "TND":
        keyHeadNum = key.size(1)
    else:
        keyHeadNum = key.size(2)
    if layout_query == "BSND":
        out = torch.empty([query.size(0), query.size(1), keyHeadNum, sparse_count], dtype=torch.int32, device='meta')
    elif layout_query == "TND":
        out = torch.empty([query.size(0), keyHeadNum, sparse_count], dtype=torch.int32, device='meta')
    else:
        torch._check(
            False,
            lambda: "No support of query: " + str(layout_query) + ops_error(ErrCode.VALUE),
        )

    return out
