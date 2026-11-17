/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_OPS_TRANSFORMER_CUSTOM_H_
#define ACLNN_OPS_TRANSFORMER_CUSTOM_H_

#include "aclnn_gmm_dequant_situ_quant.h"
#include "aclnn_mla_preprocess.h"
#include "aclnn_add_rms_norm_bias.h"
#include "aclnn_copy_and_expand_eagle_inputs.h"
#include "aclnn_dequant_situ_quant.h"
#include "aclnn_dequant_swiglu_quant.h"
#include "aclnn_dequant_swiglu_quant_v2.h"
#include "aclnn_hc_post.h"
#include "aclnn_hc_pre.h"
#include "aclnn_moe_gating_top_k.h"
#include "aclnn_moe_gating_top_k_hash.h"
#include "aclnn_rms_norm_cast.h"
#include "aclnn_transpose_kv_cache_by_block.h"
#include "aclnn_compressor.h"
#include "aclnn_compressor_metadata.h"
#include "aclnn_fused_quant_lightning_indexer_manage.h"
#include "aclnn_fused_sparse_attention_overlap.h"
#include "aclnn_inplace_partial_rotary_mul.h"
#include "aclnn_k2q_csr_hist.h"
#include "aclnn_k2q_csr_meta.h"
#include "aclnn_k2q_csr_row_prefix.h"
#include "aclnn_k2q_csr_scatter.h"
#include "aclnn_k2q_csr_tile_prefix.h"
#include "aclnn_kv_quant_sparse_flash_attention_vllm.h"
#include "aclnn_lightning_indexer_quant.h"
#include "aclnn_msa_index_score.h"
#include "aclnn_rms_norm_dynamic_quant.h"
#include "aclnn_sparse_attn_sharedkv.h"
#include "aclnn_sparse_flash_mla.h"
#include "aclnn_store_kv_block.h"
#include "aclnnInner_dispatch_ffn_combine.h"
#include "aclnnInner_dispatch_ffn_combine_bf16.h"
#include "aclnnInner_dispatch_ffn_combine_w4_a8.h"
#include "aclnnInner_fused_lightning_indexer_manage.h"
#include "aclnnInner_fused_scatter_copy_sparse_flash_attention.h"
#include "aclnnInner_sparse_flash_attention.h"
#include "aclnn_grouped_matmul_swiglu_quant.h"
#include "aclnn_grouped_matmul_swiglu_quant_weight_nz.h"
#include "aclnn_grouped_matmul_swiglu_quant_v2.h"
#include "aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "aclnn_grouped_matmul_swiglu_quant_weight_nz_tensor_list.h"
#include "aclnn_dispatch_ffn_combine.h"
#include "aclnn_dispatch_ffn_combine_bf16.h"
#include "aclnn_dispatch_ffn_combine_w4_a8.h"
#include "aclnn_chunk_fwd_o_vllm.h"
#include "aclnn_chunk_gated_delta_rule_fwd_h_vllm.h"
#include "aclnn_scatter_nd_update_sk.h"
#include "aclnn_attn_res_fwd.h"
#include "aclnn_fused_lightning_indexer_manage.h"
#include "aclnn_fused_scatter_copy_sparse_flash_attention.h"
#include "aclnn_kda_gate_cumsum_vllm.h"
#include "aclnn_kda_layout_swap12.h"
#include "aclnn_quant_lightning_indexer_v2_metadata.h"
#include "aclnn_sparse_attention_score.h"
#include "aclnn_sparse_attn_sharedkv_metadata.h"
#include "aclnn_sparse_flash_attention.h"
#include "aclnn_sparse_flash_mla_metadata.h"
#include "aclnn_store_kv_block_metadata.h"

#endif // ACLNN_OPS_TRANSFORMER_CUSTOM_H_
