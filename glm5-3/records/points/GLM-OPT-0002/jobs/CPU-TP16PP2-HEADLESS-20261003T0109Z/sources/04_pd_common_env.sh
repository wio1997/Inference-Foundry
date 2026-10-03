#!/bin/bash
# ============================================================================
# GLM-5.2 PD disaggregation — shared environment for both P (166) and D (167).
# Sourced by start_p_166.sh and start_d_167.sh inside the glm52-single container.
# Reference: docs/source/tutorials/models/GLM5.2.md 5.2.1 / 5.2.3
# ----------------------------------------------------------------------------
# NOTE: this file is NEW (task-local). The pre-existing scripts
#       start_single_tp16.sh / start_glm52_single.sh are left untouched.
# ============================================================================

# --- clear variables that could pollute the run -----------------------------
unset VLLM_VERSION
unset LD_PRELOAD
unset VLLM_ASCEND_ENABLE_FLASHCOMM1
unset VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE
unset ASCEND_LAUNCH_BLOCKING
unset VLLM_ASCEND_BALANCE_SCHEDULING
unset VLLM_ASCEND_ENABLE_MLAPO

# --- NPU --------------------------------------------------------------------
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# --- HCCL / host network (GLM5.2.md 5.2.x A3 RoCE deployment) --------------
# PD_LOCAL_IP / PD_NIC are exported by the per-node start script.
export HCCL_IF_IP="${PD_LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${PD_NIC}"
export TP_SOCKET_IFNAME="${PD_NIC}"
export HCCL_SOCKET_IFNAME="${PD_NIC}"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_BUFFSIZE=768

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

# Mooncake abort timeout for long PD transfers (GLM5.2.md 5.2.3)
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480

# Long startup budget (774 GB weights to load)
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# --- shared knobs -----------------------------------------------------------
export PD_MODEL_LEN="${PD_MODEL_LEN:-144384}"
export PD_TP="${PD_TP:-16}"
export PD_DCP="${PD_DCP:-16}"
export PD_MAX_SEQS="${PD_MAX_SEQS:-8}"
export PD_GMU="${PD_GMU:-0.92}"
export PD_MTP_TOKENS="${PD_MTP_TOKENS:-3}"
export PD_KV_PORT="${PD_KV_PORT:-30000}"
export PD_SERVED_NAME="${PD_SERVED_NAME:-glm-52}"
