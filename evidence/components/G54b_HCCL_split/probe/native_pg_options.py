import os
_is_npu=True
def get_torch_distributed_pg_options(group_name=None):
    if not _is_npu:
        return None

    # Only create HCCL options for default group or MoE-related groups
    if group_name is not None and "moe" not in group_name:
        return None

    import torch_npu

    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    hccl_buffer_size = int(
        os.environ.get("DEEPEP_HCCL_BUFFSIZE") or os.environ.get("HCCL_BUFFSIZE") or 200
    )
    options.hccl_config = {"hccl_buffer_size": hccl_buffer_size}
    return options
