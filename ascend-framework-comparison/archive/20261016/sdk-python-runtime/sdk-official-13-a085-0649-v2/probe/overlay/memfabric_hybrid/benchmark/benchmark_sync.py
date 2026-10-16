# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Coordinate benchmark ranks without a device collective or clock synchronization."""

import json
import logging
import time
from datetime import timedelta

LOGGER = logging.getLogger(__name__)


class BenchmarkSync:
    def __init__(self, params, rank_id):
        from torch.distributed import TCPStore

        self.rank_id = rank_id
        self.world_size = params["world_size"]
        self.timeout = params["sync_timeout"]
        self.store = TCPStore(
            params["sync_host"],
            params["sync_port"],
            self.world_size,
            rank_id == 0,
            timedelta(seconds=self.timeout),
            wait_for_workers=False,
        )

    def _wait(self, keys, stage):
        deadline = time.monotonic() + self.timeout
        while True:
            if self.store.check(["failure"]):
                raise RuntimeError(self.store.get("failure").decode())
            if self.store.check(keys):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"rank={self.rank_id} timed out at {stage}")
            time.sleep(0.05)

    def barrier(self, stage, ranks=None):
        ranks = range(self.world_size) if ranks is None else ranks
        self.store.set(f"{stage}/{self.rank_id}", "ready")
        self._wait([f"{stage}/{rank}" for rank in ranks], stage)
        # Acknowledge the release so rank 0 cannot destroy the store while a
        # peer is still reading the final barrier keys.
        self.store.set(f"{stage}/ack/{self.rank_id}", "done")
        if self.rank_id == 0:
            self._wait([f"{stage}/ack/{rank}" for rank in ranks], f"{stage} acknowledgements")

    def check_config(self, params):
        fields = (
            "protocol",
            "peer_mem_name",
            "copy_size",
            "batch_size",
            "copy_count",
            "local_hbm_size",
            "local_rank_size",
            "warmup_count",
        )
        signature = {key: params[key] for key in fields}
        signature["direction"] = params["direction"]
        self.store.set(f"config/{self.rank_id}", json.dumps(signature, sort_keys=True))
        self.barrier("configuration ready")
        for rank in range(self.world_size):
            expected = dict(signature)
            expected["direction"] = "write" if rank < params["local_rank_size"] else "read"
            actual = json.loads(self.store.get(f"config/{rank}"))
            if actual != expected:
                raise ValueError(f"paired configuration mismatch at rank={rank}: {actual}, expected={expected}")

    def fail(self, message):
        try:
            self.store.set("failure", f"rank={self.rank_id}: {message}")
        except Exception as error:  # noqa: BLE001 - the store may have failed with the peer
            LOGGER.error("failed to notify peers, rank=%s, error=%s", self.rank_id, error)
