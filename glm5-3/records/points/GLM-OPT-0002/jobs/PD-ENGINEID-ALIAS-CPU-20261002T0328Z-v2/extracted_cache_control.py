if meta.remote_engine_id not in self.local_remote_block_port_mapping:
    self.local_remote_block_port_mapping[meta.remote_engine_id] = None

if self.local_remote_block_port_mapping[meta.remote_engine_id] is None:
    local_remote_block_port_mappings = get_local_remote_block_port_mappings()
    self.local_remote_block_port_mapping[meta.remote_engine_id] = local_remote_block_port_mappings[
        self.handshake_port
    ]
    self.remote_port_send_num[meta.remote_engine_id] = get_remote_port_send_num(
        local_remote_block_port_mappings
    )

local_remote_block_port_mapping = copy.deepcopy(self.local_remote_block_port_mapping[meta.remote_engine_id])

