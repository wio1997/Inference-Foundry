# SPDX-License-Identifier: Apache-2.0
"""Source-only exact pinned native PD transport binding; Store stays native."""
import argparse,ast,difflib,hashlib,json,pathlib

def once(s,a,b):
 if s.count(a)!=1:raise ValueError('Unexpected pinned source anchor: '+a[:80])
 return s.replace(a,b,1)

def patch_worker(s):
 s=once(s,'from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (','from vllm_ascend.distributed.kv_transfer.utils.vllm_flash_sdma_performance_engine import (')
 a='        device_name = str(torch.npu.current_device()) if self.pp_size > 1 else None\n'
 b='''        # Preserve native address generation and all DP/TP/MTP/cache logic.
        # Global DP identity is required: external API rank launchers have
        # local DP size/rank 1/0 even when global DP is 2 or 16.
        if self.pp_size != 1 or self.pcp_size != 1 or self.dcp_size != 1:
            raise ValueError("Flash SDMA candidate requires PP/PCP/DCP1")
        global_dp_size = vllm_config.parallel_config.data_parallel_size
        global_dp_rank = vllm_config.parallel_config.data_parallel_rank
        global_te.configure(
            "Prefill" if self.kv_transfer_config.is_kv_producer else "Decode",
            int(torch.npu.current_device()),
            {"tp_size": self.tp_size, "dp_size": global_dp_size,
             "tp_rank": self.tp_rank, "dp_rank": global_dp_rank,
             "global_device_ordinal": global_dp_rank * self.tp_size + self.tp_rank},
        )
'''+a
 s=once(s,a,b);ast.parse(s);return s

def patch_connector(s):
 a='''    def wait_for_save(self) -> None:
        """MooncakeConnector does not save KV cache explicitly."""
'''
 b='''    def wait_for_save(self) -> None:
        """Publish native producer completion before host scheduler export."""
        if self.connector_worker is not None and self.kv_transfer_config.is_kv_producer:
            self.connector_worker.engine.synchronize_producer_stream()
'''
 s=once(s,a,b);ast.parse(s);return s

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-root',required=True,type=pathlib.Path);ap.add_argument('--output',required=True,type=pathlib.Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);m={}
 for name,fn in [('base_worker.py',patch_worker),('connector.py',patch_connector)]:
  p=a.source_root/'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake'/name;b=p.read_bytes();s=fn(b.decode());(a.output/name).write_text(s);(a.output/(name+'.patch')).write_text(''.join(difflib.unified_diff(b.decode().splitlines(True),s.splitlines(True),fromfile='a/'+name,tofile='b/'+name)));m[name]={'base_sha256':hashlib.sha256(b).hexdigest(),'candidate_sha256':hashlib.sha256(s.encode()).hexdigest()}
 m['status']='SOURCE_ONLY_NOT_ACTUAL_DP_GRAPH_MTP_PD_QUALIFIED';m['store_engine']='Native utils/mooncake_transfer_engine.py unchanged, not replaced by MF shim';m['producer_fence']='One current stream synchronization per actual P forward at KVConnectorMixin post-forward wait_for_save; call-chain/runtime still needs qualification';(a.output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
