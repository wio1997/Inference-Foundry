# SPDX-License-Identifier: Apache-2.0
"""TP-head-sharded host Engram lookup without CPU row selection.

Each rank registers only its contiguous hash-head shard. The initial file
mapping route is not DP-shared after CANN's write-pinning causes COW; smaps
evidence must qualify any sharing claim. Shared memfd storage is separate.
The model still receives the original full-head BF16 rows in original order.
"""
import time
import os
import torch
import torch.distributed as dist

from deepinfer.npu_ops.engram_uva910c import FileUvaRows, SharedMemfdRows, lookup


class NativeUvaEngram:
    def __init__(self, reader, history, device, group, progress):
        self.device, self.group = device, group
        self.world = dist.get_world_size(group) if group is not None else 1
        self.rank = dist.get_rank(group) if group is not None else 0
        self.heads = history.offsets.shape[-1]
        if self.heads % self.world:
            raise ValueError('Engram hash heads must divide TP size')
        self.local_heads = self.heads//self.world
        self.head_start = self.rank*self.local_heads
        self.shared=os.environ.get('DEEPINFER_910C_ENGRAM_UVA_STORAGE')=='shared_memfd'
        self.dp_group=None
        if self.shared:
            world=dist.get_world_size()
            if world!=2*self.world:
                raise ValueError('shared memfd path currently requires two local DP replicas')
            # Every rank creates all CPU groups in the same global order.
            for tp_rank in range(self.world):
                ranks=[tp_rank,tp_rank+self.world]
                cpu_group=dist.new_group(ranks,backend='gloo')
                if dist.get_rank() in ranks:self.dp_group=cpu_group
        self.tables = {}
        self.registrations = []
        try:
            for i, layer in enumerate(history.layers):
                begun=time.perf_counter()
                sizes=history.primes[i].flatten()
                first=int(history.offsets[i,self.head_start])
                last=first+int(sizes[self.head_start:self.head_start+self.local_heads].long().sum())
                name=f'layers.{layer}.engram.embed.weight'
                path,start,desc=reader.descriptor(name)
                if desc['dtype']!='I8' or desc['shape'][1]!=256:
                    raise ValueError('UVA requires original signed INT8 width256 codes')
                buffer_class=SharedMemfdRows if self.shared else FileUvaRows
                extra={'group':self.dp_group} if self.shared else {}
                code=buffer_class(path,start+desc['data_offsets'][0],first,last,256,device,**extra)
                self.registrations.append(code)
                path,start,desc=reader.descriptor(name.removesuffix('.weight')+'.scale')
                if desc['dtype']!='F8_E8M0' or desc['shape'][1]!=8:
                    raise ValueError('UVA requires original group32 E8M0 scale bytes')
                scale=buffer_class(path,start+desc['data_offsets'][0],first,last,8,device,**extra)
                self.registrations.append(scale)
                self.tables[layer]=(code,scale,first,last)
                progress({'stage':'engram_uva_registered','layer':layer,'head_start':self.head_start,
                    'local_heads':self.local_heads,'host_bytes':code.bytes+scale.bytes,
                    'seconds':time.perf_counter()-begun})
        except BaseException:
            self.close()
            raise
        self.reset_stats()

    def reset_stats(self):
        self.stats={'calls':0,'tokens':0,'cpu_selected_row_reads':0,'bf16_h2d_bytes':0,
            'host_registered_bytes':sum(x.bytes for x in self.registrations),
            'tp_head_start':self.head_start,'local_heads':self.local_heads,
            'storage':('shared memfd MAP_SHARED, original checkpoint bytes' if self.shared else
                       'read-only checkpoint source; registered MAP_PRIVATE pages duplicate after CANN write-pinning'),
            'scale':'original E8M0 bytes, device decode; no table expansion',
            'scope':'decode lookup before graph entry; not full ZOS/consumer overlap'}

    def rows(self, layer, ids, output):
        code,scale,first,last=self.tables[layer]
        local=lookup(code,scale,ids,vocab_start=first,vocab_end=last,
            head_start=self.head_start,heads=self.local_heads)
        if self.world==1:
            output.copy_(local.reshape_as(output))
        else:
            gathered=local.new_empty((self.world*local.shape[0],self.local_heads,256))
            dist.all_gather_into_tensor(gathered,local,group=self.group)
            output.copy_(gathered.view(self.world,ids.shape[0],self.local_heads,256)
                .permute(1,0,2,3).reshape_as(output))
        self.stats['calls']+=1
        self.stats['tokens']+=ids.shape[0]

    def close(self):
        for registration in reversed(self.registrations):
            registration.close()
        self.registrations=[]
