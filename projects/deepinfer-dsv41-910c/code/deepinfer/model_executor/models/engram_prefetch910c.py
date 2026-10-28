# SPDX-License-Identifier: Apache-2.0
"""Selected Engram rows fetched ahead of their consuming model layers.

The INT8 tables remain CPU-owned. A single producer reads and decodes the
same rows as the synchronous path, then queues pinned H2D on a copy stream.
The model stream waits only at the consuming layer. No weight conversion,
token lookup substitution or output reuse is involved.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import torch


@dataclass
class EngramRows:
    host_rows: torch.Tensor
    host_valid: torch.Tensor
    rows: torch.Tensor
    valid: torch.Tensor
    copied: object
    read_seconds: float
    bytes: int


class EngramFetchBatch:
    def __init__(self,runtime,hashes):
        self.runtime=runtime
        self.futures={layer:runtime.executor.submit(runtime._fetch,layer,indices)
                      for layer,indices in sorted(hashes.items())}
        self.consumed=set()
        self.consumer_done=None
        runtime.stats['batches']+=1
        runtime.stats['layers_submitted']+=len(self.futures)

    def get(self,layer):
        if layer in self.consumed:
            raise ValueError('Engram fetch must have exactly one consumer')
        started=time.perf_counter()
        packet=self.futures[layer].result()
        self.runtime.stats['host_wait_seconds']+=time.perf_counter()-started
        torch.npu.current_stream().wait_event(packet.copied)
        self.consumed.add(layer)
        self.runtime.stats['layers_consumed']+=1
        self.runtime.stats['cpu_read_seconds']+=packet.read_seconds
        self.runtime.stats['h2d_bytes']+=packet.bytes
        return packet.rows,packet.valid

    def finish(self):
        if self.consumed!=set(self.futures):
            raise ValueError('not every prefetched Engram layer was consumed')
        self.consumer_done=torch.npu.Event()
        self.consumer_done.record()

    def drain(self):
        # Retain host and device buffers until every asynchronous use ends.
        # A batch is retired before the next prefill, normally after the
        # runner's existing global-busy synchronization has already completed.
        for future in self.futures.values():
            future.result().copied.synchronize()
        if self.consumer_done is not None:self.consumer_done.synchronize()


class EngramPrefetch:
    def __init__(self,reader,device):
        self.reader,self.device=reader,device
        self.copy_stream=torch.npu.Stream()
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='EngramFetch')
        self.pending=None
        self.reset_stats()

    @torch.inference_mode()
    def _fetch(self,layer,indices):
        torch.npu.set_device(self.device)
        started=time.perf_counter()
        values=self.reader.engram_rows(layer,indices,'cpu').reshape(indices.shape[0],-1)
        host_rows=values.pin_memory()
        host_valid=(indices>=0).all(-1).pin_memory()
        read_seconds=time.perf_counter()-started
        with torch.npu.stream(self.copy_stream):
            rows=host_rows.to(self.device,non_blocking=True)
            valid=host_valid.to(self.device,non_blocking=True)
            copied=torch.npu.Event()
            copied.record()
        return EngramRows(host_rows,host_valid,rows,valid,copied,read_seconds,
                          host_rows.numel()*host_rows.element_size()+host_valid.numel())

    def begin(self,hashes):
        if self.pending is not None:self.pending.drain()
        self.pending=EngramFetchBatch(self,hashes)
        return self.pending

    def reset_stats(self):
        if getattr(self,'pending',None) is not None:
            self.pending.drain();self.pending=None
        self.stats={'batches':0,'layers_submitted':0,'layers_consumed':0,
                    'cpu_read_seconds':0.,'host_wait_seconds':0.,'h2d_bytes':0,
                    'scope':'selected CPU Engram rows only; independent copy stream, per-layer wait'}

