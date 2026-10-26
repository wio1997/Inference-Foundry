# SPDX-License-Identifier: Apache-2.0
"""Observe native PD scheduler exit and close its owned bootstrap HTTP loop."""
import os

def scheduler_exit(scheduler,role):
 from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
 import torch
 if not scheduler.gracefully_exit or os.environ['FLASH_PD_ROLE']!=role:raise ValueError('Unexpected native PD shutdown role/flag')
 append('pd_scheduler_shutdown_seen',role=role,device=torch.npu.current_device(),native_graceful_flag=True)

def close_bootstrap(tokenizer):
 from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
 server=getattr(tokenizer,'bootstrap_server',None);role=os.environ['FLASH_PD_ROLE']
 if server is None:
  if role!='Decode':raise ValueError('Missing owned Prefill bootstrap HTTP server')
  append('pd_bootstrap_closed',role=role,had_server=False,thread_alive=False);return
 if role!='Prefill' or server.host!='0.0.0.0' or server.port!=19032:raise ValueError('Unexpected bootstrap shutdown target')
 server.close()
 alive=server.thread.is_alive();closed=server._loop.is_closed()
 append('pd_bootstrap_closed',role=role,had_server=True,thread_alive=alive,loop_closed=closed,host=server.host,port=server.port)
 if alive or not closed:raise RuntimeError('Native bootstrap HTTP thread did not close within its own join bound')
 tokenizer.bootstrap_server=None
