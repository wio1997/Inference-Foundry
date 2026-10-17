# SPDX-License-Identifier: Apache-2.0
"""Use native stoppable PD receive for explicit Flash graceful teardown only."""
import argparse,ast,hashlib,pathlib


def patch(source):
    old='''        if not self.server_args.enable_pd_role_switch:
            return socket.recv_multipart

        poller = zmq.Poller()
'''
    new='''        import os
        flash_graceful = os.environ.get('FLASH_PD_GRACEFUL_RECV') == '1'
        if not self.server_args.enable_pd_role_switch and not flash_graceful:
            return socket.recv_multipart

        if flash_graceful:
            import hashlib, pathlib
            from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
            append('pd_native_worker_recv_mode', device=self.attn_tp_rank,
                   native_timeout_ms=timeout_ms, flash_graceful=True,
                   role_switch=bool(self.server_args.enable_pd_role_switch),
                   source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
        poller = zmq.Poller()
'''
    if source.count(old)!=1:
        raise ValueError('Wrong native CommonKVManager receive anchor')
    result=source.replace(old,new,1)
    result=result.replace('Plain blocking recv unless role switching is enabled: teardown flips a',
                          'Plain blocking recv unless role switching or explicit Flash graceful\n        receive is enabled: teardown flips a',1)
    result=result.replace('--enable-pd-role-switch keep the original blocking recv and pay nothing.',
                          'either mode keep the original blocking recv and pay nothing.',1)
    ast.parse(result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();b=a.source.read_bytes()
    if hashlib.sha256(b).hexdigest()!=a.expected_sha256:raise ValueError('Wrong pinned native Common source')
    a.output.write_text(patch(b.decode()))
