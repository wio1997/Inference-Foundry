#!/usr/bin/env python3
"""ISOLATED NPU preflight, intentionally not run by source preparation.
Run only with benchmark/service stopped. Uses known-order waits in preflight only.
Produces measured empirical calibration, not a universal timer/scheduling guarantee.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--devices',default='0,1,2,3,4,5,6,7')
    ap.add_argument('--trials',type=int,default=32)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.trials<16:raise ValueError('at least 16 isolated trials required')
    import torch
    import torch_npu
    devices=[int(x) for x in a.devices.split(',')]
    rows=[]; errors=[]; enqueues=[]; uncertainty=[]
    for device in devices:
        torch.npu.set_device(device)
        copy_stream=torch.npu.Stream(device=device)
        writer=torch.npu.Stream(device=device)
        source=torch.empty(12,dtype=torch.int32,device=f'npu:{device}')
        destination=torch.empty(12,dtype=torch.int32,pin_memory=True)
        for trial in range(a.trials):
            source.fill_(trial+1)
            events=[torch.npu.Event(enable_timing=True) for _ in range(6)]
            original=torch.npu.Event()
            def mark(e):
                begin=time.monotonic_ns();e.record();enqueues.append((time.monotonic_ns()-begin)/1e6)
            copy_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(copy_stream):
                mark(events[0]);destination.copy_(source,non_blocking=True)
                original.record();mark(events[1])
            with torch.npu.stream(writer):
                writer.wait_event(events[1])
                mark(events[2]);source.fill_(trial+1000);mark(events[3])
                # Adjacent-event calibration includes lazy event initialization.
                mark(events[4])
            original.synchronize()
            visible=destination.tolist()==[trial+1]*12
            for e in events[:5]:e.synchronize()
            events[5].record();events[5].synchronize()
            direct=events[1].elapsed_time(events[2])
            anchor=events[1].elapsed_time(events[5])-events[2].elapsed_time(events[5])
            adjacent=events[3].elapsed_time(events[4])
            uncertainty.extend([abs(direct-anchor),abs(adjacent)])
            ok=visible and direct>=0 and anchor>=-max(abs(adjacent),1e-6)
            if not ok:errors.append(dict(device=device,trial=trial,visible=visible,direct=direct,anchor=anchor))
            rows.append(dict(device=device,trial=trial,host_visible=visible,direct_ms=direct,
                anchor_ms=anchor,adjacent_event_ms=adjacent,copy_stream=int(copy_stream.npu_stream),
                writer_stream=int(writer.npu_stream)))
    # Empirical conservative allowance from worst observed adjacent-event interval,
    # disagreement and four worst host enqueue intervals. These are not hard bounds.
    u=max(uncertainty+[1e-6])*2
    passed=not errors and len(rows)==len(devices)*a.trials
    try: event_source=inspect.getsource(torch.npu.Event)
    except (OSError,TypeError):event_source=repr(torch.npu.Event)
    report=dict(passed=passed,devices=devices,trials=a.trials,rows=rows,errors=errors,
        producer_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,
        event_api_source=event_source,event_api_source_sha256=hashlib.sha256(event_source.encode()).hexdigest(),
        timing=dict(same_device_cross_stream_validated=passed,d2h_completion_host_visibility_validated=passed,
            lazy_event_preflight_passed=passed,direct_uncertainty_ms=u,anchor_uncertainty_ms=2*u,
            local_marker_allowance_ms=4*max(enqueues+[1e-6])+u),
        limitation='Empirical known-order timing/data calibration only; finite samples do not establish a universal API guarantee or a hard instrumentation bound.')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=passed,devices=devices,trials=a.trials)))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
