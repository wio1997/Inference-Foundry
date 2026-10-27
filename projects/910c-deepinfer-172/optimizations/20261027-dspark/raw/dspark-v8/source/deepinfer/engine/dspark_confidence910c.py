# SPDX-License-Identifier: Apache-2.0
"""CPU counterpart of upstream csrc/engine/mtp_varlen.hpp::MTPVarlen.

Native file-IPC serving has synchronous completion, so process/schedule are
paired. Keep the upstream two-round budget lag and one-round prefix-allocation
lag. Current proposal confidence must not select its own verification budget.
Hardware curve entries must be measured on the actual target graphs.
"""
import bisect
import collections
import statistics


class DSparkConfidenceScheduler:
    def __init__(self, curve, requests=4, nextn=5):
        self.curve=sorted((int(size),float(ms)) for size,ms in curve)
        if (not self.curve or any(ms<=0 for _,ms in self.curve)
                or self.curve[-1][0]<requests*(nextn+1)):
            raise ValueError('measured target graph curve must cover the full draft budget')
        self.requests,self.nextn=requests,nextn
        self.accept_rates=None
        self.prev_accept_rates=None
        self.sampler_times=collections.deque(maxlen=9)
        self.forward_samples=collections.deque(maxlen=5000)

    def observe_forward_ms(self,graph_size,elapsed):
        index=bisect.bisect_left([size for size,_ in self.curve],graph_size)
        baseline=self.curve[index][1]
        self.forward_samples.append((graph_size,(elapsed-baseline)/graph_size))

    def observe_confidence(self, conditional):
        if len(conditional)!=self.requests or any(len(row)!=self.nextn for row in conditional):
            raise ValueError('one confidence value per request/draft position required')
        marginal=[]
        for row in conditional:
            cumulative=1.
            values=[]
            for rate in row:
                if not 0<=rate<=1:raise ValueError('invalid learned acceptance confidence')
                cumulative*=rate
                values.append(cumulative)
            marginal.append(values)
        self.prev_accept_rates=self.accept_rates
        self.accept_rates=marginal

    def observe_sampler_ms(self, elapsed):
        if elapsed>=0:self.sampler_times.append(float(elapsed))

    def schedule(self):
        count,n=self.requests,self.nextn
        default=[[1.]*n for _ in range(count)]
        budget_rates=self.prev_accept_rates or default
        allocation_rates=self.accept_rates or default
        budget_queue=sorted(((rate,seq) for seq,row in enumerate(budget_rates)
                             for rate in row),key=lambda x:(-x[0],x[1]))
        allocation_queue=sorted(((rate,seq) for seq,row in enumerate(allocation_rates)
                                 for rate in row),key=lambda x:(-x[0],x[1]))
        sizes=[size for size,_ in self.curve]
        sampling=statistics.median(self.sampler_times) if len(self.sampler_times)>=5 else 0.
        groups=collections.defaultdict(list)
        for size,residual in self.forward_samples:groups[size].append(residual)
        medians=[statistics.median(values) for values in groups.values() if len(values)>=10]
        slope=max(0.,statistics.median(medians)) if medians else 0.
        expected=float(count)
        best_throughput,budget=-1.,count
        for total in range(count,count*(n+1)+1):
            if total>count:expected+=budget_queue[total-count-1][0]
            index=bisect.bisect_left(sizes,total)
            elapsed=self.curve[index][1]+slope*self.curve[index][0]+sampling
            throughput=expected/elapsed
            if throughput>best_throughput:
                best_throughput,budget=throughput,total
        lengths=[1]*count
        for _,seq in allocation_queue[:budget-count]:lengths[seq]+=1
        return lengths
