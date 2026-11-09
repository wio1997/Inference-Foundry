# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostic comparison of real DSpark greedy proposals.

This deliberately does not drive model outputs, scheduling, or state commits.
It introduces synchronization and must be disabled for performance benchmarks.
"""
import os
import traceback


class GreedyShadow:
    def __init__(self):
        try:
            limit = int(os.environ.get('DEEPINFER_910C_GREEDY_SHADOW_LIMIT', '3'))
            if not 1 <= limit <= 16:
                raise ValueError('shadow limit must be 1..16')
            error = None
        except ValueError as exc:
            limit, error = 0, str(exc)
        self.stats = {'enabled': True, 'limit': limit, 'calls': 0, 'passes': 0,
                      'failures': 0, 'skipped_non_full': 0, 'skipped_eos': 0,
                      'skipped_nonformal': 0, 'config_error': error, 'details': []}

    def wants(self, lengths, requests, variable):
        if self.stats['calls'] >= self.stats['limit']:
            return False
        if not variable:
            self.stats['skipped_nonformal'] += 1
            return False
        if len(lengths) != 4 or list(lengths) != [6] * 4:
            self.stats['skipped_non_full'] += 1
            return False
        if len(requests) != 4 or not all(req.get('ignore_eos', False) for _, req in requests):
            self.stats['skipped_eos'] += 1
            return False
        return True

    def compare(self, proposal_device, target_device, proposal_cpu, target_cpu,
                remaining, finished, accepted, commits, emitted):
        self.stats['calls'] += 1
        call = self.stats['calls']
        try:
            import torch
            from deepinfer.npu_ops.greedy_prefix910c import GreedyPrefixBuffers, verify_greedy_prefix

            device = target_device.device
            draft = proposal_device.to(dtype=torch.int32).contiguous()
            target = target_device.reshape(4, 6).to(dtype=torch.int32).contiguous()
            lengths_t = torch.full((4,), 6, dtype=torch.int32, device=device)
            remaining_t = torch.tensor(remaining, dtype=torch.int32, device=device)
            finished_t = torch.tensor(finished, dtype=torch.bool, device=device)
            buffers = GreedyPrefixBuffers.allocate(device)
            verify_greedy_prefix(draft, target, lengths_t, remaining_t, finished_t, buffers)
            actual = {name: getattr(buffers, name).cpu().tolist() for name in
                      ('accepted', 'commit', 'last', 'emitted', 'accept_mask', 'final_mask', 'status')}
            expected = self.expected(accepted, commits, emitted, finished)
            mismatches = []
            for name in actual:
                for slot, (got, want) in enumerate(zip(actual[name], expected[name])):
                    if got != want:
                        mismatches.append({'field': name, 'slot': slot, 'actual': got, 'expected': want})
            if mismatches:
                self.stats['failures'] += 1
                self.stats['details'].append({'call': call, 'mismatches': mismatches,
                    'draft': proposal_cpu, 'target': target_cpu,
                    'remaining': remaining, 'finished': finished})
            else:
                self.stats['passes'] += 1
        except Exception:
            self.stats['failures'] += 1
            self.stats['details'].append({'call': call, 'error': traceback.format_exc()})

    @staticmethod
    def expected(accepted, commits, emitted, finished):
        """Use actual CPU-loop decisions/output, not a new acceptance oracle."""
        expected = {name: [] for name in
                    ('accepted', 'commit', 'last', 'emitted', 'accept_mask', 'final_mask', 'status')}
        for a, c, tokens, done in zip(accepted, commits, emitted, finished):
            active = not done and c > 0
            expected['accepted'].append(a if active else 0)
            expected['commit'].append(c)
            expected['last'].append(tokens[-1] if tokens else -1)
            expected['emitted'].append(tokens + [-1] * (6 - len(tokens)))
            expected['accept_mask'].append([active and j < a for j in range(6)])
            expected['final_mask'].append([active and j == a for j in range(6)])
            expected['status'].append(0)
        return expected
