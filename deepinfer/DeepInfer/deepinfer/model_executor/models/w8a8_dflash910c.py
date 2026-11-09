"""Original DFlash sampling over already resident W8A8 draft computation.

Weights come only from the existing DSpark object. Sharded output weights
are assembled once for original driver-only sampling, never loaded again.
The original DFlashSampling implementation owns proposal distributions,
random parameters, Markov recurrence, confidence and MTP pool writes.
"""
import math
import weakref

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from deepinfer import envs
from deepinfer.model_executor.models.mtp.sampling import DFlashSampling
from deepinfer.model_executor.parallel_utils.communication_op import cp_broadcast_multi_tensor
from deepinfer.model_executor.parallel_utils.parallel_state import PG
from .w8a8_math910c import project, rms_norm_mixed


class _DraftNorm(nn.Module):
    def __init__(self, draft):
        super().__init__()
        self.register_buffer('weight', draft.norm, persistent=False)
        self.eps = draft.cfg['rms_norm_eps']

    def forward(self, hidden):
        return rms_norm_mixed(hidden, self.weight, self.eps)[0]


class _MarkovOutput(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer('weight', weight, persistent=False)

    def forward(self, hidden, *, out_acc):
        out_acc.add_(project(hidden.float(), self.weight))
        return out_acc


class _Confidence(nn.Module):
    def __init__(self, draft):
        super().__init__()
        self.register_buffer('weight', draft.confidence_weight, persistent=False)
        self.temperatures = draft.temperatures

    def forward(self, hidden, markov, *, out, step_i):
        # This is the resident backend's existing confidence computation.
        features = torch.cat((hidden, markov), -1).float()
        raw = (features * self.weight.float().flatten()).sum(-1) / self.temperatures[step_i]
        out.copy_(torch.sigmoid(raw.clamp(-60, 60)))
        return out


class W8A8DFlash(DFlashSampling, nn.Module):
    def __init__(self, model):
        super().__init__()
        draft = model.dspark
        required = ('forward_hidden', 'head', 'markov_head', 'markov_embed',
                    'confidence_weight', 'norm', 'temperatures', 'nextn', 'cfg')
        if draft is None or any(not hasattr(draft, name) for name in required):
            raise NotImplementedError('resident W8A8 DFlash compute/weight interfaces are incomplete')
        if draft.nextn != envs.DEEPINFER_USE_NEXTN_DECODE:
            raise ValueError('original nextn differs from resident W8A8 DFlash')
        self._model = weakref.ref(model)
        self.draft = draft
        self.worker = None
        self.vocab_size = model.cfg['vocab_size']
        self.max_MTP_value = draft.nextn
        self.noise_token_id = draft.cfg['dspark_noise_token_id']
        self.force_topk = envs.DEEPINFER_SAMPLER_FORCE_TOPK
        self.logprobs_dim = self.force_topk or self.vocab_size
        self.enable_markov = True
        self.enable_confidence = self.has_confidence = True
        self.mtp_info_pool = None
        self.lm_head = None
        self.markov_w1 = self.markov_w2 = None
        self.post_norm = _DraftNorm(draft)
        self.confidence_head = _Confidence(draft)
        self.decode_graph_runner = None
        self.sampling_validation = None

    def capture_graphs(self):
        if self.decode_graph_runner is not None:
            return
        if self.worker is None:
            raise RuntimeError('initialize original DFlash resources before graph capture')
        from .w8a8_dflash_graph910c import OriginalDraftGraphs
        self.decode_graph_runner = OriginalDraftGraphs(self, self._model(), self.worker)

    def destroy_graphs(self):
        if self.decode_graph_runner is not None:
            self.decode_graph_runner.destroy()
            self.decode_graph_runner = None

    def run_drafts(self, context, decode_input, nextn_kv_caches):
        """Pure device stage captured by the original DFlash graph runner."""
        driver = decode_input.driver
        if driver is None:
            offsets = torch.empty_like(decode_input.first_token_positions)
            golden = torch.empty_like(decode_input.first_token_positions)
        else:
            offsets, golden = driver.last_valid_offset, driver.golden_tokens
        cp_broadcast_multi_tensor([offsets, golden])
        count = decode_input.req_cnt
        first = decode_input.first_token_positions
        valid = context.valid[:count] & (first >= 0)
        starts = context.positions.index_select(0, first.clamp_min(0).long()) + offsets + 1
        padding_starts = ((~valid).to(torch.int32).cumsum(0).to(torch.int32) - 1).clamp_min(0) * self.max_MTP_value
        starts = torch.where(valid, starts, torch.where(context.slots[:count] == context.scratch, padding_starts, 0))
        from .decode_trace910c import capture
        with capture(self,count,decode_input) as observed:
            hidden = self.draft.forward_hidden(golden, starts, context.slots[:count],
                mc2_global_bs=context.global_bs, active_requests=valid)
            if observed is not None:
                observed['values'].update(hidden=hidden,starts=starts,slots=context.slots[:count],valid=valid)
            if driver is None:
                return None
            with self.sampling_validation.record():
                return self.sample_drafts(hidden.flatten(0, 1), driver)

    def take_graph_checks(self):
        return [] if self.sampling_validation is None else self.sampling_validation.take()

    def initialize_worker(self, worker):
        if self.worker is worker:
            return
        if self.worker is not None:
            raise RuntimeError('resident DFlash already belongs to another Worker')
        group = PG.context_parallel_group()
        error = None
        if group.size() != self.draft.tp_size or group.rank() != self.draft.tp_rank:
            error = 'original CP topology differs from resident draft shards'
        if bool(worker.is_driver_worker) != (group.rank() == 0):
            error = 'original DFlash sampling must belong to CP rank zero'
        self._vote(error)
        # CP followers only compute hidden rows. Original DFlash samples and
        # writes the proposal pool on the driver; it needs the complete head.
        for name, local in [('head', self.draft.head.bfloat16()),
                            ('markov', self.draft.markov_head)]:
            full, error = None, None
            try:
                full = local.new_empty((local.shape[0] * group.size(), local.shape[1]))
            except Exception as exc:
                error = str(exc)
            self._vote(error)
            if group.size() == 1:
                full.copy_(local)
            else:
                dist.all_gather_into_tensor(full, local.contiguous(), group=group)
            if worker.is_driver_worker:
                if name == 'head':
                    self.lm_head = nn.Module()
                    self.lm_head.register_buffer('weight', full, persistent=False)
                else:
                    self.markov_w2 = _MarkovOutput(full)
            del full
        error = None
        try:
            from deepinfer.npu_ops.sampling_validation910c import SamplingValidation
            self.sampling_validation = SamplingValidation(self.draft.head.device)
            if worker.is_driver_worker:
                self.markov_w1 = nn.Embedding.from_pretrained(self.draft.markov_embed, freeze=True)
                self.init_mtp_info_pool(worker.scheduler_config.max_num_seqs)
        except Exception as exc:
            error = str(exc)
        self._vote(error)
        self.worker = worker

    @staticmethod
    def _vote(error):
        status = torch.tensor([int(error is not None)], dtype=torch.int32, device='cpu')
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=PG.cpu_ep_in_dp)
        if status.item():
            raise RuntimeError(f'original DFlash resource validation failed before collective: {error}')

    def decode_forward(self, main_input_metadata, nextn_kv_caches, decode_input):
        if self.worker is None:
            raise RuntimeError('initialize original DFlash resources before decoding')
        packet = main_input_metadata.original_packet
        error = None
        try:
            if main_input_metadata.is_prompt or packet is None or packet.is_prompt:
                raise ValueError('DFlash requires original decode metadata')
            model = self._model()
            if model is None or nextn_kv_caches is not model.cache:
                raise ValueError('DFlash must use its resident original cache owner')
            slots = [row.state_id - 1 for row in packet.rows] or [model.requests - 1]
            count = len(slots)
            if tuple(decode_input.first_token_positions.shape) != (count,):
                raise ValueError('DFlash request rows differ from original metadata')
            first = decode_input.first_token_positions
            if first.dtype != torch.int32 or first.device != model.device:
                raise ValueError('invalid original DFlash first-token tensor')
            if self.worker.is_driver_worker != (decode_input.driver is not None):
                raise ValueError('DFlash driver sampling inputs differ from original rank ownership')
            if decode_input.driver is not None:
                for tensor in (decode_input.driver.last_valid_offset, decode_input.driver.golden_tokens):
                    if tensor.shape != first.shape or tensor.dtype != torch.int32 or tensor.device != model.device:
                        raise ValueError('invalid original DFlash accepted offset/golden tensor')
        except Exception as exc:
            error = str(exc)
            count = 0
        status = torch.tensor([int(error is not None), count * self.max_MTP_value],
                              dtype=torch.int32, device='cpu')
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=PG.cpu_ep_in_dp)
        if status[0].item():
            raise RuntimeError(f'original DFlash input validation failed before EP: {error}')
        if self.decode_graph_runner is not None:
            self.decode_graph_runner.stage(main_input_metadata.positions, slots, real_requests=bool(packet.rows))
            result = self.decode_graph_runner.replay(decode_input)
            self.draft.proposals += 1
            return result
        driver = decode_input.driver
        if driver is None:
            offsets = torch.empty_like(decode_input.first_token_positions)
            golden = torch.empty_like(decode_input.first_token_positions)
        else:
            offsets, golden = driver.last_valid_offset, driver.golden_tokens
        cp_broadcast_multi_tensor([offsets, golden])
        # Exactly the original build_dflash_input coordinate: the golden token
        # follows the last accepted target row. No draft token is committed here.
        starts = main_input_metadata.positions.index_select(
            0, decode_input.first_token_positions.clamp_min(0).long()) + offsets + 1
        hidden = self.draft.forward_hidden(golden, starts, slots,
            mc2_global_bs=dist.get_world_size() * math.ceil(int(status[1].item()) / self.draft.tp_size))
        self.draft.proposals += 1
        if driver is None:
            return None
        self.sampling_validation.reset()
        with self.sampling_validation.record():
            result = self.sample_drafts(hidden.flatten(0, 1), driver)
        self.sampling_validation.enqueue()
        return result

    def destroy(self):
        self.destroy_graphs()
        self.worker = None
        self.mtp_info_pool = self.lm_head = self.markov_w1 = self.markov_w2 = None
        self.sampling_validation = None
