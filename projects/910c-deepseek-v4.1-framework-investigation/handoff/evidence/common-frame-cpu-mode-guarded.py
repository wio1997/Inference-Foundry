class GuardedCPUFrame:
    """One caller-supplied immutable input lease, not a global memo table.

        Tensor versions do not detect NumPy writes. Eligibility requires the
        runner's closed invocation: no writer until the frame is retired.
        """

    def __init__(self, epoch):
        self.epoch = epoch
        self.key = None
        self.value = None
        self.inputs = None

    def call(self, nreq, ninput, q, l, local_query_start_loc=None, local_seq_lens=None, start_pos_out=None, is_noncausal=False):
        if q.device.type != 'cpu' or l.device.type != 'cpu' or any((v is not None for v in [local_query_start_loc, local_seq_lens, start_pos_out])):
            return original._build_local_token_metadata(nreq, ninput, q, l, local_query_start_loc, local_seq_lens, start_pos_out, is_noncausal)
        key = (self.epoch, id(q), id(l), version_token(q), version_token(l), q.dtype, l.dtype, tuple(q.shape), tuple(l.shape), group.world_size, group.rank_in_group, nreq, ninput, bool(is_noncausal))
        if self.key is None:
            self.inputs = (q, l, q.clone(), l.clone())
            self.key = key
            self.value = original._build_local_token_metadata(nreq, ninput, self.inputs[2], self.inputs[3], is_noncausal=is_noncausal)
            return self.value
        if key != self.key:
            return original._build_local_token_metadata(nreq, ninput, q, l, is_noncausal=is_noncausal)
        return self.value
