"""Allocate the complete native target/draft pool topology on each NPU.

No model weights, attention backend or network PD is exercised here.
"""
from pathlib import Path
import datetime, hashlib, json, os, traceback
import torch, torch_npu

rank = int(os.environ['LOCAL_RANK'])
os.sched_setaffinity(0, {168 + rank})
torch.set_num_threads(1)
torch.set_grad_enabled(False)
torch.npu.set_device(rank)
root = Path(__file__).resolve().parent
out = root / 'evidence'
out.mkdir(exist_ok=True)
record = dict(rank=rank, pid=os.getpid(), phase='initializing', cases=[],
              scope='Native full target/draft pool topology, actual allocations, '
                    'local transfer byte geometry and index graphs; not weights, '
                    'attention, cross-host PD, full-model capacity or performance')

def save():
    p = out / f'rank{rank}.json'
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(record, indent=2))
    q.replace(p)

try:
    save()
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.runtime_context import create_context, use_context
    from lowratio_pool import LowRatioNPUTokenToKVPool
    from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4
    candidate = json.loads((root / 'candidate_d.json').read_text())
    model = Path('/task/models/DeepSeek-V4.1-Flash-1017-w4a8')
    config = json.loads((model / 'config.json').read_text())
    argv = [str(model) if x == '/models/ds41' else x
            for x in candidate['argv'][1:]]
    # This component has no serving role and creates no transport endpoints.
    argv[argv.index('--disaggregation-mode') + 1] = 'null'
    argv.remove('--disaggregation-decode-enable-radix-cache')
    args = prepare_server_args(argv)
    context = create_context(args, role='model_runner')
    ratios = config['compress_ratios']
    sources = config['kv_source_layer_ids']
    target_layers = config['num_hidden_layers']
    record['checkpoint_topology'] = dict(ratios=ratios, sources=sources,
        target_layers=target_layers, config_sha256=hashlib.sha256(
            (model / 'config.json').read_bytes()).hexdigest())
    with use_context(context):
        kwargs = dict(max_num_reqs=16, num_req_slots=17, swa_size=512,
            c4_size=0, c128_size=0, c4_state_pool_size=0,
            c128_state_pool_size=0, page_size=128, swa_page_size=128,
            dtype=torch.bfloat16, c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32, qk_nope_head_dim=448,
            qk_rope_head_dim=64, indexer_head_dim=128,
            device=f'npu:{rank}', enable_memory_saver=False,
            compression_ratios=ratios, sliding_window=128,
            online_mtp_max_draft_tokens=6, kv_source_layers=sources,
            full_size=8192)
        target = LowRatioNPUTokenToKVPool(
            **kwargs, layer_num=target_layers, start_layer=0,
            end_layer=target_layers, is_draft_worker=False)
        draft = LowRatioNPUTokenToKVPool(
            **kwargs, layer_num=len(ratios)-target_layers,
            start_layer=target_layers, end_layer=len(ratios),
            is_draft_worker=True)
        assert target.c4_kv_pool is None and target.c128_kv_pool is None
        assert draft.get_contiguous_buf_infos() == ([], [], [])
        assert target.get_c128_kv_buf_infos() == ([], [], [])
        ptrs, sizes, items = target.get_contiguous_buf_infos()
        expected = []
        for ratio, owners in target.sources_by_ratio.items():
            pool = target.kv_pools[ratio]
            indexer = target.index_pools[ratio]
            assert indexer.get_bytes_per_token() == 68
            assert not hasattr(indexer, 'index_k_buffer')
            assert pool.get_bytes_per_token() == 1024
            rows = 128 // ratio
            expected.extend([rows*1024]*len(owners) + [rows*68]*len(owners))
            view = pool.paged_buffer_info()
            assert view.page_slots == rows and view.page_bytes == rows*1024
            assert all(a.data_ptr() == b.data_ptr()
                       for a,b in zip(view.buffers, pool.kv_buffer))
        assert items == expected and len(items) == 2*len(sources)
        for layer in range(target_layers):
            ratio = ratios[layer]
            if ratio not in (1,2):
                continue
            owner = max(x for x in sources if x <= layer and ratios[x] == ratio)
            assert target.source_layer_of(layer) == owner
            assert target.get_key_buffer(layer).data_ptr() == target.get_key_buffer(owner).data_ptr()
        pairs = [p for p in target.compress_state_pools if p is not None]
        assert len(pairs) == len(target.sources_by_ratio[2])
        assert all(p.ratio == 2 and p.request_scoped and p.ring_size == 8
                   and p.kv_score_buffer.kv_score.dtype == torch.float32 for p in pairs)
        record['topology'] = dict(main_regions=len(ptrs), item_bytes=items,
            region_bytes=sizes, native_source_mapping=True,
            no_INT8_side_indexer_buffers=True, C2_request_ring_size=8,
            pair_state_buffers=len(pairs), draft_layers=len(ratios)-target_layers)
        save()
        # Decode actual raw wire items at nonzero FULL page ids, including the
        # C1 index-page boundary and both halves of a C2 physical page.
        gen = torch.Generator().manual_seed(222)
        for ratio, owners in target.sources_by_ratio.items():
            pool = target.kv_pools[ratio]
            indexer = target.index_pools[ratio]
            slots_per_page = 128 // ratio
            buf = pool.kv_buffer[0]
            flat = buf.view(-1,512)
            for page in [1,2,3]:
                first = page * slots_per_page
                cpu = torch.randn(slots_per_page,512,generator=gen).bfloat16()
                flat[first:first+slots_per_page].copy_(cpu)
                wire = buf.view(torch.uint8).flatten()[
                    page*slots_per_page*1024:(page+1)*slots_per_page*1024]
                assert torch.equal(wire.cpu().view(torch.bfloat16).view(-1,512),cpu)
            slots_cpu = torch.arange(slots_per_page, slots_per_page*4)
            slots = slots_cpu.to(f'npu:{rank}')
            x_cpu = torch.randn(len(slots_cpu),128,generator=gen).bfloat16()
            x = x_cpu.to(f'npu:{rank}')
            def call():
                indexer.set_index_fp4(0,slots,x)
                return target.get_low_ratio_index_k_dequant(owners[0],slots)
            actual = call()
            torch.npu.synchronize()
            assert torch.equal(actual.cpu(),fake_quant_fp4(x_cpu))
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph,pool=torch.npu.graph_pool_handle(),
                                 auto_dispatch_capture=True):
                captured=call()
            for replay in range(3):
                changed=(x_cpu.float()+.125*(replay+1)).bfloat16()
                x.copy_(changed)
                graph.replay()
                torch.npu.synchronize()
                assert torch.equal(captured.cpu(),fake_quant_fp4(changed))
            record['cases'].append(dict(ratio=ratio,owners=owners,phase='passed',
                graph_captures=1,changed_input_replays=3,
                wire_pages_checked=[1,2,3], bitwise_CPU_FP4_readback=True))
            save()
    record.update(phase='passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
except BaseException as exc:
    record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc())
    save()
    raise
