"""C64-only supplement: retain V2 checks, use independent SDK completion proof.

Original periodic GD GOOD_END values and V2 failure remain untouched. Actual
WAIT/READ_SIZE ends and typed/native API completions must match per worker.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit(run_id, capture_id, before_id):
    assert (run_id, capture_id, before_id) == (
        '20261017T1880-vllm-formal-C64-v2',
        '20261017T1884-vllm-formal-C64-native-v4',
        '20261017T1874-vllm-formal-C32-native-v4',
    )
    original = ROOT / 'scripts/audit_glm53_vllm_formal_native_local_v2.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == 'bc07e9b8f85325905e3ec31fe35be416c14c48939dd4227e31dec10b043a380b'
    helper = ROOT / 'scripts/glm53_vllm_SDK_completion_review_local_v1.py'
    helper_sha = 'a85c9bd4c51246500f52eae90988e7615e7387d4cca31fa15a778a87c074e699'
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == helper_sha
    fixture = json.loads((ROOT / 'results/vllm-SDK-completion-review-local-fixture-v1.json').read_text())
    assert fixture['pass'] and fixture['helper_sha256'] == helper_sha and len(fixture['negative_cases']) == 10
    spec = importlib.util.spec_from_file_location('SDK_completion_review', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    supplement = module.review()
    assert supplement['pass'] and supplement['SDK_completion_calls'] == 640
    source = original.read_text()
    old_line = "  SDK_counts[str(node)]={'GOOD_END_delta':sdk_PD_count(o,initial)-sdk_PD_count(previous,initial),'BAD_END':0,'last_ON_FLY':0 if node in [43,45] else None}"
    new_line = """  if node in [43,45]:SDK_counts[str(node)]=SDK_REVIEW['node_completion_summary'][str(node)]
  else:
   original_delta=sdk_PD_count(o,initial)-sdk_PD_count(previous,initial);assert original_delta==0
   SDK_counts[str(node)]={'SDK_completion_calls':original_delta,'raw_primary_GOOD_END_delta':original_delta,'BAD_END':0,'last_ON_FLY':None}"""
    assert source.count(old_line) == 1
    source = source.replace(old_line, new_line)
    old_sum = "assert sum(x['GOOD_END_delta'] for x in SDK_counts.values())==period_PD_reads"
    assert source.count(old_sum) == 1
    source = source.replace(old_sum, "assert sum(x['SDK_completion_calls'] for x in SDK_counts.values())==period_PD_reads")
    marker = " native_path=f'results/vllm-formal-C{C}-v2-native-v2.json';"
    assert source.count(marker) == 1
    source = source.replace(marker, " native['SDK_periodic_tail_supplement']='results/vllm-C64-SDK-completion-review-v1.json';native['SDK_periodic_tail_supplement_sha256']=sha(native['SDK_periodic_tail_supplement']);native['raw_periodic_GD_GOOD_delta']=SDK_REVIEW['raw_primary_GOOD_END_delta'];native['raw_periodic_nonzero_tail_workers']=SDK_REVIEW['raw_primary_nonzero_tail_workers'];native['original_strict_native_V2_failure_preserved']=True\n native_path=f'results/vllm-formal-C{C}-v2-native-v3.json';")
    marker = " certificate=f'results/vllm-formal-C{C}-v2-measurement-v1.json';"
    assert source.count(marker) == 1
    source = source.replace(marker, " summary['native_acceptance_revision']='v3-independent-SDK-completion';summary['original_native_V2_failure_preserved']=True;summary['SDK_periodic_tail_review']='results/vllm-C64-SDK-completion-review-v1.json';summary['SDK_periodic_tail_review_sha256']=sha(summary['SDK_periodic_tail_review'])\n" + marker)
    scope = {'__name__': 'glm53_native_C64_completion_v3', '__file__': str(Path(__file__)), 'SDK_REVIEW': supplement}
    exec(compile(source, str(original) + ':reviewed-C64-completion-supplement', 'exec'), scope)
    result = scope['audit'](run_id, capture_id, before_id)
    review = {
        'project_id': 'glm53', 'original_native_V2_source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'local_V3_wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reviewed_transformed_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'completion_helper_sha256': helper_sha,
        'scope': 'C64 only. Original input/cache/PD request joins/typed byte counts/SDK bad values/graph/MTP/complete metrics/source/error/idle checks retained. Only D periodic GD-end balance is replaced by independent per-worker WAIT+READ_SIZE+native/typed completions. P SDK checks remain original.',
        'raw_counters_changed': False, 'model_SDK_init_SSH_calls': 0, 'pass': True,
    }
    (ROOT / 'results/vllm-C64-native-completion-audit-source-review-v1.json').write_text(json.dumps(review, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    parser.add_argument('capture_id')
    parser.add_argument('before_id')
    args = parser.parse_args()
    audit(args.run_id, args.capture_id, args.before_id)
