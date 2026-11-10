#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"
int main(){
 for(int64_t m: {0,3,6,24,416,540,544,816,817,818,3328,6540}){
  optiling::MoeInitRoutingQuantV2TilingBase t;
  bool ok=t.DoTiling(m,5120,6,0,385,0,0,2,false,2,1,0,40,196352);
  auto &g=t.quantTilingData.gatherOutComputeParamsOp;
  std::printf("%lld %d %llu %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",(long long)m,ok,(unsigned long long)t.tilingKey_,(long long)t.quantTilingData.smoothType,(long long)g.needCoreNum,(long long)g.perCoreRows,(long long)g.lastCoreRows,(long long)g.perCorePerLoopRows,(long long)g.lastCorePerLoopRows,(long long)g.perCoreLoops,(long long)g.lastCoreLoops,(long long)g.perCoreLastLoopRows,(long long)g.lastCoreLastLoopRows,(long long)g.colLoops);
 }
}
