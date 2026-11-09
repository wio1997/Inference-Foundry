// Offline execution of the unchanged frozen host tiler. No CANN/NPU runtime.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include "moe_init_routing_quant_v2_tiling.h"

int main() {
    std::cout << "{\"scope\":\"frozen host selector, target E384 K5120 topK6 AIV40 UB196352 (BLOCK_NUM20)\",\"rows\":[";
    for (int m = 1; m <= 8192; ++m) {
        optiling::MoeInitRoutingQuantV2TilingBase t;
        if (!t.DoTiling(m,5120,6,0,385,0,0,2,false,2,1,0,40,196352)) return 2;
        auto g = t.quantTilingData.gatherOutComputeParamsOp;
        if(m>1)std::cout << ',';
        std::cout << '[' << m << ',' << t.tilingKey_ << ',' << g.needCoreNum
                  << ',' << g.perCoreRows << ',' << g.lastCoreRows << ','
                  << g.perCorePerLoopRows << ',' << g.perCoreLastLoopRows << ','
                  << g.perCoreLoops << ',' << g.lastCoreLoops << ','
                  << g.perLoopCols << ',' << g.colLoops << ','
                  << g.lastCorePerLoopRows << ']';
    }
    std::cout << "]}" << std::endl;
}
