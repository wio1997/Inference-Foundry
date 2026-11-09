// Execute only unchanged frozen host shape arithmetic, without CANN or NPU.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include "moe_init_routing_quant_v2_tiling.h"

int main() {
    std::cout << "{\"scope\":\"offline draft E128 top3 K5120 AIV40 UB196352 host selector, not loaded native tiling\",\"rows\":[";
    for (int m = 1; m <= 3; ++m) {
        optiling::MoeInitRoutingQuantV2TilingBase t;
        if (!t.DoTiling(m,5120,3,0,129,0,0,2,false,2,1,0,40,196352)) return 2;
        auto g = t.quantTilingData.gatherOutComputeParamsOp;
        if (m > 1) std::cout << ',';
        std::cout << '[' << m << ',' << t.tilingKey_ << ',' << g.needCoreNum
                  << ',' << g.perCoreRows << ',' << g.lastCoreRows
                  << ',' << g.perCorePerLoopRows << ',' << g.perLoopCols
                  << ',' << g.colLoops << ']';
    }
    std::cout << "]}" << std::endl;
}
