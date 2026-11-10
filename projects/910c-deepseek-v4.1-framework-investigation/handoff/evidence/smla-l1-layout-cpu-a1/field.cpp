#include <cstdint>
#include <iostream>
#define __aicore__
struct Nd2NzParams {
    __aicore__ Nd2NzParams() {}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    __aicore__ Nd2NzParams(
        const uint16_t ndNumIn, const uint16_t nValueIn, const uint32_t dValueIn, const uint64_t srcNdMatrixStrideIn,
        const uint64_t srcDValueIn, const uint16_t dstNzC0StrideIn, const uint16_t dstNzNStrideIn,
        const uint32_t dstNzMatrixStrideIn)
        : ndNum(ndNumIn),
          nValue(nValueIn),
          dValue(dValueIn),
          srcNdMatrixStride(srcNdMatrixStrideIn),
          srcDValue(srcDValueIn),
          dstNzC0Stride(dstNzC0StrideIn),
          dstNzNStride(dstNzNStrideIn),
          dstNzMatrixStride(dstNzMatrixStrideIn)
    {}

    uint16_t ndNum = 0;
    uint16_t nValue = 0;
    uint32_t dValue = 0;
    uint64_t srcNdMatrixStride = 0;
    uint64_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint32_t dstNzMatrixStride = 0;
#else
    __aicore__ Nd2NzParams(
        const uint16_t ndNumIn, const uint16_t nValueIn, const uint16_t dValueIn, const uint16_t srcNdMatrixStrideIn,
        const uint16_t srcDValueIn, const uint16_t dstNzC0StrideIn, const uint16_t dstNzNStrideIn,
        const uint16_t dstNzMatrixStrideIn)
        : ndNum(ndNumIn),
          nValue(nValueIn),
          dValue(dValueIn),
          srcNdMatrixStride(srcNdMatrixStrideIn),
          srcDValue(srcDValueIn),
          dstNzC0Stride(dstNzC0StrideIn),
          dstNzNStride(dstNzNStrideIn),
          dstNzMatrixStride(dstNzMatrixStrideIn)
    {}

    uint16_t ndNum = 0;
    uint16_t nValue = 0;
    uint16_t dValue = 0;
    uint16_t srcNdMatrixStride = 0;
    uint16_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint16_t dstNzMatrixStride = 0;
#endif
};
int main(){Nd2NzParams p;uint64_t x=127*512,y=128*512;p.srcDValue=x;std::cout<<sizeof(p.srcDValue)<<" "<<p.srcDValue<<" ";p.srcDValue=y;std::cout<<p.srcDValue<<"\n";}
