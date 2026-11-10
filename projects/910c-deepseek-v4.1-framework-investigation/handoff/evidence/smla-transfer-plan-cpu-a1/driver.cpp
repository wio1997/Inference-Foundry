
#include <algorithm>
#include <cstdint>
#include <climits>
#include <iostream>
#include <map>
#include <vector>
#define __aicore__
#define unlikely(x) (x)
namespace AscendC{enum class HardEvent{MTE3_MTE2,MTE2_MTE3};}
template<AscendC::HardEvent E>void SetFlag(int64_t){}
template<AscendC::HardEvent E>void WaitFlag(int64_t){}
int sub=0;int GetSubBlockIdx(){return sub;}
template<class T> T CeilDiv(T a,T b){return (a+b-1)/b;}
template<class T> T SMLAAlign(T a,T b){return CeilDiv(a,b)*b;}
enum class SMLA_LAYOUT{PA_BBND,BSND,TND};
enum Tag{KV,UB,PANEL};
struct Ptr{Tag tag;int64_t off=0;Ptr operator[](int64_t x)const{return {tag,off+x};}};
struct Table{std::vector<int32_t> values;int32_t GetValue(int64_t i)const{return values.at(i);}};
struct DataCopyExtParams{uint32_t blockLen=0,blockCount=0,dstStride=0,srcStride=0;};
template<class T>struct DataCopyPadExtParams{bool pad;int a,b,c;};
std::map<int64_t,int64_t> ub[2],panel;
template<class T>void DataCopyPad(Ptr dst,Ptr src,DataCopyExtParams p,DataCopyPadExtParams<T>){
 for(uint32_t b=0;b<p.blockCount;b++)for(uint32_t j=0;j<p.blockLen/sizeof(T);j+=512){
  auto d=dst.off+b*(p.blockLen+p.dstStride)/sizeof(T)+j;
  auto s=src.off+b*(p.blockLen+p.srcStride)/sizeof(T)+j;
  ub[sub][d/512]=s;
 }
}
void DataCopyPad(Ptr dst,Ptr src,DataCopyExtParams p){
 for(uint32_t b=0;b<p.blockCount;b++)for(uint32_t j=0;j<p.blockLen/2;j+=512){
  auto d=dst.off+b*(p.blockLen+p.dstStride)/2+j;
  auto s=src.off+b*(p.blockLen+p.srcStride)/2+j;
  panel[d/512]=ub[sub].at(s/512);
 }
}
struct RunInfo{int64_t cmpS2IdLimit=0,bIdx=0,n2Idx=0,tensorCmpBOffset=0,v0S2DealSize=512,v0S2Start=0,topKBaseOffset=0,cmpLoop=0;};
struct Info{int64_t paCmpBlockSize=64,cmpMaxBlockNumPerBatch=0,cmpKvStride0=64*512,kvHeadNum=1,headDim=512,cmpSeqSize=0,sparseBlockSize=1,sparseBlockCount=512;};
template<class SMLAT>class SMLAVectorBlock{public:
 using KV_T=uint16_t;static constexpr SMLA_LAYOUT KV_LAYOUT_T=SMLA_LAYOUT::PA_BBND;
 static constexpr int INPUT2_BUFFER_OFFSET=16384,SYNC_INPUT_BUF2_FLAG=4,MERGE_CACHE_GM_BUF_NUM=3;
 Info constInfo;Table cmpBlockTableGm_,topkGm_;Ptr kvMergUb_{UB},cmpKvGm_{KV},kvMergeGm_{PANEL};uint32_t mergeMte3Idx=0;
 void GetRealS2Idx(int64_t,int64_t&,int64_t,const RunInfo&);
 int64_t GetKeyGmOffset(int64_t,const RunInfo&,int64_t);
 void CopyInSingleKv(int64_t&,int64_t,int64_t,int64_t,int64_t,int64_t,const RunInfo&);
 void CopyInKv(int64_t&,int64_t,int64_t,int64_t,int64_t,const RunInfo&);
 void CopyOutMrgeResult(int64_t,int64_t,int64_t,int64_t,const RunInfo&);
 void ProcessVec0L(const RunInfo&);
};
template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                            int64_t topkGmBaseOffset, const RunInfo &runInfo)
{
    int64_t cmpS2Offset = s2GmOffset;
    int64_t topkGmIdx = cmpS2Offset / constInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= constInfo.sparseBlockCount || s2GmOffset >= runInfo.v0S2DealSize)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>(cmpS2Offset % constInfo.sparseBlockSize);
}

template <typename SMLAT>
__aicore__ inline int64_t SMLAVectorBlock<SMLAT>::GetKeyGmOffset(int64_t realS2Idx, const RunInfo &runInfo,
                                                                 int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyGmOffset = 0;
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        int64_t blkTableIdx = realS2Idx / constInfo.paCmpBlockSize;
        int64_t blkTableOffset = realS2Idx % constInfo.paCmpBlockSize;
        realKeyGmOffset =
            cmpBlockTableGm_.GetValue(runInfo.bIdx * constInfo.cmpMaxBlockNumPerBatch + blkTableIdx) *
                static_cast<int64_t>(constInfo.cmpKvStride0) +
            blkTableOffset * static_cast<int64_t>(constInfo.kvHeadNum) * static_cast<int64_t>(constInfo.headDim);

    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::BSND) {
        int64_t batchStride = (constInfo.cmpKvStride0 == 0) ? static_cast<int64_t>(constInfo.cmpSeqSize) *
                                                                  static_cast<int64_t>(constInfo.kvHeadNum) *
                                                                  static_cast<int64_t>(constInfo.headDim) :
                                                              static_cast<int64_t>(constInfo.cmpKvStride0);
        realKeyGmOffset =
            static_cast<int64_t>(runInfo.bIdx) * batchStride +
            realS2Idx * static_cast<int64_t>(constInfo.kvHeadNum) * static_cast<int64_t>(constInfo.headDim) +
            static_cast<int64_t>(runInfo.n2Idx) * static_cast<int64_t>(constInfo.headDim);
    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        realKeyGmOffset =
            runInfo.tensorCmpBOffset +
            realS2Idx * static_cast<int64_t>(constInfo.kvHeadNum) * static_cast<int64_t>(constInfo.headDim) +
            static_cast<int64_t>(runInfo.n2Idx) * static_cast<int64_t>(constInfo.headDim);
    }
    return realKeyGmOffset;
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                              int64_t realS2Idx, int64_t keyBNBOffset,
                                                              int64_t s2IdLimit, const RunInfo &runInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams;
    intriParams.blockLen = validS2Count * constInfo.headDim * sizeof(KV_T);
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams{false, 0, 0, 0};
    DataCopyPad(
        kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) * constInfo.headDim],
        cmpKvGm_[keyBNBOffset], intriParams, padParams);
    mte2Size += validS2Count;
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                        int64_t realS2Idx1, int64_t realS2Idx2, const RunInfo &runInfo)
{
    int64_t s2IdLimit = runInfo.cmpS2IdLimit;

    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }

    int64_t keySrcStride = 0;
    keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                    constInfo.sparseBlockSize * constInfo.headDim) *
                   sizeof(KV_T);
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 || realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                 realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit)) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        // 因为需要拷贝两块
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, runInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, runInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.headDim * sizeof(KV_T);
        intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams{false, 0, 0, 0};

        int64_t startGmOffset = keyOffset1 > -1 ? keyOffset1 : keyOffset2;
        if (keyOffset2 > -1 && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T) +
                              (mte2Size - mte3Size) * constInfo.headDim],
                    cmpKvGm_[startGmOffset], intriParams, padParams);
        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                 int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                 const RunInfo &runInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = mte2Size - mte3Size;
    dataCopyParams.blockLen = constInfo.headDim * sizeof(KV_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(kvMergeGm_[runInfo.cmpLoop % MERGE_CACHE_GM_BUF_NUM * 512 * 512 +
                           (s2GmStartOffset + mte3Size) * constInfo.headDim],
                kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T)], dataCopyParams);
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::ProcessVec0L(const RunInfo &runInfo)
{
    int64_t s2ProcessSize = runInfo.v0S2DealSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2 * constInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = runInfo.topKBaseOffset + runInfo.v0S2Start;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    int64_t s2SplitPoint = SMLAAlign(s2Pair, 2) * constInfo.sparseBlockSize;
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : s2SplitPoint;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ? s2SplitPoint : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    // 处理两个基本块
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
         s2GmOffsetArray += 2 * constInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, runInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
            mergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset, runInfo);
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo);
        if ((mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > 16) ||
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }
    return;
}
int main(){int count;std::cin>>count;std::cout<<"[";
 for(int t=0;t<count;t++){
  int block,limit,seed,ring,np;std::cin>>block>>limit>>seed>>ring>>np;
  std::vector<int32_t> pages(np),ids(512);for(auto &v:pages)std::cin>>v;for(auto &v:ids)std::cin>>v;
  panel.clear();ub[0].clear();ub[1].clear();
  for(sub=0;sub<2;sub++){
   SMLAVectorBlock<int> op;op.constInfo.paCmpBlockSize=block;op.constInfo.cmpKvStride0=block*512;
   op.constInfo.cmpMaxBlockNumPerBatch=np;op.cmpBlockTableGm_.values=pages;op.topkGm_.values=ids;op.mergeMte3Idx=seed;
   RunInfo r;r.cmpS2IdLimit=limit;r.cmpLoop=ring;op.ProcessVec0L(r);
  }
  std::cout<<(t?",":"")<<"[";bool first=true;for(auto [d,s]:panel){if(!first)std::cout<<",";first=false;std::cout<<"["<<(d-(ring%3)*512)<<","<<s<<"]";}std::cout<<"]";
 }
 std::cout<<"]\n";
}
