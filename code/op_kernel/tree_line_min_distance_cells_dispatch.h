#pragma once
namespace TreeDistanceCells {
template<typename T>
__aicore__ inline bool UseSharedCellsForAllBatches(GM_ADDR s,uint32_t batches,uint32_t m)
{
    GlobalTensor<T> segments;
    segments.SetGlobalBuffer((__gm__ T*)s);
    LocalTensor<T> raw(TPosition::VECCALC,0,512/sizeof(T));
    const uint32_t step=(m-1)/15;
    constexpr uint32_t stride=32/sizeof(T);
    for(uint32_t batch=0;batch<batches;++batch) {
        if(batch) Hop<HardEvent::S_MTE2>();
        DataCopyPad(raw,segments[(uint64_t)batch*m*6],
            {16,6*(uint32_t)sizeof(T),(step-1)*6*(uint32_t)sizeof(T),0,0},{false,0,0,0});
        Hop<HardEvent::MTE2_S>();
        float low[3]={kGuardDist,kGuardDist,kGuardDist};
        float high[3]={-kGuardDist,-kGuardDist,-kGuardDist};
        float width[3]={0.0f,0.0f,0.0f};
        for(uint32_t k=0;k<16;++k) {
            #pragma unroll
            for(uint32_t a=0;a<3;++a) {
                const float x=(float)raw.GetValue(k*stride+a);
                const float y=(float)raw.GetValue(k*stride+a+3);
                const float center=x*0.5f+y*0.5f;
                low[a]=center<low[a]?center:low[a];
                high[a]=center>high[a]?center:high[a];
                width[a]+=x<y?y-x:x-y;
            }
        }
        uint32_t axis=0;
        if(high[1]-low[1]>high[axis]-low[axis]) axis=1;
        if(high[2]-low[2]>high[axis]-low[axis]) axis=2;
        const float extent=high[axis]-low[axis];
        if(!(extent>0.0f && extent<3.0e30f)) continue;
        if(!(width[axis]*(float)(int32_t)(m-1)>512.0f*extent)) return false;
    }
    return true;
}
}
