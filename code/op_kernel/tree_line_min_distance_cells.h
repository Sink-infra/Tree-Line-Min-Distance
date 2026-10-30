#pragma once
#include "tree_line_min_distance_cells_dispatch.h"
#include "tree_line_min_distance_cells_prepare.h"
#include "tree_line_min_distance_cells_query.h"
namespace TreeDistanceCells {
template<typename T>
__aicore__ inline bool Run(GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,GM_ADDR workspace,
    const TreeLineMinDistanceTilingData& t)
{
    if(t.b*t.chunks!=GetBlockNum() || t.chunkPoints<2048 || t.chunkPoints>3328 || t.m<128 || workspace==nullptr) return false;
    if(!UseSharedCellsForAllBatches<T>(s,t.b,t.m)) return false;
    const uint32_t task=GetBlockIdx(),batch=task/t.chunks;
    const uint32_t begin=(task%t.chunks)*t.chunkPoints;
    if(begin>=t.n) return false;
    const uint32_t owned=t.n-begin<t.chunkPoints?t.n-begin:t.chunkPoints;

    GlobalTensor<T> points,segments,distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T*)p); segments.SetGlobalBuffer((__gm__ T*)s);
    distances.SetGlobalBuffer((__gm__ T*)d); indices.SetGlobalBuffer((__gm__ int32_t*)i);
    float low[3],width[3],radius;
    const bool ready=Prepare(points,segments,batch,begin,owned,t,low,width,radius);
    GlobalTensor<float> records;
    records.SetGlobalBuffer((__gm__ float*)workspace);
    LocalTensor<uint8_t> candidateMask(TPosition::VECCALC,187776,256);
    const float threshold=ready?radius*radius*1.0001f+0.01f:0.0f;
    if(ready && task%t.chunks==0) {
        Hop<HardEvent::V_MTE3>();
        DataCopy(records[(uint64_t)t.b*512*2304+(uint64_t)batch*9216],F(0,9216),9216);
        Hop<HardEvent::MTE3_V>();
    }
    bool sent=false;
    for(uint32_t cell=task%t.chunks;cell<343;cell+=t.chunks) {
        if(sent) { Hop<HardEvent::MTE3_V>(); Hop<HardEvent::MTE3_S>(); }
        uint64_t candidates=0;
        if(ready) {
            const float cx=low[0]+((float)(int32_t)(cell/49)+0.5f)*width[0];
            const float cy=low[1]+((float)(int32_t)((cell/7)%7)+0.5f)*width[1];
            const float cz=low[2]+((float)(int32_t)(cell%7)+0.5f)*width[2];
            Hop<HardEvent::S_V>();
            auto dx=F(166976),dy=F(171136),dz=F(175296),ratio=F(179456),value=F(183616);
            Adds(dx,F(0),-cx,t.m); Adds(dy,F(4096),-cy,t.m); Adds(dz,F(8192),-cz,t.m);
            Mul(ratio,dx,F(24576),t.m); MulAddDst(ratio,dy,F(28672),t.m); MulAddDst(ratio,dz,F(32768),t.m);
            Maxs(ratio,ratio,0.0f,t.m); Mins(ratio,ratio,1.0f,t.m);
            MulAddDst(dx,ratio,F(12288),t.m); MulAddDst(dy,ratio,F(16384),t.m); MulAddDst(dz,ratio,F(20480),t.m);
            Mul(value,dx,dx,t.m); MulAddDst(value,dy,dy,t.m); MulAddDst(value,dz,dz,t.m);
            CompareScalar(candidateMask,value,threshold,CMPMODE::LE,t.m);
            GatherMask(F(157184),F(130048,3328),candidateMask.ReinterpretCast<uint32_t>(),
                true,t.m,{1,1,8,0},candidates);
            Hop<HardEvent::V_S>();
            Hop<HardEvent::S_V>();
            if(candidates<=128) {
                if(candidates) {
                    Muls(value,F(157184),4.0f,(uint32_t)candidates);
                    Cast(I(187232,128),value,RoundMode::CAST_RINT,(uint32_t)candidates);
                }
                for(uint32_t a=0;a<9;++a) {
                    Duplicate(F(161280+a*512,128),a<3?kGuardCoord:0.0f,128);
                    if(candidates) Gather(F(161280+a*512,128),F(a*4096),I(187232,128).ReinterpretCast<uint32_t>(),0u,(uint32_t)candidates);
                }
            }
        }
        Hop<HardEvent::V_S>();
        I(156672,8).SetValue(0,ready && candidates<=128?(int32_t)candidates:-1);
        Hop<HardEvent::S_MTE3>(); Hop<HardEvent::V_MTE3>();
        DataCopy(records[((uint64_t)batch*512+cell)*2304],F(156672,2304),2304);
        sent=true;
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    if(!ready) return false;
    Duplicate(F(151552,32),0.0f,32); Duplicate(F(151680,32),0.0f,32); Duplicate(F(151808,32),0.0f,32);
    Duplicate(F(155680,64),0.0f,64); Duplicate(F(155392,64),0.0f,64);
    LocalTensor<uint8_t> pointMask(TPosition::VECCALC,156672,512);
    uint32_t failed=0;
    for(uint32_t cell=0;cell<343;++cell) {
        CompareScalar(pointMask,F(76800,3328),(float)(int32_t)cell,CMPMODE::EQ,AlignLanes(owned));
        uint64_t pointCount=0;
        GatherMask(F(143360,3328),F(130048,3328),pointMask.ReinterpretCast<uint32_t>(),
            true,AlignLanes(owned),{1,1,8,0},pointCount);
        Hop<HardEvent::V_S>();
        if(pointCount==0) { Hop<HardEvent::S_V>(); continue; }
        Hop<HardEvent::V_MTE2>();
        DataCopy(F(156672,2304),records[((uint64_t)batch*512+cell)*2304],2304);
        Hop<HardEvent::MTE2_S>();
        const int32_t candidateCount=I(156672,8).GetValue(0);
        if(candidateCount<0 || candidateCount>128) return false;
        const uint32_t candidates=(uint32_t)candidateCount;
        Hop<HardEvent::MTE2_V>(); Hop<HardEvent::S_V>();
        if(!Group((uint32_t)pointCount,(uint32_t)candidates,failed)) return false;
    }
    if(failed!=0) {
        Hop<HardEvent::V_MTE2>();
        DataCopy(F(0,9216),records[(uint64_t)t.b*512*2304+(uint64_t)batch*9216],9216);
        Hop<HardEvent::MTE2_V>();
        Refine(failed,t.m);
    }
    Hop<HardEvent::S_V>();
    Sqrt(F(103424,3328),F(103424,3328),owned);
    const uint64_t at=(uint64_t)batch*t.n+begin;
    if constexpr(sizeof(T)==2) {
        LocalTensor<T> narrow(TPosition::VECCALC,143360,3328);
        Cast(narrow,F(103424,3328),RoundMode::CAST_RINT,owned);
        Hop<HardEvent::V_MTE3>();
        DataCopyPad(distances[at],narrow,{1,owned*(uint32_t)sizeof(T),0,0,0});
    } else {
        Hop<HardEvent::V_MTE3>();
        DataCopyPad(distances[at],F(103424,3328).template ReinterpretCast<T>(),{1,owned*4,0,0,0});
    }
    DataCopyPad(indices[at],I(116736,3328),{1,owned*4,0,0,0});
    return true;
}
}
