#pragma once
namespace TreeDistanceCells {
__aicore__ inline bool Group(uint32_t pointCount,uint32_t candidateCount,uint32_t& failed)
{
    if(pointCount>2048) return false;
    auto qx=F(151552,32),qy=F(151680,32),qz=F(151808,32);
    auto xv=F(151936,256),yv=F(152992,256),zv=F(154048,256);
    auto dx=F(0,2048),dy=F(8224,2048),dz=F(16448,2048);
    auto ratio=F(166016,2048),dsq=F(174240,2048);
    auto pair=F(155104,64),cmin=F(155392,64),cidx=F(155680,64);
    auto best=F(155968,64),winner=F(156256,64);
    LocalTensor<uint8_t> keep(TPosition::VECCALC,156544,128);
    const BinaryRepeatParams spread(1,1,0,8,0,1),fold(1,1,1,8,8,0);
    const BinaryRepeatParams plain(1,1,1,8,8,8);
    const UnaryRepeatParams clamp(1,1,8,8);
    for(uint32_t base=0;base<pointCount;base+=32) {
        const uint8_t span=(uint8_t)(pointCount-base<32?pointCount-base:32);
        auto pointBytes=F(182464,64);
        auto pointOffsets=I(182752,64);
        Muls(pointBytes,F(143360,3328)[base],4.0f,(uint32_t)span);
        Cast(pointOffsets,pointBytes,RoundMode::CAST_RINT,(uint32_t)span);
        const auto offsets=pointOffsets.ReinterpretCast<uint32_t>();
        Gather(qx,F(36864,3328),offsets,0u,(uint32_t)span);
        Gather(qy,F(50176,3328),offsets,0u,(uint32_t)span);
        Gather(qz,F(63488,3328),offsets,0u,(uint32_t)span);
        Brcb(xv,qx,(uint8_t)((span+7)/8),{1,8}); Brcb(yv,qy,(uint8_t)((span+7)/8),{1,8}); Brcb(zv,qz,(uint8_t)((span+7)/8),{1,8});
        Duplicate(best,kGuardDist,64); Duplicate(winner,0.0f,64);
        for(uint32_t window=0;window<(candidateCount+63)/64;++window) {
            const uint32_t seg=window*64;
            SetMaskNorm(); SetVectorMask<float,MaskMode::NORMAL>(64);
            Sub<float,false>(dx,F(161280,128)[seg],xv,MASK_PLACEHOLDER,span,spread);
            Sub<float,false>(dy,F(161792,128)[seg],yv,MASK_PLACEHOLDER,span,spread);
            Sub<float,false>(dz,F(162304,128)[seg],zv,MASK_PLACEHOLDER,span,spread);
            Mul<float,false>(ratio,dx,F(164352,128)[seg],MASK_PLACEHOLDER,span,fold);
            MulAddDst<float,float,false>(ratio,dy,F(164864,128)[seg],MASK_PLACEHOLDER,span,fold);
            MulAddDst<float,float,false>(ratio,dz,F(165376,128)[seg],MASK_PLACEHOLDER,span,fold);
            Maxs<float,false>(ratio,ratio,0.0f,MASK_PLACEHOLDER,span,clamp);
            Mins<float,false>(ratio,ratio,1.0f,MASK_PLACEHOLDER,span,clamp);
            MulAddDst<float,float,false>(dx,ratio,F(162816,128)[seg],MASK_PLACEHOLDER,span,fold);
            MulAddDst<float,float,false>(dy,ratio,F(163328,128)[seg],MASK_PLACEHOLDER,span,fold);
            MulAddDst<float,float,false>(dz,ratio,F(163840,128)[seg],MASK_PLACEHOLDER,span,fold);
            Mul<float,false>(dsq,dx,dx,MASK_PLACEHOLDER,span,plain);
            MulAddDst<float,float,false>(dsq,dy,dy,MASK_PLACEHOLDER,span,plain);
            MulAddDst<float,float,false>(dsq,dz,dz,MASK_PLACEHOLDER,span,plain);
            ResetMask();
            WholeReduceMin(cmin,dsq,(int32_t)64,(int32_t)span,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            WholeReduceMin(cidx,dsq,(int32_t)64,(int32_t)span,1,1,8,ReduceOrder::ORDER_ONLY_INDEX);
            PipeBarrier<PIPE_V>();
            Cast(pair,cidx.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);
            Adds(pair,pair,(float)(int32_t)seg,64);
            Compare(keep,cmin,best,CMPMODE::GE,64);
            Min(best,best,cmin,64);
            Select(winner,keep,winner,pair,SELMODE::VSEL_CMPMASK_SPR,64);
        }
        Hop<HardEvent::V_S>();
        for(uint32_t q=0;q<span;++q) {
            const uint32_t at=(uint32_t)(int32_t)F(143360,3328).GetValue(base+q);
            const float value=best.GetValue(q);
            if(candidateCount==0 || !(F(90112,3328).GetValue(at)>value*1.0001f+0.0001f)) {
                if(failed>=32) return false;
                I(187936,32).SetValue(failed++,(int32_t)at);
                F(103424,3328).SetValue(at,0.0f); I(116736,3328).SetValue(at,0);
            } else {
                const uint32_t position=(uint32_t)(int32_t)winner.GetValue(q);
                if(position>=candidateCount) return false;
                F(103424,3328).SetValue(at,value);
                I(116736,3328).SetValue(at,(int32_t)F(157184).GetValue(position));
            }
        }
        Hop<HardEvent::S_V>();
    }
    return true;
}
__aicore__ inline void Refine(uint32_t failed,uint32_t m)
{
    auto dx=F(166976),dy=F(171136),dz=F(175296),ratio=F(179456),value=F(183616);
    for(uint32_t q=0;q<failed;++q) {
        Hop<HardEvent::V_S>();
        const uint32_t at=(uint32_t)I(187936,32).GetValue(q);
        const float px=F(36864,3328).GetValue(at),py=F(50176,3328).GetValue(at),pz=F(63488,3328).GetValue(at);
        Hop<HardEvent::S_V>();
        Adds(dx,F(0),-px,m); Adds(dy,F(4096),-py,m); Adds(dz,F(8192),-pz,m);
        Mul(ratio,dx,F(24576),m); MulAddDst(ratio,dy,F(28672),m); MulAddDst(ratio,dz,F(32768),m);
        Maxs(ratio,ratio,0.0f,m); Mins(ratio,ratio,1.0f,m);
        MulAddDst(dx,ratio,F(12288),m); MulAddDst(dy,ratio,F(16384),m); MulAddDst(dz,ratio,F(20480),m);
        Mul(value,dx,dx,m); MulAddDst(value,dy,dy,m); MulAddDst(value,dz,dz,m);
        ReduceMin(F(179360,64),value,F(143360,3328),m,true);
        Hop<HardEvent::V_S>();
        F(103424,3328).SetValue(at,F(179360,64).GetValue(0));
        I(116736,3328).SetValue(at,I(179360,64).GetValue(1));
        Hop<HardEvent::S_V>();
    }
}
}
