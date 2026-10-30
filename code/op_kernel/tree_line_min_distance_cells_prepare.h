#pragma once
namespace TreeDistanceCells {
__aicore__ inline LocalTensor<float> F(uint32_t offset,uint32_t count=1024) {
    return LocalTensor<float>(TPosition::VECCALC,offset,count);
}
__aicore__ inline LocalTensor<int32_t> I(uint32_t offset,uint32_t count=1024) {
    return LocalTensor<int32_t>(TPosition::VECCALC,offset,count);
}
template<typename T>
__aicore__ inline bool Prepare(const GlobalTensor<T>& points,const GlobalTensor<T>& segments,
    uint32_t batch,uint32_t begin,uint32_t owned,const TreeLineMinDistanceTilingData& t,
    float* low,float* width,float& radius)
{
    LocalTensor<T> pointRaw(TPosition::VECCALC,143360,9984);
    auto pointFloat=F(103424,9984);
    DataCopyPad(pointRaw,points[((uint64_t)batch*t.n+begin)*3],
        {1,owned*3*(uint32_t)sizeof(T),0,0,0},{false,0,0,0});
    CreateVecIndex(F(13312,3328),0.0f,owned);
    Muls(F(13312,3328),F(13312,3328),12.0f,owned);
    Cast(I(0,3328),F(13312,3328),RoundMode::CAST_RINT,owned);
    Hop<HardEvent::MTE2_V>();
    if constexpr(sizeof(T)==2) Cast(pointFloat,pointRaw,RoundMode::CAST_NONE,owned*3);
    else pointFloat=pointRaw.template ReinterpretCast<float>();
    Gather(F(36864,3328),pointFloat,I(0,3328).ReinterpretCast<uint32_t>(),0u,owned);
    Gather(F(50176,3328),pointFloat,I(0,3328).ReinterpretCast<uint32_t>(),4u,owned);
    Gather(F(63488,3328),pointFloat,I(0,3328).ReinterpretCast<uint32_t>(),8u,owned);
    Hop<HardEvent::V_MTE2>();
    LocalTensor<T> segmentRaw(TPosition::VECCALC,103424,6144);
    DataCopyPad(segmentRaw,segments[(uint64_t)batch*t.m*6],
        {1,t.m*6*(uint32_t)sizeof(T),0,0,0},{false,0,0,0});
    CreateVecIndex(F(76800),0.0f,t.m); Muls(F(76800),F(76800),24.0f,t.m);
    Cast(I(90112),F(76800),RoundMode::CAST_RINT,t.m);
    Hop<HardEvent::MTE2_V>();
    auto source=segmentRaw.template ReinterpretCast<float>();
    if constexpr(sizeof(T)==2) {
        source=F(143360,6144); Cast(source,segmentRaw,RoundMode::CAST_NONE,t.m*6);
    }
    Hop<HardEvent::V_S>();

    Hop<HardEvent::S_V>();
    for(uint32_t a=0;a<3;++a) {
        Gather(F(a*4096),source,I(90112).ReinterpretCast<uint32_t>(),a*4,t.m);
        Gather(F((a+3)*4096),source,I(90112).ReinterpretCast<uint32_t>(),a*4+12,t.m);
    }
    float radiusSquared=0.0f;
    for(uint32_t a=0;a<3;++a) {
        Min(F(183616),F(a*4096),F((a+3)*4096),t.m);
        ReduceMin(F(179360,64),F(183616),F(143360,3328),t.m,false);
        Max(F(183616),F(a*4096),F((a+3)*4096),t.m);
        ReduceMax(F(179392,64),F(183616),F(143360,3328),t.m,false);
        Hop<HardEvent::V_S>();
        low[a]=F(179360,64).GetValue(0);
        const float extent=F(179392,64).GetValue(0)-low[a];
        if(!(extent>0.0001f && extent<1.0e10f)) return false;
        width[a]=extent*(1.0f/7.0f); radiusSquared+=width[a]*width[a]*0.25f;
        Hop<HardEvent::S_V>();
        Sub(F((a+3)*4096),F((a+3)*4096),F(a*4096),t.m);
    }
    Mul(F(76800),F(12288),F(12288),t.m);
    MulAddDst(F(76800),F(16384),F(16384),t.m);
    MulAddDst(F(76800),F(20480),F(20480),t.m);
    LocalTensor<uint8_t> mask(TPosition::VECCALC,156672,512);
    CompareScalar(mask,F(76800),0.0f,CMPMODE::NE,t.m);
    Select(F(76800),mask,F(76800),1.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,t.m);
    Duplicate(F(90112),-1.0f,t.m); Div(F(90112),F(90112),F(76800),t.m);
    for(uint32_t a=0;a<3;++a) Mul(F((a+6)*4096),F((a+3)*4096),F(90112),t.m);
    Hop<HardEvent::V_S>(); radius=TinyRoot(radiusSquared); Hop<HardEvent::S_V>();
    Duplicate(F(76800,3328),0.0f,owned); Duplicate(F(90112,3328),0.0f,owned);
    for(uint32_t a=0;a<3;++a) {
        const auto point=F(36864+a*13312,3328);
        auto temp=F(143360,3328),gap=F(103424,3328);
        Adds(temp,point,-low[a],owned); Muls(temp,temp,1.0f/width[a],owned);
        Maxs(temp,temp,0.0f,owned); Mins(temp,temp,6.0f,owned);
        Cast(I(130048,3328),temp,RoundMode::CAST_FLOOR,owned);
        Cast(temp,I(130048,3328),RoundMode::CAST_NONE,owned);
        Muls(F(76800,3328),F(76800,3328),7.0f,owned);
        Add(F(76800,3328),F(76800,3328),temp,owned);
        Muls(gap,temp,width[a],owned); Adds(gap,gap,low[a]+width[a]*0.5f,owned);
        Sub(gap,point,gap,owned); MulAddDst(F(90112,3328),gap,gap,owned);
    }
    Sqrt(F(90112,3328),F(90112,3328),owned);
    Muls(F(90112,3328),F(90112,3328),-1.0f,owned);
    Adds(F(90112,3328),F(90112,3328),radius*0.9999f-0.01f,owned);
    Maxs(F(90112,3328),F(90112,3328),0.0f,owned);
    Mul(F(90112,3328),F(90112,3328),F(90112,3328),owned);
    Hop<HardEvent::V_S>();
    for(uint32_t q=owned;q<AlignLanes(owned);++q) F(76800,3328).SetValue(q,343.0f);
    CreateVecIndex(F(130048,3328),0.0f,AlignLanes(owned));
    return true;
}
}
