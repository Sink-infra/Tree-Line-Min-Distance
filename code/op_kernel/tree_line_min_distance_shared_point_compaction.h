#pragma once

    __aicore__ inline void CompactPointGroup(uint32_t span,uint32_t real)
    {
        CompareScalar(Mask(0),F(165888,64),0.0f,CMPMODE::GT,64);
        uint64_t kept=0;
        for (uint32_t axis=0;axis<3;++axis)
            GatherMask(F(55296+axis*256,64),F(155904+axis*256,64),
                Mask(0).ReinterpretCast<uint32_t>(),true,span,{1,1,0,0},kept);
        Duplicate(F(165888,64),0.0f,64);
        Duplicate(F(165888,64),1.0f,real);
    }

    __aicore__ inline void ExpandPointGroup(uint32_t span,uint32_t real)
    {
        auto inverse=F(185632,64).ReinterpretCast<uint32_t>();
        Adds(F(49920,64),F(165120,64),0.0f,real);
        Adds(F(50176,64),F(165376,64),0.0f,real);
        Gather(F(165120,64),F(49920,64),inverse,0u,span);
        Gather(F(165376,64),F(50176,64),inverse,0u,span);
    }
    static_assert(185632+64*4<=185888);
    static_assert(185888+64*4<=186368);
    static_assert(98304+4096*4<=114816);
    static_assert(114816+4096*4<=131200);
    static_assert(131200+4096*4<=147712);
    static_assert(147712+4096*4<=164096);
    static_assert(164096+128*4<=164608);
    static_assert(165376+64*4<=165888);
    static_assert(165888+64*4<=166144);
    static_assert(166144+64*4<=168704);
    static_assert(53248+512*4<=55296);
    static_assert(55808+64*4<=61440);
