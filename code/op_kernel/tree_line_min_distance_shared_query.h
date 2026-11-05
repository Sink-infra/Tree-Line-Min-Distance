#pragma once

    __aicore__ inline uint32_t PrepareCandidates(uint32_t cell, float radius)
    {
        auto enter=F(49152), exit=F(53248), endpoint=F(57344);
        auto other=F(164608), near=F(98304);
        Duplicate(enter,0.0f,wideM);Duplicate(exit,1.0f,wideM);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const uint32_t last = FINE && axis == 2 ? 7u : 3u;
            const uint32_t at = (cell >> (axis * 2)) & last;
            const float left = at == 0 ? -kGuardCoord : low[axis] + at * step[axis] - radius - margin;
            const float right = at == last ? kGuardCoord : low[axis] + (at + 1) * step[axis] + radius + margin;
            auto start=F(axis*4096), inverse=F(36864+axis*4096);
            Adds(endpoint,start,-left,wideM);Mul(endpoint,endpoint,inverse,wideM);
            Adds(other,start,-right,wideM);Mul(other,other,inverse,wideM);
            Min(near,endpoint,other,wideM);Max(endpoint,endpoint,other,wideM);
            CompareScalar(Mask(1),inverse,0.0f,CMPMODE::NE,wideM);
            // Ignoring a tiny direction enlarges its interval conservatively.
            Select(near,Mask(1),near,-1.0e30f,SELMODE::VSEL_TENSOR_SCALAR_MODE,wideM);
            Select(endpoint,Mask(1),endpoint,1.0e30f,SELMODE::VSEL_TENSOR_SCALAR_MODE,wideM);
            Max(enter,enter,near,wideM);Min(exit,exit,endpoint,wideM);
        }
        Compare(Mask(0),enter,exit,CMPMODE::LE,wideM);
        CreateVecIndex(F(164608),0.0f,wideM);
        CompareScalar(Mask(1),F(164608),(float)(int32_t)t.m,CMPMODE::LT,wideM);
        And(Mask(0).ReinterpretCast<uint16_t>(),Mask(0).ReinterpretCast<uint16_t>(),
            Mask(1).ReinterpretCast<uint16_t>(),wideM/16);
        Muls(F(164608),F(164608),4.0f,wideM);
        const uint32_t count = Compact(F(168704), F(164608), Mask(0), wideM);
        if (count == 0) return 0;
        auto offsets = F(98304).ReinterpretCast<int32_t>();
        Cast(offsets, F(168704), RoundMode::CAST_RINT, count);
        for (uint32_t a = 0; a < 9; ++a) {
            Duplicate(F(61440 + a * 4096), a < 3 ? kGuardCoord : 0.0f, AlignLanes(count));
            Gather(F(61440 + a * 4096), F(a * 4096), offsets.ReinterpretCast<uint32_t>(), 0u, count);
        }
        PipeBarrier<PIPE_V>();
        return count;
    }
    __aicore__ inline float ComputeGroup(uint32_t span, uint32_t candidates)
    {
        auto dx=F(98304,3584), dy=F(112768,3584), dz=F(127104,3584), ratio=F(141568,3584);
        auto pair=F(163072,128), cmin=F(163584,64), cidx=F(163840,64);
        auto best=F(164096,64), winner=F(164352,64);
        auto bx=F(156672,512), by=F(158720,512), bz=F(160768,512);
        Brcb(bx,F(155904,64),(span+7)/8,{1,8});
        Brcb(by,F(156160,64),(span+7)/8,{1,8});
        Brcb(bz,F(156416,64),(span+7)/8,{1,8});
        Duplicate(best,kGuardDist,64); Duplicate(winner,0.0f,64);
        const BinaryRepeatParams spread(1,1,0,8,0,1), fold(1,1,1,8,8,0), plain(1,1,1,8,8,8);
        const UnaryRepeatParams unary(1,1,8,8);
        for (uint32_t at=0;at<AlignLanes(candidates);at+=64) {
            SetVectorMask<float>(64);
            Sub<float,false>(dx,F(61440)[at],bx,MASK_PLACEHOLDER,span,spread);
            Sub<float,false>(dy,F(65536)[at],by,MASK_PLACEHOLDER,span,spread);
            Sub<float,false>(dz,F(69632)[at],bz,MASK_PLACEHOLDER,span,spread);
            Mul<float,false>(ratio,dx,F(86016)[at],MASK_PLACEHOLDER,span,fold);
            MulAddDst<float,float,false>(ratio,dy,F(90112)[at],MASK_PLACEHOLDER,span,fold);
            MulAddDst<float,float,false>(ratio,dz,F(94208)[at],MASK_PLACEHOLDER,span,fold);
            Maxs<float,false>(ratio,ratio,0.0f,MASK_PLACEHOLDER,span,unary);
            Mins<float,false>(ratio,ratio,1.0f,MASK_PLACEHOLDER,span,unary);
            MulAddDst<float,float,false>(dx,ratio,F(73728)[at],MASK_PLACEHOLDER,span,fold);
            MulAddDst<float,float,false>(dy,ratio,F(77824)[at],MASK_PLACEHOLDER,span,fold);
            MulAddDst<float,float,false>(dz,ratio,F(81920)[at],MASK_PLACEHOLDER,span,fold);
            Mul<float,false>(dx,dx,dx,MASK_PLACEHOLDER,span,plain);
            MulAddDst<float,float,false>(dx,dy,dy,MASK_PLACEHOLDER,span,plain);
            MulAddDst<float,float,false>(dx,dz,dz,MASK_PLACEHOLDER,span,plain);
            ResetMask();
            WholeReduceMin(pair,dx,64,span,1,1,8);
            uint64_t ignored=0;
            GatherMask(cmin,pair,(uint8_t)1,false,0,{1,(uint8_t)((span+31)/32),8,8},ignored);
            GatherMask(cidx,pair,(uint8_t)2,false,0,{1,(uint8_t)((span+31)/32),8,8},ignored);
            Cast(pair,cidx.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);
            Adds(pair,pair,(float)(int32_t)at,64);
            Compare(Mask(0),cmin,best,CMPMODE::GE,64);
            Min(best,best,cmin,64);
            Select(winner,Mask(0),winner,pair,SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
        }
        Muls(pair,winner,4.0f,64);
        Cast(cidx.ReinterpretCast<int32_t>(),pair,RoundMode::CAST_RINT,64);
        Gather(winner,F(168704),cidx.ReinterpretCast<uint32_t>(),0u,span);
        Muls(winner,winner,0.25f,span);
        Sqrt(best,best,span);
        CompareScalar(Mask(0),F(162816,64),0.0f,CMPMODE::GT,64);
        Select(pair,Mask(0),best,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
        ReduceMax(cmin,pair,F(164608),64,false);
        Hop<HardEvent::V_S>();
        return cmin.GetValue(0);
    }
    __aicore__ inline void LoadGroup(uint32_t cell,uint32_t groupStart,uint32_t span)
    {
        if (stored) Hop<HardEvent::MTE3_V>();
        for (uint32_t axis=0;axis<3;++axis) Duplicate(F(155904+axis*256,64),0.0f,64);
        Duplicate(F(162816,64),0.0f,64);
        Hop<HardEvent::V_MTE2>();
        uint32_t position=0;
        for (uint32_t q=0;q<t.chunks;++q) {
            const uint32_t source=batch*t.chunks+q, entry=Entry(source,cell);
            const uint32_t count=entry&4095u, end=position+count;
            const uint32_t left=position>groupStart?position:groupStart;
            const uint32_t right=end<groupStart+span?end:groupStart+span;
            if (right>left) {
                const uint32_t local=left-groupStart, length=right-left, offset=(entry>>12)+left-position;
                Duplicate(F(162816,64)[local],1.0f,length);
                for (uint32_t axis=0;axis<3;++axis)
                    DataCopyPad(F(155904+axis*256,64)[local],work[Stream(source,axis,offset)],
                        {1,length*4,0,0,0},{false,0,0,0});
            }
            position+=Round8(count);
        }
        Hop<HardEvent::MTE2_V>();
    }
    __aicore__ inline void StoreGroup(uint32_t cell,uint32_t groupStart,uint32_t span)
    {
        Hop<HardEvent::V_MTE3>();
        uint32_t position=0;
        for (uint32_t q=0;q<t.chunks;++q) {
            const uint32_t source=batch*t.chunks+q,entry=Entry(source,cell);
            const uint32_t count=entry&4095u,end=position+count;
            const uint32_t left=position>groupStart?position:groupStart;
            const uint32_t right=end<groupStart+span?end:groupStart+span;
            if (right>left) {
                const uint32_t local=left-groupStart,length=right-left,offset=(entry>>12)+left-position;
                DataCopyPad(work[Stream(source,3,offset)],F(164096,64)[local],{1,length*4,0,0,0});
                DataCopyPad(work[Stream(source,4,offset)],F(164352,64)[local],{1,length*4,0,0,0});
            }
            position+=Round8(count);
        }
        stored=true;
    }
    __aicore__ inline bool QueryHistorical()
    {
        bool valid=true;
        for (uint32_t ordinal=owner%t.chunks;ordinal<64;ordinal+=t.chunks) {
            const uint32_t x=(ordinal&1u)|((ordinal>>2)&2u);
            const uint32_t y=((ordinal>>1)&1u)|((ordinal>>3)&2u);
            const uint32_t z=((ordinal>>2)&1u)|((ordinal>>4)&2u);
            const uint32_t cell=x|(y<<2)|(z<<4);
            uint32_t total=0;
            for (uint32_t q=0;q<t.chunks;++q) total+=Round8(Entry(batch*t.chunks+q,cell)&4095u);
            if (total==0) continue;
            float radius=extent*0.1f;
            bool certified=false;
            for (uint32_t attempt=0;attempt<2;++attempt) {
                if (stored) Hop<HardEvent::MTE3_V>();
                const uint32_t candidates=PrepareCandidates(cell,radius);
                if (candidates==0) {radius=4.0f*(coordinate+pointBound+extent);continue;}
                float maximum=0.0f;
                for (uint32_t at=0;at<total;at+=kGroup) {
                    const uint32_t span=total-at<kGroup?total-at:kGroup;
                    LoadGroup(cell,at,span);
                    const float value=ComputeGroup(span,candidates);
                    maximum=value>maximum?value:maximum;
                    StoreGroup(cell,at,span);
                }
                if (maximum<=radius) {certified=true;break;}
                radius=maximum+margin;
            }
            valid=valid&&certified;
        }
        return valid;
    }

    __aicore__ inline bool QueryCells()
    {
        if constexpr (!FINE) return QueryHistorical();
        else {
        bool valid=true;
        for (uint32_t ordinal=owner%t.chunks;ordinal<kCells;ordinal+=t.chunks) {
            const uint32_t x=(ordinal&1u)|((ordinal>>2)&2u);
            const uint32_t y=((ordinal>>1)&1u)|((ordinal>>3)&2u);
            const uint32_t z=((ordinal>>2)&1u)|((ordinal>>4)&2u)|(FINE?((ordinal>>4)&4u):0u);
            const uint32_t cell=x|(y<<2)|(z<<4);
            uint32_t total=0;
            for (uint32_t q=0;q<t.chunks;++q) total+=Round8(Entry(batch*t.chunks+q,cell)&4095u);
            if (total==0) continue;
            float radius=extent*0.1f;
            bool certified=false,firstPass=false;
            uint32_t retryCount=0;
            auto retries=F(186368,320).ReinterpretCast<uint32_t>();
            static_assert(320*kGroup>=16384+7*40);
            static_assert(186368+320*4<=kArenaBytes);
            for (uint32_t attempt=0;attempt<2;++attempt) {
                if (stored) Hop<HardEvent::MTE3_V>();
                const uint32_t candidates=PrepareCandidates(cell,radius);
                if (candidates==0) {radius=4.0f*(coordinate+pointBound+extent);continue;}
                float maximum=0.0f;
                const bool partial=attempt!=0 && firstPass;
                const uint32_t groups=partial?retryCount:(total+kGroup-1)/kGroup;
                for (uint32_t group=0;group<groups;++group) {
                    const uint32_t at=partial?retries.GetValue(group):group*kGroup;
                    const uint32_t span=total-at<kGroup?total-at:kGroup;
                    LoadGroup(cell,at,span);
                    const float value=ComputeGroup(span,candidates);
                    if (attempt==0 && value>radius) retries.SetValue(retryCount++,at);
                    maximum=value>maximum?value:maximum;
                    StoreGroup(cell,at,span);
                }
                if (attempt==0) firstPass=true;
                if (maximum<=radius) {certified=true;break;}
                radius=maximum+margin;
            }
            valid=valid&&certified;
        }
        return valid;
            }
    }
