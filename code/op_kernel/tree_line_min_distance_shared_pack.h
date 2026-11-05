#pragma once

    __aicore__ inline bool PackPoints()
    {
        if (owned == 0) return true;
        const uint32_t wide = AlignLanes(owned);
        auto raw = F(0, kPoints * 3);
        auto offsets = F(79872, kPoints).ReinterpretCast<int32_t>();
        Hop<HardEvent::V_MTE2>();
        auto input=F(133120,kPoints*3).template ReinterpretCast<T>();
        if constexpr (sizeof(T)==4) input=raw.template ReinterpretCast<T>();
        DataCopyPad(input, points[((uint64_t)batch * t.n + begin) * 3],
            {1, (uint32_t)(owned * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        CreateVecIndex(F(93184, kPoints), 0.0f, wide);
        Muls(F(93184, kPoints), F(93184, kPoints), 12.0f, wide);
        Cast(offsets, F(93184, kPoints), RoundMode::CAST_RINT, wide);
        Hop<HardEvent::MTE2_V>();
        if constexpr (sizeof(T)==2) Cast(raw,input,RoundMode::CAST_NONE,owned*3);
        PipeBarrier<PIPE_V>();
        for (uint32_t axis = 0; axis < 3; ++axis) {
            auto out = F(39936 + axis * kPoints * 4, kPoints);
            Duplicate(out, 0.0f, wide);
            Gather(out, raw, offsets.ReinterpretCast<uint32_t>(), axis * 4, owned);
        }
        PipeBarrier<PIPE_V>();
        Abs(F(93184, kPoints), F(39936, kPoints), wide);
        Abs(F(106496, kPoints), F(53248, kPoints), wide);
        Max(F(93184, kPoints), F(93184, kPoints), F(106496, kPoints), wide);
        Abs(F(106496, kPoints), F(66560, kPoints), wide);
        Max(F(93184, kPoints), F(93184, kPoints), F(106496, kPoints), wide);
        ReduceMax(F(123904, 128), F(93184, kPoints), F(119808, kPoints), wide, false);
        Hop<HardEvent::V_S>();
        pointBound = F(123904, 128).GetValue(0);
        if (!(pointBound < 1.0e10f)) return false;
        margin = 0.00000762939453125f * (coordinate + pointBound + extent) + 0.0001f;
        Hop<HardEvent::S_V>();
        auto cells = F(0, kPoints);
        Duplicate(cells, 0.0f, wide);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            Adds(F(93184, kPoints), F(39936 + axis * kPoints * 4, kPoints), -low[axis], wide);
            Muls(F(93184, kPoints), F(93184, kPoints), 1.0f / step[axis], wide);
            Maxs(F(93184, kPoints), F(93184, kPoints), 0.0f, wide);
            Mins(F(93184, kPoints), F(93184, kPoints), FINE && axis == 2 ? 7.0f : 3.0f, wide);
            Cast(F(106496, kPoints).ReinterpretCast<int32_t>(), F(93184, kPoints), RoundMode::CAST_FLOOR, wide);
            Cast(F(93184, kPoints), F(106496, kPoints).ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, wide);
            Muls(F(93184, kPoints), F(93184, kPoints), (float)(int32_t)(1u << (axis * 2)), wide);
            Add(cells, cells, F(93184, kPoints), wide);
        }
        auto order=F(13312,kPoints).ReinterpretCast<uint32_t>();
        auto inverse=F(26624,kPoints).ReinterpretCast<uint32_t>();
        CreateVecIndex(order.ReinterpretCast<int32_t>(),0,wide);
        Muls(cells,cells,-1.0f,wide);
        Hop<HardEvent::V_S>();
        for (uint32_t q=owned;q<wide;++q) cells.SetValue(q,-1.0e30f);
        Hop<HardEvent::S_V>();
        auto concat=F(79872,kPoints*2), sorted=F(106496,kPoints*2), sortTmp=F(133120,kPoints*2);
        Concat(concat,cells,F(159744,64),wide/16);
        Sort<float,true>(sorted,concat,order,sortTmp,wide/32);
        Extract(cells,order,sorted,wide/32);
        Hop<HardEvent::V_S>();
        uint32_t cursor=0;
        for (uint32_t cell=0;cell<kCells;++cell) {
            const uint32_t start=cursor;
            uint32_t left=cursor,right=owned;
            const float key=-(float)(int32_t)cell;
            while (left<right) {
                const uint32_t middle=left+(right-left)/2;
                if (cells.GetValue(middle)>=key) left=middle+1;
                else right=middle;
            }
            cursor=left;
            if constexpr (FINE) SelfMeta().template ReinterpretCast<uint16_t>().SetValue(cell,(uint16_t)start);
            else SelfMeta().SetValue(cell,(start<<12)|(cursor-start));
        }
        if constexpr (FINE) SelfMeta().template ReinterpretCast<uint16_t>().SetValue(kCells,(uint16_t)cursor);
        if (cursor!=owned) return false;
        Hop<HardEvent::S_V>();
        Cast(cells,order.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,wide);
        Muls(cells,cells,-1.0f,wide);
        CreateVecIndex(inverse.ReinterpretCast<int32_t>(),0,wide);
        Concat(concat,cells,F(159744,64),wide/16);
        Sort<float,true>(sorted,concat,inverse,sortTmp,wide/32);
        Extract(cells,inverse,sorted,wide/32);
        Cast(cells,inverse.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,wide);
        Muls(cells,cells,4.0f,wide);
        Cast(inverse.ReinterpretCast<int32_t>(),cells,RoundMode::CAST_RINT,wide);
        SelfMeta().SetValue(kPacked,owned);
        Hop<HardEvent::S_V>();
        Cast(cells,order.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,wide);
        Muls(cells,cells,4.0f,wide);
        Cast(order.ReinterpretCast<int32_t>(),cells,RoundMode::CAST_RINT,wide);
        for (uint32_t axis=0;axis<3;++axis)
            Gather(F(79872+axis*kPoints*4,kPoints),F(39936+axis*kPoints*4,kPoints),order,0u,owned);
        Hop<HardEvent::V_MTE3>();
        for (uint32_t axis=0;axis<3;++axis)
            DataCopyPad(work[Stream(owner,axis)],F(79872+axis*kPoints*4,kPoints),{1,owned*4,0,0,0});
        Hop<HardEvent::S_MTE3>();
        DataCopyPad(workBits[Stream(owner,5)],inverse,{1,owned*4,0,0,0});
        stored=true;
        return true;
    }
