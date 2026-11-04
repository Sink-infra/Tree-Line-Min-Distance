#pragma once

template <typename T>
__aicore__ inline void AffineOrFallback(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                                       const TreeLineMinDistanceTilingData& t)
{
    {
        TPipe modelPipe;
        TBuf<TPosition::VECCALC> modelStorage;
        modelPipe.InitBuffer(modelStorage, 125184);
        if (AffineSegmentSearch::Run<T>(p, s, d, i, t)) return;
    }
    PipeBarrier<PIPE_ALL>();
    TPipe pipe;
    if (t.chunkPoints < 384) {
        auto grouped = t;
        grouped.group = HoistedPointGroupLayout::kPoints;
        grouped.tilePoints = HoistedPointGroupLayout::kPoints;
        KernelPointGroups<T, 3, true, false, true, true, false, false, false, true> op;
        op.Init(p, s, d, i, grouped, &pipe);
        op.Process();
    } else {
        KernelPointGroups<T, 3, true, false, true, true, false, false, true> op;
        op.Init(p, s, d, i, t, &pipe);
        op.Process();
    }
}
