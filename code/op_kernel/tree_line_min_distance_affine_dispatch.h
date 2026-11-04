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
        KernelPointGroups<T, 3, true, false, true, true> op;
        op.Init(p, s, d, i, t, &pipe);
        op.Process();
    } else {
        KernelPointGroups<T, 3, true, false, true, true, false, false, true> op;
        op.Init(p, s, d, i, t, &pipe);
        op.Process();
    }
}
