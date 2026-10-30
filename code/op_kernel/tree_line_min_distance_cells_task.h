#pragma once
namespace TreeDistanceCells {
template<typename T>
__aicore__ inline void Task(GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,GM_ADDR workspace,
    const TreeLineMinDistanceTilingData& t)
{
    {
        TPipe gridPipe;
        TBuf<TPosition::VECCALC> gridStorage;
        gridPipe.InitBuffer(gridStorage,188416);
        if(Run<T>(p,s,d,i,workspace,t)) return;
    }
    PipeBarrier<PIPE_ALL>();
    {
        TPipe candidatePipe;
        TBuf<TPosition::VECCALC> candidateStorage;
        candidatePipe.InitBuffer(candidateStorage,150528);
        if(TreeDistanceSpatial::Run<T,true>(p,s,d,i,t)) return;
    }
    PipeBarrier<PIPE_ALL>();
    TPipe pipe;
    if(t.chunkPoints<1536) {
        KernelPointGroups<T,3,true,false,true,true,true,false,true> op;
        op.Init(p,s,d,i,t,&pipe);
        op.Process();
    } else {
        TBuf<TPosition::VECCALC> denseStorage;
        pipe.InitBuffer(denseStorage,188416);
        DensePointTask<T>(p,s,d,i,t);
    }
}
}
