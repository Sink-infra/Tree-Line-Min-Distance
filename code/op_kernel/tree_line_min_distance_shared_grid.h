#pragma once

namespace SharedGridSearch {
constexpr uint32_t kArenaBytes = 188416;
constexpr uint32_t kPoints = 3328;
constexpr uint32_t kGroup = 56;
constexpr uint32_t kMeta = 72;
static_assert(kArenaBytes <= 192 * 1024 - 8 * 1024);
__aicore__ inline LocalTensor<float> F(uint32_t byte, uint32_t size = 1024)
{ return LocalTensor<float>(TPosition::VECCALC, byte, size); }
__aicore__ inline LocalTensor<uint8_t> Mask(uint32_t number)
{ return LocalTensor<uint8_t>(TPosition::VECCALC, 184320 + number * 512, 512); }
__aicore__ inline uint32_t Round8(uint32_t n) { return (n + 7) / 8 * 8; }
__aicore__ inline uint32_t Compact(const LocalTensor<float>& dst,
    const LocalTensor<float>& src, const LocalTensor<uint8_t>& mask, uint32_t n)
{
    uint64_t kept = 0;
    GatherMask(dst, src, mask.ReinterpretCast<uint32_t>(), true, n, {1, 1, 0, 0}, kept);
    return (uint32_t)kept;
}

class Search {
    GlobalTensor<float> points, segments, distances, work;
    GlobalTensor<int32_t> indices;
    GlobalTensor<uint32_t> metadata, workBits;
    TreeLineMinDistanceTilingData t;
    uint32_t blocks, owner, batch, begin, owned, capacity, wideM;
    float low[3], step[3], extent, coordinate, pointBound, margin;
    bool stored = false;

    __aicore__ inline LocalTensor<uint32_t> Meta()
    { return F(172800, 2880).ReinterpretCast<uint32_t>(); }
    __aicore__ inline LocalTensor<uint32_t> SelfMeta()
    { return F(185344, 72).ReinterpretCast<uint32_t>(); }
    __aicore__ inline uint64_t Stream(uint32_t source, uint32_t field, uint32_t offset = 0)
    { return ((uint64_t)source * 6 + field) * capacity + offset; }
    __aicore__ inline uint32_t Entry(uint32_t source, uint32_t cell)
    { return Meta().GetValue(source * kMeta + cell); }
    __aicore__ inline bool BatchValid(bool final = false)
    {
        DataCopyPad(Meta(), metadata, {1, blocks * kMeta * 4, 0, 0, 0}, {false, 0, 0, 0});
        Hop<HardEvent::MTE2_S>();
        for (uint32_t q = batch*t.chunks; q < (batch+1)*t.chunks; ++q)
            if (Meta().GetValue(q * kMeta + (final ? 67 : 64)) == 0) return false;
        return true;
    }
    __aicore__ inline void SaveMeta(bool valid, bool final = false)
    {
        SelfMeta().SetValue(final ? 67 : 64, valid ? 1u : 0u);
        Hop<HardEvent::S_MTE3>();
        DataCopyPad(metadata[owner * kMeta], SelfMeta(), {1, kMeta * 4, 0, 0, 0});
    }
    __aicore__ inline bool PrepareSegments()
    {
        auto raw = F(98304, 6144), scratch = F(159744), pair = F(171264, 128);
        auto offsets = F(155648).ReinterpretCast<int32_t>();
        DataCopyPad(raw, segments[(uint64_t)batch * t.m * 6],
            {1, t.m * 24, 0, 0, 0}, {false, 0, 0, 0});
        CreateVecIndex(scratch, 0.0f, wideM);
        Muls(scratch, scratch, 24.0f, wideM);
        Cast(offsets, scratch, RoundMode::CAST_RINT, wideM);
        Hop<HardEvent::MTE2_V>();
        Hop<HardEvent::MTE2_S>();
        if (TreeDistanceSpatial::PreferDenseSample(raw,t.m)==0) return false;
        Hop<HardEvent::S_V>();
        for (uint32_t axis = 0; axis < 3; ++axis) {
            auto a = F(axis * 4096), u = F(12288 + axis * 4096);
            auto lo = F(36864 + axis * 4096), hi = F(49152 + axis * 4096);
            Duplicate(a, kGuardCoord, wideM); Duplicate(u, 0.0f, wideM);
            Gather(a, raw, offsets.ReinterpretCast<uint32_t>(), axis * 4, t.m);
            Gather(u, raw, offsets.ReinterpretCast<uint32_t>(), axis * 4 + 12, t.m);
            Duplicate(lo, kGuardDist, wideM); Duplicate(hi, -kGuardDist, wideM);
            Min(lo, a, u, t.m); Max(hi, a, u, t.m);
            ReduceMin(pair[axis * 16], lo, scratch, wideM, false);
            ReduceMax(pair[axis * 16 + 8], hi, scratch, wideM, false);
            Sub(u, u, a, t.m);
        }
        Hop<HardEvent::V_S>();
        bool valid = true;
        extent = 0.0f; coordinate = 0.0f;
        for (uint32_t axis = 0; axis < 3; ++axis) {
            low[axis] = pair.GetValue(axis * 16);
            const float high = pair.GetValue(axis * 16 + 8), length = high - low[axis];
            valid = valid && length > 1.0e-6f && length < 1.0e10f;
            step[axis] = length * 0.25f;
            extent = length > extent ? length : extent;
            const float a = low[axis] < 0.0f ? -low[axis] : low[axis];
            const float z = high < 0.0f ? -high : high;
            coordinate = a > coordinate ? a : coordinate;
            coordinate = z > coordinate ? z : coordinate;
        }
        if (!(coordinate < 1.0e10f) || !valid) return false;
        Hop<HardEvent::S_V>();
        Mul(scratch, F(12288), F(12288), wideM);
        MulAddDst(scratch, F(16384), F(16384), wideM);
        MulAddDst(scratch, F(20480), F(20480), wideM);
        CompareScalar(Mask(0), scratch, 0.0f, CMPMODE::NE, wideM);
        Select(scratch, Mask(0), scratch, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, wideM);
        Duplicate(F(163840), -1.0f, wideM);
        Div(F(167936), F(163840), scratch, wideM);
        for (uint32_t axis = 0; axis < 3; ++axis)
            Mul(F(24576 + axis * 4096), F(12288 + axis * 4096), F(167936), wideM);
        PipeBarrier<PIPE_V>();
        for (uint32_t axis = 0; axis < 3; ++axis) {
            auto direction=F(12288+axis*4096), inverse=F(36864+axis*4096);
            Abs(scratch,direction,wideM);
            CompareScalar(Mask(0),scratch,1.0e-10f,CMPMODE::GT,wideM);
            Select(scratch,Mask(0),direction,1.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,wideM);
            Div(inverse,F(163840),scratch,wideM);
            Select(inverse,Mask(0),inverse,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,wideM);
        }
        return true;
    }
#include "tree_line_min_distance_shared_pack.h"
#include "tree_line_min_distance_shared_query.h"
    __aicore__ inline void RestoreOrder()
    {
        if (owned==0) return;
        const uint32_t packed=Meta().GetValue(owner*kMeta+65);
        auto packedD=F(61440,3840), packedI=F(76800,3840);
        auto inverse=F(92160,kPoints).ReinterpretCast<uint32_t>();
        DataCopyPad(packedD,work[Stream(owner,3)],{1,packed*4,0,0,0},{false,0,0,0});
        DataCopyPad(packedI,work[Stream(owner,4)],{1,packed*4,0,0,0},{false,0,0,0});
        DataCopyPad(inverse,workBits[Stream(owner,5)],
            {1,owned*4,0,0,0},{false,0,0,0});
        Hop<HardEvent::MTE2_V>();
        Gather(F(105472,kPoints),packedD,inverse,0u,owned);
        Gather(F(118784,kPoints),packedI,inverse,0u,owned);
        Cast(F(118784,kPoints).ReinterpretCast<int32_t>(),F(118784,kPoints),RoundMode::CAST_RINT,owned);
        Hop<HardEvent::V_MTE3>();
        const uint64_t out=(uint64_t)batch*t.n+begin;
        DataCopyPad(distances[out],F(105472,kPoints),{1,owned*4,0,0,0});
        DataCopyPad(indices[out],F(118784,kPoints).ReinterpretCast<int32_t>(),{1,owned*4,0,0,0});
    }
public:
    __aicore__ inline bool Run(GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,GM_ADDR workspace,
                              const TreeLineMinDistanceTilingData& shape)
    {
        t=shape;blocks=GetBlockNum();owner=GetBlockIdx();
        if (blocks!=t.b*t.chunks || blocks>40 || t.chunkPoints>kPoints) return false;
        batch=owner/t.chunks;begin=(owner%t.chunks)*t.chunkPoints;
        owned=begin<t.n?(t.n-begin<t.chunkPoints?t.n-begin:t.chunkPoints):0;
        capacity=Round8(t.chunkPoints)+512;wideM=AlignLanes(t.m);
        points.SetGlobalBuffer((__gm__ float*)p);segments.SetGlobalBuffer((__gm__ float*)s);
        distances.SetGlobalBuffer((__gm__ float*)d);indices.SetGlobalBuffer((__gm__ int32_t*)i);
        work.SetGlobalBuffer((__gm__ float*)workspace);
        workBits.SetGlobalBuffer((__gm__ uint32_t*)workspace);
        metadata.SetGlobalBuffer((__gm__ uint32_t*)((__gm__ float*)workspace+(uint64_t)blocks*6*capacity));
        Duplicate(F(185344,72),0.0f,72);Hop<HardEvent::V_S>();
        pointBound=0.0f;
        const bool valid=PrepareSegments()&&PackPoints();
        union FloatBits {float f;uint32_t u;} bound;bound.f=pointBound;
        SelfMeta().SetValue(66,bound.u);
        SaveMeta(valid);
        PipeBarrier<PIPE_ALL>();SyncAll();
        const bool gridReady=BatchValid();
        if (!gridReady) {
            bool anyPrepared=false;
            for (uint32_t q=0;q<blocks;++q)
                anyPrepared=anyPrepared||Meta().GetValue(q*kMeta+64)!=0;
            if (!anyPrepared) return false;
        }
        bool complete=true;
        if (gridReady) {
            PrepareSegments();
            for (uint32_t q=0;q<t.chunks;++q) {
                bound.u=Meta().GetValue((batch*t.chunks+q)*kMeta+66);
                pointBound=bound.f>pointBound?bound.f:pointBound;
            }
            margin=0.00000762939453125f*(coordinate+pointBound+extent)+0.0001f;
            stored=false;
            complete=QueryCells();
        } else {
            PipeBarrier<PIPE_ALL>();
            if (!TreeDistanceSpatial::Run<float,true>(p,s,d,i,t)) {
                PipeBarrier<PIPE_ALL>();
                DensePointTask<float>(p,s,d,i,t);
            }
            PipeBarrier<PIPE_ALL>();
            Duplicate(F(185344,72),0.0f,72);
            Hop<HardEvent::V_S>();
            SelfMeta().SetValue(64,valid?1u:0u);
        }
        SaveMeta(complete,true);
        PipeBarrier<PIPE_ALL>();SyncAll();
        const bool exactBatch=BatchValid(true);
        if (!gridReady) return true;
        if (!exactBatch) return false;
        RestoreOrder();
        return true;
    }
};
} // namespace SharedGridSearch
