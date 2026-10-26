#pragma once

__aicore__ inline bool OrderedBounds(const LocalTensor<float>& prefix,
                                     const LocalTensor<float>& suffix,
                                     uint32_t cells)
{
    if (cells < 2) return true;
    const uint32_t count = cells - 1;
    const uint32_t wide = AlignLanes(count);
    LocalTensor<float> shifted(TPosition::VECCALC, 80000, 1024);
    LocalTensor<float> difference(TPosition::VECCALC, 84128, 1024);
    LocalTensor<int32_t> offsets(TPosition::VECCALC, 88256, 1024);
    LocalTensor<float> minima(TPosition::VECCALC, 92384, 64);
    Hop<HardEvent::S_V>();
    CreateVecIndex(shifted, 0.0f, wide);
    Muls(shifted, shifted, 4.0f, wide);
    Cast(offsets, shifted, RoundMode::CAST_RINT, wide);
    PipeBarrier<PIPE_V>();
    for (uint32_t side = 0; side < 2; ++side) {
        if (side != 0) Hop<HardEvent::S_V>();
        LocalTensor<float> bound = side == 0 ? prefix : suffix;
        Duplicate(difference, kGuardDist, wide);
        Gather(shifted, bound, offsets.ReinterpretCast<uint32_t>(), 4u, count);
        Sub(difference, shifted, bound, count);
        WholeReduceMin(minima, difference, 64, (int32_t)(wide / 64), 1, 1, 8);
        Hop<HardEvent::V_S>();
        for (uint32_t group = 0; group < wide / 64; ++group) {
            if (!(minima.GetValue(2 * group) >= 0.0f)) return false;
        }
    }
    return true;
}
