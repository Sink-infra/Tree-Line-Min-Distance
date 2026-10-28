#pragma once

namespace TreeDistanceSpatial {
__aicore__ inline LocalTensor<int32_t> PreparePlanarCells(const LocalTensor<float>& coordinates,
    const LocalTensor<int32_t>& heads, const LocalTensor<int32_t>& sizes,
    const uint32_t* axes, const float* origin, const float* scale, uint32_t side, uint32_t count)
{
    LocalTensor<float> cell[2] = {
        LocalTensor<float>(TPosition::VECCALC, 104576, 1024),
        LocalTensor<float>(TPosition::VECCALC, 108672, 1024)};
    LocalTensor<int32_t> offsets(TPosition::VECCALC, 112768, 1024);
    LocalTensor<float> temporary(TPosition::VECCALC, 116864, 1024);
    LocalTensor<int32_t> ids(TPosition::VECCALC, 122880, 1024);
    Hop<HardEvent::S_V>();
    CreateVecIndex(temporary, 0.0f, count); Muls(temporary, temporary, 24.0f, count);
    Cast(offsets, temporary, RoundMode::CAST_RINT, count);
    const auto bytes = offsets.ReinterpretCast<uint32_t>();
    for (uint32_t k = 0; k < 2; ++k) {
        Gather(cell[k], coordinates, bytes, axes[k] * 4, count);
        Gather(temporary, coordinates, bytes, axes[k] * 4 + 12, count);
        Muls(cell[k], cell[k], 0.5f, count); Axpy(cell[k], temporary, 0.5f, count);
        Adds(cell[k], cell[k], -origin[k], count); Muls(cell[k], cell[k], scale[k], count);
        Adds(cell[k], cell[k], 0.5f, count);
        Maxs(cell[k], cell[k], 0.0f, count); Mins(cell[k], cell[k], (float)(int32_t)(side - 1), count);
        Cast(ids, cell[k], RoundMode::CAST_FLOOR, count);
        Cast(cell[k], ids, RoundMode::CAST_NONE, count);
    }
    Muls(cell[0], cell[0], (float)(int32_t)side, count);
    Add(temporary, cell[0], cell[1], count);
    Cast(ids, temporary, RoundMode::CAST_RINT, count);
    Duplicate(heads, (int32_t)-1, side * side);
    Duplicate(sizes, (int32_t)0, side * side);
    Hop<HardEvent::V_S>();
    return ids;
}
}
