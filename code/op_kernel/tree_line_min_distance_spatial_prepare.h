#pragma once

__aicore__ inline LocalTensor<float> PreparedDirection(uint32_t axis)
{
    return LocalTensor<float>(TPosition::VECCALC, 92288 + axis * 4096, 1024);
}

__aicore__ inline void PrepareGeometry(const LocalTensor<float>& coordinates,
    const LocalTensor<float>& inverse, uint32_t count, float* low, float* high, float* width)
{
    LocalTensor<float> a[3] = {
        LocalTensor<float>(TPosition::VECCALC, 80000, 1024),
        LocalTensor<float>(TPosition::VECCALC, 84096, 1024),
        LocalTensor<float>(TPosition::VECCALC, 88192, 1024)};
    LocalTensor<float> c[3] = {PreparedDirection(0), PreparedDirection(1), PreparedDirection(2)};
    LocalTensor<float> temporary(TPosition::VECCALC, 104576, 1024);
    LocalTensor<float> scratch(TPosition::VECCALC, 108672, 1024);
    LocalTensor<int32_t> offsets(TPosition::VECCALC, 112768, 1024);
    LocalTensor<float> positions(TPosition::VECCALC, 116864, 1024);
    LocalTensor<float> reduced(TPosition::VECCALC, 120960, 128);
    Hop<HardEvent::S_V>();
    CreateVecIndex(positions, 0.0f, count); Muls(positions, positions, 24.0f, count);
    Cast(offsets, positions, RoundMode::CAST_RINT, count);
    const auto bytes = offsets.ReinterpretCast<uint32_t>();
    for (uint32_t axis = 0; axis < 3; ++axis) {
        Gather(a[axis], coordinates, bytes, 4u * axis, count);
        Gather(c[axis], coordinates, bytes, 12u + 4u * axis, count);
        Muls(temporary, a[axis], 0.5f, count); Axpy(temporary, c[axis], 0.5f, count);
        ReduceMin(reduced[axis * 16], temporary, scratch, count, false);
        ReduceMax(reduced[axis * 16 + 8], temporary, scratch, count, false);
        PipeBarrier<PIPE_V>();
        Sub(c[axis], c[axis], a[axis], count);
        Abs(temporary, c[axis], count);
        ReduceMax(reduced[64 + axis * 8], temporary, scratch, count, false);
        PipeBarrier<PIPE_V>();
    }
    Mul(temporary, c[0], c[0], count);
    MulAddDst(temporary, c[1], c[1], count); MulAddDst(temporary, c[2], c[2], count);
    LocalTensor<uint8_t> mask(TPosition::VECCALC, 121472, 256);
    CompareScalar(mask, temporary, 0.0f, CMPMODE::NE, count);
    Select(temporary, mask, temporary, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    Duplicate(scratch, 1.0f, count); Div(inverse, scratch, temporary, count);
    Hop<HardEvent::V_S>();
    for (uint32_t axis = 0; axis < 3; ++axis) {
        low[axis] = reduced.GetValue(axis * 16);
        high[axis] = reduced.GetValue(axis * 16 + 8);
        width[axis] = reduced.GetValue(64 + axis * 8);
    }
}
