#pragma once

// Refine the reciprocal root using multiply chains, then recover the root.
// The zero test precedes refinement so degenerate distances stay exactly zero.
template <uint32_t STEPS>
__aicore__ inline float TinyRootRefined(float x)
{
    if (!(x > 0.0f)) return 0.0f;
    union Bits { float value; uint32_t bits; } estimate;
    estimate.value = x;
    estimate.bits = 0x5f375a86u - (estimate.bits >> 1);
    float reciprocal = estimate.value;
    #pragma unroll
    for (uint32_t step = 0; step < STEPS; ++step) {
        reciprocal = reciprocal * (1.5f - ((0.5f * x) * reciprocal) * reciprocal);
    }
    return x * reciprocal;
}

__aicore__ inline float TinyRoot(float x)
{
    return TinyRootRefined<3>(x);
}

__aicore__ inline float TinyRootTwoSteps(float x)
{
    return TinyRootRefined<2>(x);
}
