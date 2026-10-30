#pragma once

__aicore__ inline float TinyRoot(float x)
{
    if (!(x > 0.0f)) return 0.0f;
    union Bits { float value; uint32_t bits; } estimate;
    estimate.value = x;
    estimate.bits = (estimate.bits >> 1) + 0x1fc00000u;
    float root = estimate.value;
    root = 0.5f * (root + x / root);
    root = 0.5f * (root + x / root);
    root = 0.5f * (root + x / root);
    return root;
}


__aicore__ inline float TinyRootTwoSteps(float x)
{
    if (!(x > 0.0f)) return 0.0f;
    union Bits { float value; uint32_t bits; } estimate;
    estimate.value = x;
    estimate.bits = (estimate.bits >> 1) + 0x1fc00000u;
    float root = estimate.value;
    root = 0.5f * (root + x / root);
    root = 0.5f * (root + x / root);
    return root;
}
