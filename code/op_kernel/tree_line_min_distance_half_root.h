#pragma once
__aicore__ inline float TinyHalfRootOneStep(float x)
{
    if(!(x>0.0f)) return 0.0f;
    if(x>4290774016.0f) return TinyRootTwoSteps(x);
    union Bits { float value; uint32_t bits; } estimate;
    estimate.value=x;
    estimate.bits=(estimate.bits>>1)+0x1fbb7000u;
    float root=estimate.value;
    root=0.5f*(root+x/root);
    root*=0.9996966123580933f;
    return root<65504.0f?root:65504.0f;
}
