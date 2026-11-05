#pragma once

    __aicore__ inline void TailLanes(uint32_t span,uint32_t first)
    {
        auto px=arena[oPx],py=arena[oPy],pz=arena[oPz];
        auto dx=arena[oDx],dy=arena[oDy],dz=arena[oDz];
        auto ratio=arena[oTt],value=arena[oDsq];
        auto best=arena[oBest],winner=arena[oFidx],candidate=arena[oCidx];
        auto keep=arena[oMask].ReinterpretCast<uint8_t>();
        Hop<HardEvent::V_S>();
        for (uint32_t j=first;j<m;++j) {
            const float ax=arena[oAx].GetValue(j),ay=arena[oAy].GetValue(j),az=arena[oAz].GetValue(j);
            const float ux=arena[oUx].GetValue(j),uy=arena[oUy].GetValue(j),uz=arena[oUz].GetValue(j);
            const float kx=arena[oKx].GetValue(j),ky=arena[oKy].GetValue(j),kz=arena[oKz].GetValue(j);
            Adds(dx,px,-ax,span);Adds(dy,py,-ay,span);Adds(dz,pz,-az,span);
            // k=-u/L, so the point-oriented projection is clamped to[-1,0].
            Muls(ratio,dx,kx,span);Axpy(ratio,dy,ky,span);Axpy(ratio,dz,kz,span);
            Maxs(ratio,ratio,-1.0f,span);Mins(ratio,ratio,0.0f,span);
            Axpy(dx,ratio,ux,span);Axpy(dy,ratio,uy,span);Axpy(dz,ratio,uz,span);
            Mul(value,dx,dx,span);
            if constexpr (SPLIT) {
                Mul(dy,dy,dy,span);Mul(dz,dz,dz,span);
                Add(value,value,dy,span);Add(value,value,dz,span);
            } else {
                MulAddDst(value,dy,dy,span);MulAddDst(value,dz,dz,span);
            }
            Duplicate(candidate,(float)(int32_t)j,kLanes);
            Compare(keep,value,best,CMPMODE::GE,kLanes);
            Min(best,best,value,kLanes);
            Select(winner,keep,winner,candidate,SELMODE::VSEL_TENSOR_TENSOR_MODE,kLanes);
        }
    }
