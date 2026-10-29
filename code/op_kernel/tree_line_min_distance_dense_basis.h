#pragma once


__aicore__ inline void MakeLineBasis(uint32_t m,
 LocalTensor<float> ax,LocalTensor<float> ay,LocalTensor<float> az,
 LocalTensor<float> ux,LocalTensor<float> uy,LocalTensor<float> uz,
 LocalTensor<float> vx,LocalTensor<float> vy,LocalTensor<float> wx,
 LocalTensor<float> wy,LocalTensor<float> wz,LocalTensor<float> cv,
 LocalTensor<float> cw,LocalTensor<float> zero)
{
 const uint32_t wide=AlignLanes(m);
 LocalTensor<uint8_t> xyMask(TPosition::VECCALC,187904,128);
 LocalTensor<uint8_t> lenMask(TPosition::VECCALC,188032,128);
 Duplicate(cv,0.0f,wide);Duplicate(cw,0.0f,wide);
 Mul(cv,ux,ux,m);MulAddDst(cv,uy,uy,m);
 Add(cw,cv,cw,m);MulAddDst(cw,uz,uz,m);
 Sqrt(cv,cv,m);Sqrt(cw,cw,m);
 CompareScalar(xyMask,cv,0.0f,CMPMODE::NE,wide);
 CompareScalar(lenMask,cw,0.0f,CMPMODE::NE,wide);
 Duplicate(zero,0.0f,wide);
 Select(zero,lenMask,zero,1.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,wide);
 Maxs(cv,cv,1.0e-30f,wide);Maxs(cw,cw,1.0e-30f,wide);
 Div(vx,uy,cv,m);Muls(vx,vx,-1.0f,m);
 Div(vy,ux,cv,m);
 Select(vx,xyMask,vx,1.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,wide);
 Select(vy,xyMask,vy,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,wide);
 Mul(wx,uz,vy,m);Muls(wx,wx,-1.0f,m);Div(wx,wx,cw,m);
 Mul(wy,uz,vx,m);Div(wy,wy,cw,m);
 Div(wz,cv,cw,m);
 Select(wx,lenMask,wx,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,wide);
 Select(wy,lenMask,wy,1.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,wide);
 Select(wz,xyMask,wz,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,wide);
 Select(wz,lenMask,wz,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,wide);
 Mul(cv,ax,vx,m);MulAddDst(cv,ay,vy,m);
 Mul(cw,ax,wx,m);MulAddDst(cw,ay,wy,m);MulAddDst(cw,az,wz,m);
}



__aicore__ inline void VerifyInfiniteNative(
    uint32_t count,uint32_t wide,uint32_t m,
    LocalTensor<float> ax,LocalTensor<float> ay,LocalTensor<float> az,
    LocalTensor<float> ux,LocalTensor<float> uy,LocalTensor<float> uz,
    LocalTensor<float> kx,LocalTensor<float> ky,LocalTensor<float> kz,
    LocalTensor<float> px,LocalTensor<float> py,LocalTensor<float> pz,
    LocalTensor<float> best,LocalTensor<float> winner,
    LocalTensor<float> dx,LocalTensor<float> dy,LocalTensor<float> dz,
    LocalTensor<float> ratio,LocalTensor<float> dsq,LocalTensor<int32_t> offsets)
{
    Muls(dx,winner,4.0f,wide);Cast(offsets,dx,RoundMode::CAST_RINT,wide);
    auto off=offsets.ReinterpretCast<uint32_t>();
    Gather(dx,ax,off,0u,wide);Gather(dy,ay,off,0u,wide);Gather(dz,az,off,0u,wide);
    Add(dx,dx,px,wide);Add(dy,dy,py,wide);Add(dz,dz,pz,wide);
    Gather(ratio,kx,off,0u,wide);Mul(ratio,ratio,dx,wide);
    Gather(dsq,ky,off,0u,wide);MulAddDst(ratio,dsq,dy,wide);
    Gather(dsq,kz,off,0u,wide);MulAddDst(ratio,dsq,dz,wide);
    Gather(dsq,ux,off,0u,wide);MulAddDst(dx,ratio,dsq,wide);
    Gather(dsq,uy,off,0u,wide);MulAddDst(dy,ratio,dsq,wide);
    Gather(dsq,uz,off,0u,wide);MulAddDst(dz,ratio,dsq,wide);
    Mul(best,dx,dx,wide);MulAddDst(best,dy,dy,wide);MulAddDst(best,dz,dz,wide);
    PipeBarrier<PIPE_V>();
    auto bad=offsets.ReinterpretCast<uint8_t>();
    auto upper=bad[512],nan=bad[1024];
    CompareScalar(bad,ratio,1.0e-5f,CMPMODE::LE,wide);
    CompareScalar(upper,ratio,0.99999f,CMPMODE::GE,wide);
    Compare(nan,ratio,ratio,CMPMODE::NE,wide);
    Or(bad.ReinterpretCast<uint16_t>(),bad.ReinterpretCast<uint16_t>(),upper.ReinterpretCast<uint16_t>(),wide/16);
    Or(bad.ReinterpretCast<uint16_t>(),bad.ReinterpretCast<uint16_t>(),nan.ReinterpretCast<uint16_t>(),wide/16);
    Mul(dsq,px,px,wide);MulAddDst(dsq,py,py,wide);MulAddDst(dsq,pz,pz,wide);
    Muls(dsq,dsq,1.0e-12f,wide);Adds(dsq,dsq,1.0e-20f,wide);
    Compare(upper,best,dsq,CMPMODE::LE,wide);
    Or(bad.ReinterpretCast<uint16_t>(),bad.ReinterpretCast<uint16_t>(),upper.ReinterpretCast<uint16_t>(),wide/16);
    Hop<HardEvent::V_S>();CreateVecIndex(dx,0.0f,wide);Hop<HardEvent::S_V>();
    uint64_t invalid=0;
    GatherMask(dy,dx,bad.ReinterpretCast<uint32_t>(),true,count,{1,1,8,0},invalid);
    Hop<HardEvent::V_S>();
    auto fx=dx,fy=dx[1024],fz=dx[2048];
    auto work=dz,pair=dz[2048];
    for(uint32_t k=0;k<(uint32_t)invalid;++k){
        const uint32_t point=(uint32_t)(int32_t)dy.GetValue(k);
        const float nx=px.GetValue(point),ny=py.GetValue(point),nz=pz.GetValue(point);
        Adds(fx,ax,nx,m);Adds(fy,ay,ny,m);Adds(fz,az,nz,m);
        Mul(ratio,fx,kx,m);MulAddDst(ratio,fy,ky,m);MulAddDst(ratio,fz,kz,m);
        Maxs(ratio,ratio,0.0f,m);Mins(ratio,ratio,1.0f,m);
        MulAddDst(fx,ratio,ux,m);MulAddDst(fy,ratio,uy,m);MulAddDst(fz,ratio,uz,m);
        Mul(dsq,fx,fx,m);MulAddDst(dsq,fy,fy,m);MulAddDst(dsq,fz,fz,m);
        ReduceMin(pair,dsq,work,m,true);
        Hop<HardEvent::V_S>();
        const float value=pair.GetValue(0);
        const int32_t index=pair.ReinterpretCast<int32_t>().GetValue(1);
        best.SetValue(point,value);winner.SetValue(point,(float)index);
    }
    Hop<HardEvent::S_V>();
}


template <typename T>
__aicore__ inline bool PreferBasisTask(const GlobalTensor<T>& points,uint64_t base,uint32_t n,uint32_t m,
 LocalTensor<float> ax,LocalTensor<float> ay,LocalTensor<float> az,
 LocalTensor<float> kx,LocalTensor<float> ky,LocalTensor<float> kz,
 LocalTensor<float> vx,LocalTensor<float> vy,LocalTensor<float> wx,LocalTensor<float> wy,LocalTensor<float> wz,
 LocalTensor<float> cv,LocalTensor<float> cw,LocalTensor<float> zero,
 LocalTensor<float> dx,LocalTensor<float> dy)
{
 LocalTensor<T> raw(TPosition::VECCALC,76800,32*32/sizeof(T));
 Hop<HardEvent::V_MTE2>();
 DataCopyPad(raw,points[base*3],{32,3u*(uint32_t)sizeof(T),(n/32-1)*3u*(uint32_t)sizeof(T),0,0},{false,0,0,0});
 Hop<HardEvent::MTE2_S>();
 auto fx=dx,fy=dx[1024],temp=dx[2048];
 auto score=dy,pair=dy[1024],work=dy[2048];
 uint32_t bad=0;
 for(uint32_t q=0;q<32;++q){
  const float x=(float)raw.GetValue(q*(32/sizeof(T)));
  const float y=(float)raw.GetValue(q*(32/sizeof(T))+1);
  const float z=(float)raw.GetValue(q*(32/sizeof(T))+2);
  Muls(fx,vx,-x,m);Axpy(fx,vy,-y,m);Add(fx,fx,cv,m);
  Muls(fy,wx,-x,m);Axpy(fy,wy,-y,m);Axpy(fy,wz,-z,m);Add(fy,fy,cw,m);
  Mul(score,fx,fx,m);MulAddDst(score,fy,fy,m);
  Adds(temp,az,-z,m);Mul(temp,temp,temp,m);Mul(temp,temp,zero,m);Add(score,score,temp,m);
  ReduceMin(pair,score,work,m,true);
  Hop<HardEvent::V_S>();
  const int32_t j=pair.ReinterpretCast<int32_t>().GetValue(1);
  const float value=pair.GetValue(0);
  if(j<0 || (uint32_t)j>=m || !(value>=0.0f)){Hop<HardEvent::S_V>();return false;}
  const float t=(ax.GetValue(j)-x)*kx.GetValue(j)+(ay.GetValue(j)-y)*ky.GetValue(j)+(az.GetValue(j)-z)*kz.GetValue(j);
  const float epsilon=1.0e-12f*(x*x+y*y+z*z)+1.0e-20f;
  if(!(t>1.0e-5f && t<0.99999f) || value<=epsilon)++bad;
  Hop<HardEvent::S_V>();
  if(bad>2)return false;
 }
 return true;
}


template <typename T>
__aicore__ inline bool DensePointBasisTask(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                                    const TreeLineMinDistanceTilingData& t)
{
    constexpr uint32_t tile = 3328;
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T*)p);
    segments.SetGlobalBuffer((__gm__ T*)s);
    distances.SetGlobalBuffer((__gm__ T*)d);
    indices.SetGlobalBuffer((__gm__ int32_t*)i);
    LocalTensor<float> ax(TPosition::VECCALC, 0, 1024);
    LocalTensor<float> ay(TPosition::VECCALC, 4096, 1024);
    LocalTensor<float> az(TPosition::VECCALC, 8192, 1024);
    LocalTensor<float> ux(TPosition::VECCALC, 12288, 1024);
    LocalTensor<float> uy(TPosition::VECCALC, 16384, 1024);
    LocalTensor<float> uz(TPosition::VECCALC, 20480, 1024);
    LocalTensor<float> kx(TPosition::VECCALC, 24576, 1024);
    LocalTensor<float> ky(TPosition::VECCALC, 28672, 1024);
    LocalTensor<float> kz(TPosition::VECCALC, 32768, 1024);
    LocalTensor<float> dx(TPosition::VECCALC, 116992, tile);
    LocalTensor<float> dy(TPosition::VECCALC, 130560, tile);
    LocalTensor<float> dz(TPosition::VECCALC, 36864, tile);
    LocalTensor<float> ratio(TPosition::VECCALC, 50176, tile);
    LocalTensor<float> dsq(TPosition::VECCALC, 63488, tile);
    LocalTensor<float> best(TPosition::VECCALC, 144128, tile);
    LocalTensor<float> winner(TPosition::VECCALC, 157440, tile);
    LocalTensor<float> px(TPosition::VECCALC, 76800, tile);
    LocalTensor<float> py(TPosition::VECCALC, 90112, tile);
    LocalTensor<float> pz(TPosition::VECCALC, 103424, tile);
    LocalTensor<int32_t> pointOffsets(TPosition::VECCALC, 170752, tile);
    LocalTensor<int32_t> segmentOffsets(TPosition::VECCALC, 184320, 1024);
    LocalTensor<half> narrow(TPosition::VECCALC, 50176, tile);
    LocalTensor<uint8_t> keep(TPosition::VECCALC, 187904, 512);
    LocalTensor<T> raw(TPosition::VECCALC, 116992, tile * 3);
    LocalTensor<float> converted(TPosition::VECCALC, 116992, tile * 3);
    LocalTensor<float> indexPacket(TPosition::VECCALC,170752,4096);
    LocalTensor<float> vx(TPosition::VECCALC, 36864, 1024);
    LocalTensor<float> vy(TPosition::VECCALC, 40960, 1024);
    LocalTensor<float> wx(TPosition::VECCALC, 45056, 1024);
    LocalTensor<float> wy(TPosition::VECCALC, 49152, 1024);
    LocalTensor<float> wz(TPosition::VECCALC, 53248, 1024);
    LocalTensor<float> cv(TPosition::VECCALC, 57344, 1024);
    LocalTensor<float> cw(TPosition::VECCALC, 61440, 1024);
    LocalTensor<float> zero(TPosition::VECCALC, 65536, 1024);
    const uint32_t segmentsWide=AlignLanes(t.m);
    uint32_t cached = 0xFFFFFFFFu;
    bool stored = false;
    for (uint32_t task = GetBlockIdx(); task < t.b * t.chunks; task += GetBlockNum()) {
        const uint32_t batch = task / t.chunks;
        const uint32_t begin = (task % t.chunks) * t.chunkPoints;
        if (begin >= t.n) continue;
        const uint32_t owned = t.n - begin < t.chunkPoints ? t.n - begin : t.chunkPoints;
        if (stored) Hop<HardEvent::MTE3_V>();
        if (cached != batch) {
            Hop<HardEvent::V_S>();
            CreateVecIndex(dx,0.0f,segmentsWide);
            Muls(dx,dx,24.0f,segmentsWide);
            Cast(segmentOffsets,dx,RoundMode::CAST_RINT,segmentsWide);
            Hop<HardEvent::V_MTE2>();
            DataCopyPad(raw, segments[(uint64_t)batch * t.m * 6],
                        {1, (uint32_t)(t.m * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            Hop<HardEvent::MTE2_V>();
            LocalTensor<float> src = raw.template ReinterpretCast<float>();
            if constexpr (sizeof(T) == 2) {
                src = converted;
                Cast(src, raw, RoundMode::CAST_NONE, t.m * 6);
            }
            const LocalTensor<uint32_t> off = segmentOffsets.ReinterpretCast<uint32_t>();
            Gather(ax, src, off, 0u, t.m);
            Gather(ay, src, off, 4u, t.m);
            Gather(az, src, off, 8u, t.m);
            Gather(ux, src, off, 12u, t.m);
            Gather(uy, src, off, 16u, t.m);
            Gather(uz, src, off, 20u, t.m);
            Sub(ux, ux, ax, t.m);
            Sub(uy, uy, ay, t.m);
            Sub(uz, uz, az, t.m);
            Duplicate(best, 1.0f, segmentsWide);
            Mul(best, ux, ux, t.m);
            MulAddDst(best, uy, uy, t.m);
            MulAddDst(best, uz, uz, t.m);
            CompareScalar(keep, best, 0.0f, CMPMODE::NE, segmentsWide);
            Select(best, keep, best, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, segmentsWide);
            Duplicate(dsq, -1.0f, segmentsWide);
            Div(winner, dsq, best, segmentsWide);
            Mul(kx, ux, winner, t.m);
            Mul(ky, uy, winner, t.m);
            Mul(kz, uz, winner, t.m);
            cached = batch;
        }

        for (uint32_t done = 0; done < owned; done += tile) {
            if (done != 0) Hop<HardEvent::MTE3_V>();
            const uint32_t count = owned - done < tile ? owned - done : tile;
            const uint32_t wide = AlignLanes(count);
            Hop<HardEvent::V_S>();
            MakeLineBasis(t.m,ax,ay,az,ux,uy,uz,vx,vy,wx,wy,wz,cv,cw,zero);
            Hop<HardEvent::S_V>();
            if(done==0 && !PreferBasisTask(points,(uint64_t)batch*t.n,t.n,t.m,ax,ay,az,kx,ky,kz,vx,vy,wx,wy,wz,cv,cw,zero,dx,dy)) return false;
            CreateVecIndex(dx,0.0f,tile);
            Muls(dx,dx,12.0f,tile);
            Cast(pointOffsets,dx,RoundMode::CAST_RINT,tile);
            Hop<HardEvent::V_MTE2>();
            DataCopyPad(raw, points[((uint64_t)batch * t.n + begin + done) * 3],
                        {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            Hop<HardEvent::MTE2_V>();
            LocalTensor<float> src = raw.template ReinterpretCast<float>();
            if constexpr (sizeof(T) == 2) {
                src = converted;
                Cast(src, raw, RoundMode::CAST_NONE, count * 3);
            }
            if (wide != count) {
                Duplicate(px, 0.0f, wide);
                Duplicate(py, 0.0f, wide);
                Duplicate(pz, 0.0f, wide);
            }
            const LocalTensor<uint32_t> off = pointOffsets.ReinterpretCast<uint32_t>();
            Gather(px, src, off, 0u, count);
            Gather(py, src, off, 4u, count);
            Gather(pz, src, off, 8u, count);
            Muls(px, px, -1.0f, wide);
            Muls(py, py, -1.0f, wide);
            Muls(pz, pz, -1.0f, wide);
            Duplicate(winner, 0.0f, wide);
            Hop<HardEvent::V_S>();
            const uint32_t packetCount=segmentsWide*4;
            CreateVecIndex(indexPacket,0.0f,packetCount);
            Muls(indexPacket,indexPacket,0.25f,packetCount);
            auto packetInt=indexPacket.ReinterpretCast<int32_t>();
            Cast(packetInt,indexPacket,RoundMode::CAST_FLOOR,packetCount);
            Cast(indexPacket,packetInt,RoundMode::CAST_NONE,packetCount);
            Hop<HardEvent::V_S>();
            const uint8_t repeats = wide / 64;
            const UnaryRepeatParams unary(1, 1, 8, 8);
            const BinaryRepeatParams binary(1, 1, 1, 8, 8, 8);
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(64);
            for (int32_t j = 0; j < (int32_t)t.m; ++j) {
                const float v0=vx.GetValue(j),v1=vy.GetValue(j);
                const float w0=wx.GetValue(j),w1=wy.GetValue(j),w2=wz.GetValue(j);
                const float c0=cv.GetValue(j),c1=cw.GetValue(j);
                const bool degenerate=zero.GetValue(j)>0.0f;
                Muls<float,false>(dx,px,v0,MASK_PLACEHOLDER,repeats,unary);
                Axpy<float,float,false>(dx,py,v1,MASK_PLACEHOLDER,repeats,unary);
                Adds<float,false>(dx,dx,c0,MASK_PLACEHOLDER,repeats,unary);
                Muls<float,false>(dy,px,w0,MASK_PLACEHOLDER,repeats,unary);
                Axpy<float,float,false>(dy,py,w1,MASK_PLACEHOLDER,repeats,unary);
                Axpy<float,float,false>(dy,pz,w2,MASK_PLACEHOLDER,repeats,unary);
                Adds<float,false>(dy,dy,c1,MASK_PLACEHOLDER,repeats,unary);
                if(j==0){
                    Mul<float,false>(best,dx,dx,MASK_PLACEHOLDER,repeats,binary);
                    MulAddDst<float,float,false>(best,dy,dy,MASK_PLACEHOLDER,repeats,binary);
                    if(degenerate){
                        Adds<float,false>(dy,pz,az.GetValue(j),MASK_PLACEHOLDER,repeats,unary);
                        MulAddDst<float,float,false>(best,dy,dy,MASK_PLACEHOLDER,repeats,binary);
                    }
                } else {
                    Mul<float,false>(dx,dx,dx,MASK_PLACEHOLDER,repeats,binary);
                    MulAddDst<float,float,false>(dx,dy,dy,MASK_PLACEHOLDER,repeats,binary);
                    if(degenerate){
                        Adds<float,false>(dy,pz,az.GetValue(j),MASK_PLACEHOLDER,repeats,unary);
                        MulAddDst<float,float,false>(dx,dy,dy,MASK_PLACEHOLDER,repeats,binary);
                    }
                    Compare<float,uint8_t,false>(keep,dx,best,CMPMODE::GE,MASK_PLACEHOLDER,repeats,binary);
                    Min<float,false>(best,best,dx,MASK_PLACEHOLDER,repeats,binary);
                    SetCmpMask(indexPacket[(uint32_t)j*4]);
                    PipeBarrier<PIPE_V>();
                    Select<float,uint8_t>(winner,keep,winner,repeats,binary);
                }
            }
            ResetMask();
            VerifyInfiniteNative(count,wide,t.m,ax,ay,az,ux,uy,uz,kx,ky,kz,px,py,pz,best,winner,dx,dy,dz,ratio,dsq,pointOffsets);
            Sqrt(best, best, count);
            LocalTensor<int32_t> outIndex = winner.ReinterpretCast<int32_t>();
            Cast(outIndex, winner, RoundMode::CAST_RINT, count);
            const uint64_t output = (uint64_t)batch * t.n + begin + done;
            if constexpr (sizeof(T) == 2) {
                PipeBarrier<PIPE_V>();
                Cast(narrow, best, RoundMode::CAST_RINT, count);
                Hop<HardEvent::V_MTE3>();
                DataCopyPad(distances[output], narrow, {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
            } else {
                Hop<HardEvent::V_MTE3>();
                DataCopyPad(distances[output], best.template ReinterpretCast<T>(),
                            {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
            }
            DataCopyPad(indices[output], outIndex, {1, count * 4, 0, 0, 0});
            stored = true;
        }
    }
    return true;
}
