#include "kernel_operator.h"
#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"
using namespace AscendC;
template <typename T>
class KernelTreeLineMinDistance {
    TPipe *pipe;
    TBuf<TPosition::VECCALC> rawBuf,pointBuf,arraysBuf,workBuf,resultBuf,outputBuf,convertBuf;
    GlobalTensor<T> points,segments,distances;
    GlobalTensor<int32_t> indices;
    uint32_t b,n,m,width;
    __aicore__ inline void Sync() { PipeBarrier<PIPE_ALL>(); }
public:
    __aicore__ inline void Init(GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,TreeLineMinDistanceTilingData t,TPipe *pipeIn) {
        pipe=pipeIn;
        b=t.b; n=t.n; m=t.m; width=(m+63)/64*64;
        points.SetGlobalBuffer((__gm__ T*)p); segments.SetGlobalBuffer((__gm__ T*)s);
        distances.SetGlobalBuffer((__gm__ T*)d); indices.SetGlobalBuffer((__gm__ int32_t*)i);
        pipe->InitBuffer(rawBuf,6*1024*4);
        pipe->InitBuffer(pointBuf,96*4);
        pipe->InitBuffer(arraysBuf,7*width*4);
        pipe->InitBuffer(workBuf,7*width*4);
        pipe->InitBuffer(resultBuf,(width+64)*4);
        pipe->InitBuffer(outputBuf,32*12);
        if constexpr (sizeof(T)==2) { pipe->InitBuffer(convertBuf,6*1024*4); }
    }
    __aicore__ inline void Process() {
        auto raw=rawBuf.Get<T>();
        auto pt=pointBuf.Get<T>();
        auto a=arraysBuf.Get<float>();
        auto ax=a[0],ay=a[width],az=a[2*width];
        auto ux=a[3*width],uy=a[4*width],uz=a[5*width],len=a[6*width];
        auto work=workBuf.Get<float>();
        auto dx=work[0],dy=work[width],dz=work[2*width],t=work[3*width];
        auto tmp=work[4*width],dsq=work[5*width];
        auto red=resultBuf.Get<float>(),scratch=red[64];
        auto out=outputBuf.Get<float>();
        auto idx=outputBuf.Get<int32_t>()[32];
        auto typedOut=outputBuf.Get<T>()[256/sizeof(T)];
        uint32_t blocks=(n+31)/32,cached=b;
        for (uint32_t task=GetBlockIdx();task<b*blocks;task+=GetBlockNum()) {
            uint32_t batch=task/blocks,start=(task%blocks)*32,count=n-start;
            if(count>32) count=32;
            if(cached!=batch) {
                DataCopyPad(raw,segments[batch*m*6],{1,uint32_t(m*6*sizeof(T)),0,0,0},{false,0,0,0}); Sync();
                if(m>=32) {
                    auto src=raw.template ReinterpretCast<float>();
                    if constexpr (sizeof(T)==2) {
                        src=convertBuf.Get<float>();
                        Cast(src,raw,RoundMode::CAST_NONE,6*m);
                    }
                    auto offsets=work.template ReinterpretCast<uint32_t>()[6*width];
                    for(uint32_t j=0;j<m;++j) { offsets.SetValue(j,24*j); }
                    Sync();
                    Gather(ax,src,offsets,0,m); Gather(ay,src,offsets,4,m); Gather(az,src,offsets,8,m);
                    Gather(ux,src,offsets,12,m); Gather(uy,src,offsets,16,m); Gather(uz,src,offsets,20,m);
                    Sub(ux,ux,ax,m); Sub(uy,uy,ay,m); Sub(uz,uz,az,m);
                    Mul(len,ux,ux,m); Mul(tmp,uy,uy,m); Add(len,len,tmp,m);
                    Mul(tmp,uz,uz,m); Add(len,len,tmp,m);
                    auto mask=pointBuf.Get<uint8_t>();
                    CompareScalar(mask,len,0.0f,CMPMODE::NE,width);
                    Select(len,mask,len,1.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,m);
                } else {
                for(uint32_t j=0;j<m;++j) {
                    float x=(float)raw.GetValue(6*j),y=(float)raw.GetValue(6*j+1),z=(float)raw.GetValue(6*j+2);
                    float vx=(float)raw.GetValue(6*j+3)-x,vy=(float)raw.GetValue(6*j+4)-y,vz=(float)raw.GetValue(6*j+5)-z;
                    ax.SetValue(j,x); ay.SetValue(j,y); az.SetValue(j,z);
                    ux.SetValue(j,vx); uy.SetValue(j,vy); uz.SetValue(j,vz);
                    float l=vx*vx+vy*vy+vz*vz;
                    len.SetValue(j,l==0.0f?1.0f:l);
                }
                }
                cached=batch; Sync();
            }
            DataCopyPad(pt,points[(batch*n+start)*3],{1,uint32_t(count*3*sizeof(T)),0,0,0},{false,0,0,0}); Sync();
            if(m==1 && count>=4) {
                float x=ax.GetValue(0),y=ay.GetValue(0),z=az.GetValue(0);
                float vx=ux.GetValue(0),vy=uy.GetValue(0),vz=uz.GetValue(0);
                auto denominator=work[6*width];
                for(uint32_t p=0;p<count;++p) {
                    dx.SetValue(p,x-(float)pt.GetValue(3*p));
                    dy.SetValue(p,y-(float)pt.GetValue(3*p+1));
                    dz.SetValue(p,z-(float)pt.GetValue(3*p+2));
                }
                Sync();
                Duplicate(denominator,len.GetValue(0),count);
                Muls(t,dx,vx,count); Muls(tmp,dy,vy,count); Add(t,t,tmp,count);
                Muls(tmp,dz,vz,count); Add(t,t,tmp,count); Muls(t,t,-1.0f,count);
                Div(t,t,denominator,count); Maxs(t,t,0.0f,count); Mins(t,t,1.0f,count);
                Muls(tmp,t,vx,count); Add(dx,dx,tmp,count);
                Muls(tmp,t,vy,count); Add(dy,dy,tmp,count);
                Muls(tmp,t,vz,count); Add(dz,dz,tmp,count);
                Mul(dsq,dx,dx,count); Mul(tmp,dy,dy,count); Add(dsq,dsq,tmp,count);
                Mul(tmp,dz,dz,count); Add(dsq,dsq,tmp,count);
                Sqrt(out,dsq,count);
                Duplicate(idx,int32_t(0),count);
            } else if(m>1 && 2*m<=count) {
                auto px=rawBuf.Get<float>()[256],py=rawBuf.Get<float>()[320],pz=rawBuf.Get<float>()[384];
                auto floatIdx=work[6*width],denominator=scratch;
                auto mask=pointBuf.Get<uint8_t>();
                for(uint32_t p=0;p<count;++p) {
                    px.SetValue(p,(float)pt.GetValue(3*p));
                    py.SetValue(p,(float)pt.GetValue(3*p+1));
                    pz.SetValue(p,(float)pt.GetValue(3*p+2));
                }
                Sync();
                Duplicate(out,3.402823466e38f,count); Duplicate(floatIdx,0.0f,count);
                for(uint32_t j=0;j<m;++j) {
                    float x=ax.GetValue(j),y=ay.GetValue(j),z=az.GetValue(j);
                    float vx=ux.GetValue(j),vy=uy.GetValue(j),vz=uz.GetValue(j);
                    Muls(dx,px,-1.0f,count); Adds(dx,dx,x,count);
                    Muls(dy,py,-1.0f,count); Adds(dy,dy,y,count);
                    Muls(dz,pz,-1.0f,count); Adds(dz,dz,z,count);
                    Duplicate(denominator,len.GetValue(j),count);
                    Muls(t,dx,vx,count); Muls(tmp,dy,vy,count); Add(t,t,tmp,count);
                    Muls(tmp,dz,vz,count); Add(t,t,tmp,count); Muls(t,t,-1.0f,count);
                    Div(t,t,denominator,count); Maxs(t,t,0.0f,count); Mins(t,t,1.0f,count);
                    Muls(tmp,t,vx,count); Add(dx,dx,tmp,count);
                    Muls(tmp,t,vy,count); Add(dy,dy,tmp,count);
                    Muls(tmp,t,vz,count); Add(dz,dz,tmp,count);
                    Mul(dsq,dx,dx,count); Mul(tmp,dy,dy,count); Add(dsq,dsq,tmp,count);
                    Mul(tmp,dz,dz,count); Add(dsq,dsq,tmp,count);
                    Compare(mask,dsq,out,CMPMODE::GE,64);
                    Select(out,mask,out,dsq,SELMODE::VSEL_CMPMASK_SPR,count);
                    Duplicate(denominator,float(int32_t(j)),count);
                    Select(floatIdx,mask,floatIdx,denominator,SELMODE::VSEL_CMPMASK_SPR,count);
                }
                Sqrt(out,out,count); Cast(idx,floatIdx,RoundMode::CAST_RINT,count);
            } else {
            for(uint32_t p=0;p<count;++p) {
                float x=(float)pt.GetValue(3*p),y=(float)pt.GetValue(3*p+1),z=(float)pt.GetValue(3*p+2);
                Adds(dx,ax,-x,m); Adds(dy,ay,-y,m); Adds(dz,az,-z,m);
                if(m>=32 && width<=768) {
                    Mul(t,dx,ux,m); MulAddDst(t,dy,uy,m); MulAddDst(t,dz,uz,m); Muls(t,t,-1.0f,m);
                    Div(t,t,len,m); Maxs(t,t,0.0f,m); Mins(t,t,1.0f,m);
                    MulAddDst(dx,t,ux,m); MulAddDst(dy,t,uy,m); MulAddDst(dz,t,uz,m);
                    Mul(dsq,dx,dx,m); MulAddDst(dsq,dy,dy,m); MulAddDst(dsq,dz,dz,m);
                } else {
                Mul(t,dx,ux,m); Mul(tmp,dy,uy,m); Add(t,t,tmp,m);
                Mul(tmp,dz,uz,m); Add(t,t,tmp,m); Muls(t,t,-1.0f,m);
                Div(t,t,len,m); Maxs(t,t,0.0f,m); Mins(t,t,1.0f,m);
                Mul(tmp,t,ux,m); Add(dx,dx,tmp,m);
                Mul(tmp,t,uy,m); Add(dy,dy,tmp,m);
                Mul(tmp,t,uz,m); Add(dz,dz,tmp,m);
                Mul(dsq,dx,dx,m); Mul(tmp,dy,dy,m); Add(dsq,dsq,tmp,m);
                Mul(tmp,dz,dz,m); Add(dsq,dsq,tmp,m);
                }
                ReduceMin(red,dsq,scratch,m,true); Sync();
                int32_t winner=red.ReinterpretCast<int32_t>().GetValue(1);
                Sqrt(red,red,1); Sync();
                out.SetValue(p,red.GetValue(0)); idx.SetValue(p,winner);
            }
            }
            Sync();
            if constexpr (sizeof(T)==2) { Cast(typedOut,out,RoundMode::CAST_RINT,count); Sync(); }
            else { typedOut=out; }
            DataCopyPad(distances[batch*n+start],typedOut,{1,uint32_t(count*sizeof(T)),0,0,0});
            DataCopyPad(indices[batch*n+start],idx,{1,count*4,0,0,0}); Sync();
        }
    }
};
template <typename DT_TREE_POINTS>
__global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points,GM_ADDR line_segments,GM_ADDR min_distance,GM_ADDR nearest_segment_idx,GM_ADDR workspace,GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData,t,tiling);
    TPipe pipe;
    KernelTreeLineMinDistance<DT_TREE_POINTS> op;
    op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,t,&pipe); op.Process();
}
