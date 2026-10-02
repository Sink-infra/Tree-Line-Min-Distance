#include "kernel_operator.h"
#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"
using namespace AscendC;
template <typename T>
class KernelTreeLineMinDistance {
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf,pointBuf,arraysBuf,workBuf,resultBuf,outputBuf;
    GlobalTensor<T> points,segments,distances;
    GlobalTensor<int32_t> indices;
    uint32_t b,n,m,width;
    __aicore__ inline void Sync() { PipeBarrier<PIPE_ALL>(); }
public:
    __aicore__ inline void Init(GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,TreeLineMinDistanceTilingData t) {
        b=t.b; n=t.n; m=t.m; width=(m+63)/64*64;
        points.SetGlobalBuffer((__gm__ T*)p); segments.SetGlobalBuffer((__gm__ T*)s);
        distances.SetGlobalBuffer((__gm__ T*)d); indices.SetGlobalBuffer((__gm__ int32_t*)i);
        pipe.InitBuffer(rawBuf,6*1024*4);
        pipe.InitBuffer(pointBuf,96*4);
        pipe.InitBuffer(arraysBuf,7*width*4);
        pipe.InitBuffer(workBuf,7*width*4);
        pipe.InitBuffer(resultBuf,(width+64)*4);
        pipe.InitBuffer(outputBuf,32*12);
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
        auto out=outputBuf.Get<float>(),idx=outputBuf.Get<int32_t>()[32];
        auto typedOut=outputBuf.Get<T>()[256/sizeof(T)];
        uint32_t blocks=(n+31)/32,cached=b;
        for (uint32_t task=GetBlockIdx();task<b*blocks;task+=GetBlockNum()) {
            uint32_t batch=task/blocks,start=(task%blocks)*32,count=n-start;
            if(count>32) count=32;
            if(cached!=batch) {
                DataCopyPad(raw,segments[batch*m*6],{1,m*6*sizeof(T),0,0,0},{false,0,0,0}); Sync();
                for(uint32_t j=0;j<m;++j) {
                    float x=(float)raw.GetValue(6*j),y=(float)raw.GetValue(6*j+1),z=(float)raw.GetValue(6*j+2);
                    float vx=(float)raw.GetValue(6*j+3)-x,vy=(float)raw.GetValue(6*j+4)-y,vz=(float)raw.GetValue(6*j+5)-z;
                    ax.SetValue(j,x); ay.SetValue(j,y); az.SetValue(j,z);
                    ux.SetValue(j,vx); uy.SetValue(j,vy); uz.SetValue(j,vz);
                    float l=vx*vx+vy*vy+vz*vz;
                    len.SetValue(j,l==0.0f?1.0f:l);
                }
                cached=batch; Sync();
            }
            DataCopyPad(pt,points[(batch*n+start)*3],{1,count*3*sizeof(T),0,0,0},{false,0,0,0}); Sync();
            for(uint32_t p=0;p<count;++p) {
                float x=(float)pt.GetValue(3*p),y=(float)pt.GetValue(3*p+1),z=(float)pt.GetValue(3*p+2);
                Adds(dx,ax,-x,m); Adds(dy,ay,-y,m); Adds(dz,az,-z,m);
                Mul(t,dx,ux,m); Mul(tmp,dy,uy,m); Add(t,t,tmp,m);
                Mul(tmp,dz,uz,m); Add(t,t,tmp,m); Muls(t,t,-1.0f,m);
                Div(t,t,len,m); Maxs(t,t,0.0f,m); Mins(t,t,1.0f,m);
                Mul(tmp,t,ux,m); Add(dx,dx,tmp,m);
                Mul(tmp,t,uy,m); Add(dy,dy,tmp,m);
                Mul(tmp,t,uz,m); Add(dz,dz,tmp,m);
                Mul(dsq,dx,dx,m); Mul(tmp,dy,dy,m); Add(dsq,dsq,tmp,m);
                Mul(tmp,dz,dz,m); Add(dsq,dsq,tmp,m);
                ReduceMin(red,dsq,scratch,m,true); Sync();
                int32_t winner=red.ReinterpretCast<int32_t>().GetValue(1);
                Sqrt(red,red,1); Sync();
                out.SetValue(p,red.GetValue(0)); idx.SetValue(p,winner);
            }
            Sync();
            if constexpr (sizeof(T)==2) { Cast(typedOut,out,RoundMode::CAST_RINT,count); Sync(); }
            else { typedOut=out; }
            DataCopyPad(distances[batch*n+start],typedOut,{1,count*sizeof(T),0,0,0});
            DataCopyPad(indices[batch*n+start],idx,{1,count*4,0,0,0}); Sync();
        }
    }
};
template <typename DT_TREE_POINTS>
__global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points,GM_ADDR line_segments,GM_ADDR min_distance,GM_ADDR nearest_segment_idx,GM_ADDR workspace,GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData,t,tiling);
    KernelTreeLineMinDistance<DT_TREE_POINTS> op;
    op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,t); op.Process();
}
