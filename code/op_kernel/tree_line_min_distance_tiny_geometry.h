#pragma once
__aicore__ inline float TinyAxisGap(float p, float a, float b)
{
    const float lo = a < b ? a : b;
    const float hi = a < b ? b : a;
    return p < lo ? p - lo : (p > hi ? p - hi : 0.0f);
}
__aicore__ inline float TinyGeometryDsq(float px,float py,float pz,
    float ax,float ay,float az,float bx,float by,float bz)
{
    float dx=px-ax,dy=py-ay,dz=pz-az;
    if (ay==by && az==bz) {
        dx=TinyAxisGap(px,ax,bx);
    } else if (ax==bx && az==bz) {
        dy=TinyAxisGap(py,ay,by);
    } else if (ax==bx && ay==by) {
        dz=TinyAxisGap(pz,az,bz);
    } else {
        const float ux=bx-ax,uy=by-ay,uz=bz-az;
        const float dot=dx*ux+dy*uy+dz*uz;
        const float len=ux*ux+uy*uy+uz*uz;
        if (len>0.0f && dot>0.0f) {
            const float ratio=dot<len?dot/len:1.0f;
            dx-=ratio*ux;dy-=ratio*uy;dz-=ratio*uz;
        }
    }
    return dx*dx+dy*dy+dz*dz;
}
