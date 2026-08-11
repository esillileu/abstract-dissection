"""Fused CUDA float32 elementwise kernels for ``TimeLSTM``."""

from __future__ import annotations

from functools import lru_cache


_FORWARD = r"""
extern "C" __global__ void time_lstm_forward_f32(
 const float* xproj, const float* hproj, const float* bias,
 const float* hp, const float* cp, float* hn, float* cn, float* hs,
 float* hpseq, float* cpseq, float* gates, float* cells,
 int n, int tsize, int hsize, int t, int cache) {
  int j=blockDim.x*blockIdx.x+threadIdx.x, size=n*hsize;
  if(j>=size) return;
  int r=j/hsize, c=j-r*hsize;
  int q=(r*tsize+t)*4*hsize, rq=r*4*hsize;
  int s=(r*tsize+t)*hsize+c;
  float af=xproj[q+c]+hproj[rq+c]+bias[c];
  float ag=xproj[q+hsize+c]+hproj[rq+hsize+c]+bias[hsize+c];
  float ai=xproj[q+2*hsize+c]+hproj[rq+2*hsize+c]+bias[2*hsize+c];
  float ao=xproj[q+3*hsize+c]+hproj[rq+3*hsize+c]+bias[3*hsize+c];
  float f=1.0f/(1.0f+expf(-af)), g=tanhf(ag);
  float i=1.0f/(1.0f+expf(-ai)), o=1.0f/(1.0f+expf(-ao));
  float nc=f*cp[j]+g*i, nh=o*tanhf(nc);
  hn[j]=nh; cn[j]=nc; hs[s]=nh;
  if(cache) {
    hpseq[s]=hp[j]; cpseq[s]=cp[j]; cells[s]=nc;
    gates[q+c]=f; gates[q+hsize+c]=g;
    gates[q+2*hsize+c]=i; gates[q+3*hsize+c]=o;
  }
}
"""

_BACKWARD = r"""
extern "C" __global__ void time_lstm_backward_f32(
 const float* dhs, const float* dh, const float* dc,
 const float* cpseq, const float* gates, const float* cells,
 float* daseq, float* dcp, int n, int tsize, int hsize, int t) {
  int j=blockDim.x*blockIdx.x+threadIdx.x, size=n*hsize;
  if(j>=size) return;
  int r=j/hsize, c=j-r*hsize;
  int s=(r*tsize+t)*hsize+c, q=(r*tsize+t)*4*hsize;
  float f=gates[q+c], g=gates[q+hsize+c];
  float i=gates[q+2*hsize+c], o=gates[q+3*hsize+c];
  float tc=tanhf(cells[s]), up=dhs[s]+dh[j];
  float ds=dc[j]+up*o*(1.0f-tc*tc);
  dcp[j]=ds*f;
  daseq[q+c]=ds*cpseq[s]*f*(1.0f-f);
  daseq[q+hsize+c]=ds*i*(1.0f-g*g);
  daseq[q+2*hsize+c]=ds*g*i*(1.0f-i);
  daseq[q+3*hsize+c]=up*tc*o*(1.0f-o);
}
"""


@lru_cache(maxsize=1)
def _forward_kernel():
    import cupy as cp

    return cp.RawKernel(_FORWARD, "time_lstm_forward_f32", options=("--std=c++11",))


@lru_cache(maxsize=1)
def _backward_kernel():
    import cupy as cp

    return cp.RawKernel(_BACKWARD, "time_lstm_backward_f32", options=("--std=c++11",))


def launch_forward(
    xproj, hproj, bias, hp, c_prev, hn, cn, hs, hpseq, cpseq, gates, cells,
    *, timestep: int, cache: bool,
) -> None:
    import cupy as cp

    n, time_size, four_hidden = xproj.shape
    hidden_size = four_hidden // 4
    size = n * hidden_size
    _forward_kernel()(
        ((size + 255) // 256,), (256,),
        (xproj, hproj, bias, hp, c_prev, hn, cn, hs, hpseq, cpseq, gates, cells,
         cp.int32(n), cp.int32(time_size), cp.int32(hidden_size),
         cp.int32(timestep), cp.int32(cache)),
    )


def launch_backward(
    dhs, dh, dc, cpseq, gates, cells, daseq, dcp, *, timestep: int,
) -> None:
    import cupy as cp

    n, time_size, hidden_size = dhs.shape
    size = n * hidden_size
    _backward_kernel()(
        ((size + 255) // 256,), (256,),
        (dhs, dh, dc, cpseq, gates, cells, daseq, dcp,
         cp.int32(n), cp.int32(time_size), cp.int32(hidden_size),
         cp.int32(timestep)),
    )
