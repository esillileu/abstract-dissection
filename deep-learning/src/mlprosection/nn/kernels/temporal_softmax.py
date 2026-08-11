"""Fused CUDA float32 temporal softmax cross-entropy kernel."""

from __future__ import annotations

from functools import lru_cache


_SOURCE = r"""
extern "C" __global__ void temporal_softmax_xent_f32(
 const float* scores, const long long* labels, float* gradient,
 float* row_losses, int rows, int classes, int ignore_label,
 int use_ignore, float scale) {
  int row=blockIdx.x, lane=threadIdx.x;
  if(row>=rows) return;
  extern __shared__ float shared[];
  const float* input=scores+(long long)row*classes;
  float* output=gradient+(long long)row*classes;
  long long label=labels[row];
  bool ignored=use_ignore && label==ignore_label;
  float local_max=-3.402823466e+38F;
  for(int col=lane; col<classes; col+=blockDim.x)
    local_max=fmaxf(local_max,input[col]);
  shared[lane]=local_max;
  __syncthreads();
  for(int stride=blockDim.x/2; stride>0; stride>>=1) {
    if(lane<stride) shared[lane]=fmaxf(shared[lane],shared[lane+stride]);
    __syncthreads();
  }
  float maximum=shared[0], local_sum=0.0f;
  __syncthreads();
  for(int col=lane; col<classes; col+=blockDim.x) {
    float value=expf(input[col]-maximum);
    output[col]=value;
    local_sum+=value;
  }
  shared[lane]=local_sum;
  __syncthreads();
  for(int stride=blockDim.x/2; stride>0; stride>>=1) {
    if(lane<stride) shared[lane]+=shared[lane+stride];
    __syncthreads();
  }
  float total=shared[0];
  for(int col=lane; col<classes; col+=blockDim.x) {
    float probability=output[col]/total;
    output[col]=ignored ? 0.0f : probability*scale;
  }
  __syncthreads();
  if(lane==0) {
    float probability=ignored ? 1.0f : expf(input[label]-maximum)/total;
    row_losses[row]=ignored ? 0.0f : -logf(probability+1.0e-7f)*scale;
  }
  if(!ignored && lane==(int)(label%blockDim.x)) output[label]-=scale;
}
"""


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp

    return cp.RawKernel(
        _SOURCE, "temporal_softmax_xent_f32", options=("--std=c++11",)
    )


def temporal_softmax_loss_gradient(
    prediction, labels, *, reduction: str, ignore_label: int | None, backend
):
    """Return fused loss, gradient, and unit count for contiguous class labels."""
    xp = backend.xp
    rows = prediction.size // prediction.shape[-1]
    classes = prediction.shape[-1]
    labels = labels.reshape(-1).astype(xp.int64, copy=False)
    if labels.size != rows:
        raise ValueError("target shape does not match the logits prediction units")
    if ignore_label is None:
        unit_count, use_ignore, kernel_ignore = rows, 0, 0
    else:
        unit_count = backend.scalar_to_int((labels != ignore_label).sum())
        use_ignore, kernel_ignore = 1, ignore_label
    if unit_count == 0:
        raise ValueError("softmax_cross_entropy has no non-ignored targets")
    scale = 1.0 / unit_count if reduction == "mean" else 1.0
    gradient = xp.empty_like(prediction)
    row_losses = xp.empty(rows, dtype=xp.float32)
    _kernel()(
        (rows,), (256,),
        (prediction, labels, gradient, row_losses, xp.int32(rows),
         xp.int32(classes), xp.int32(kernel_ignore), xp.int32(use_ignore),
         xp.float32(scale)),
        shared_mem=256 * 4,
    )
    return row_losses.sum(dtype=xp.float32), gradient, unit_count
