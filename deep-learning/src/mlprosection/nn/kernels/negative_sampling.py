"""Fused score, logistic-loss, and score-gradient kernel for negative sampling."""

from __future__ import annotations

from functools import lru_cache


_CUDA_SOURCE = r"""
extern "C" __global__
void negative_sampling_f32(
    const float* hidden,
    const float* weights,
    const float* labels,
    float* losses,
    float* gradients,
    const long long terms,
    const int candidates,
    const int embedding,
    const float gradient_scale
) {
    const long long term = (
        (long long)blockDim.x * blockIdx.x + threadIdx.x
    );
    if (term >= terms) {
        return;
    }

    const long long example = term / candidates;
    const float* h = hidden + example * embedding;
    const float* w = weights + term * embedding;
    float score = 0.0f;
    for (int column = 0; column < embedding; ++column) {
        score += h[column] * w[column];
    }

    const float label = labels[term];
    losses[term] = fmaxf(score, 0.0f) - score * label
        + log1pf(expf(-fabsf(score)));
    const float probability = 1.0f / (1.0f + expf(-score));
    gradients[term] = (probability - label) * gradient_scale;
}
"""


@lru_cache(maxsize=1)
def _cuda_float_kernel():
    import cupy as cp

    return cp.RawKernel(_CUDA_SOURCE, "negative_sampling_f32")


def negative_sampling_loss_gradient(
    hidden,
    candidate_weights,
    labels,
    *,
    divisor: int,
    backend,
):
    """Return unscaled term losses and divisor-scaled score gradients.

    CUDA float32 uses one raw-kernel launch for candidate dot products,
    numerically stable logistic loss, and its derivative. Other backends use
    the same vectorized formula as a correctness fallback.
    """
    if divisor < 1:
        raise ValueError("negative-sampling divisor must be positive")
    xp = backend.xp
    flat_weights = candidate_weights.reshape(-1, hidden.shape[-1])
    flat_labels = labels.reshape(-1)
    term_count = flat_weights.shape[0]
    candidates_per_example = term_count // len(hidden)

    if backend.is_gpu and hidden.dtype == xp.float32:
        contiguous_hidden = xp.ascontiguousarray(hidden)
        contiguous_weights = xp.ascontiguousarray(flat_weights)
        contiguous_labels = xp.ascontiguousarray(
            flat_labels, dtype=xp.float32
        )
        losses = xp.empty(term_count, dtype=xp.float32)
        gradients = xp.empty(term_count, dtype=xp.float32)
        threads = 256
        blocks = (term_count + threads - 1) // threads
        _cuda_float_kernel()(
            (blocks,),
            (threads,),
            (
                contiguous_hidden,
                contiguous_weights,
                contiguous_labels,
                losses,
                gradients,
                term_count,
                candidates_per_example,
                hidden.shape[-1],
                1.0 / divisor,
            ),
        )
    else:
        hidden_view = hidden.reshape(
            (len(hidden),)
            + (1,) * (candidate_weights.ndim - 2)
            + (hidden.shape[-1],)
        )
        scores = xp.sum(hidden_view * candidate_weights, axis=-1).reshape(-1)
        losses = (
            xp.maximum(scores, 0)
            - scores * flat_labels
            + xp.log1p(xp.exp(-xp.abs(scores)))
        )
        probabilities = 1 / (1 + xp.exp(-scores))
        gradients = (probabilities - flat_labels) / divisor

    output_shape = candidate_weights.shape[:-1]
    return losses.reshape(output_shape), gradients.reshape(output_shape)
