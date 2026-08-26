"""Backend-specific fused kernels used by neural-network objectives."""

from .negative_sampling import negative_sampling_loss_gradient

__all__ = ["negative_sampling_loss_gradient"]
