"""Component fixtures preparing inputs and caches for measured operations."""

from __future__ import annotations

from ..workloads import _batch


class ComponentFixture:
    """Prepare real e02 tensors and caches for one measured component."""

    def __init__(self, workload, *, batch_size: int) -> None:
        self.workload = workload
        self.batch_x, self.batch_t = _batch(workload, 0, batch_size)
        self.model_x = None
        self.objective_t = None
        self.objective_batch = None
        self.prediction = None
        self.result = None
        self.gradient = None

    def batch_adapter(self) -> None:
        self.model_x, self.objective_t = self.workload.adapter.prepare(
            self.batch_x,
            self.batch_t,
        )

    def prepare_objective_prepare(self) -> None:
        self.batch_adapter()

    def objective_prepare(self) -> None:
        self.objective_batch = self.workload.objective.prepare(self.objective_t)

    def prepare_model_forward(self) -> None:
        self.batch_adapter()
        self.objective_prepare()

    def model_forward(self) -> None:
        self.prediction = self.workload.model.forward(
            self.model_x,
            candidates=self.objective_batch.candidates,
        )

    def prepare_objective_forward(self) -> None:
        self.prepare_model_forward()
        self.model_forward()

    def objective_forward(self) -> None:
        self.result = self.workload.objective.forward(
            self.prediction,
            self.objective_batch.target,
            replay_context=self.objective_batch.replay_context,
            example_count=len(self.batch_x),
        )

    def prepare_objective_backward(self) -> None:
        self.prepare_objective_forward()
        self.objective_forward()

    def objective_backward(self) -> None:
        self.gradient = self.workload.objective.backward()

    def prepare_model_backward(self) -> None:
        self.prepare_objective_backward()
        self.objective_backward()

    def model_backward(self) -> None:
        self.workload.model.backward(self.gradient)

    def prepare_optimizer(self) -> None:
        self.prepare_model_backward()
        self.model_backward()

    def optimizer(self) -> None:
        self.workload.optimizer.update()

    def operation(self, name: str):
        return getattr(self, name)

    def preparation(self, name: str):
        prepare = getattr(self, f"prepare_{name}", None)
        return prepare


class OriginalComponentFixture:
    """Expose the indivisible module boundaries of the book implementation."""

    def __init__(self, workload, *, batch_size: int) -> None:
        self.workload = workload
        self.batch_x, self.batch_t = _batch(workload, 0, batch_size)
        self.params = None
        self.grads = None

    def forward(self) -> None:
        self.workload.model.forward(self.batch_x, self.batch_t)

    def prepare_backward(self) -> None:
        self.forward()

    def backward(self) -> None:
        self.workload.model.backward()

    def prepare_deduplicate_shared_parameters(self) -> None:
        self.prepare_backward()
        self.backward()

    def deduplicate_shared_parameters(self) -> None:
        self.params, self.grads = self.workload.remove_duplicate(
            self.workload.model.params,
            self.workload.model.grads,
        )

    def prepare_optimizer(self) -> None:
        self.prepare_deduplicate_shared_parameters()
        self.deduplicate_shared_parameters()

    def optimizer(self) -> None:
        self.workload.optimizer.update(self.params, self.grads)

    def operation(self, name: str):
        return getattr(self, name)

    def preparation(self, name: str):
        return getattr(self, f"prepare_{name}", None)


class FusedComponentFixture:
    """Expose the combined boundaries of the fused CBOW objective."""

    def __init__(self, workload, *, batch_size: int) -> None:
        self.workload = workload
        self.batch_x, self.batch_t = _batch(workload, 0, batch_size)
        self.model_x = None
        self.objective_t = None
        self.objective_batch = None

    def batch_adapter(self) -> None:
        self.model_x, self.objective_t = self.workload.adapter.prepare(
            self.batch_x,
            self.batch_t,
        )

    def prepare_objective_prepare(self) -> None:
        self.batch_adapter()

    def objective_prepare(self) -> None:
        self.objective_batch = self.workload.objective.prepare(self.objective_t)

    def prepare_fused_forward_loss(self) -> None:
        self.prepare_objective_prepare()
        self.objective_prepare()

    def fused_forward_loss(self) -> None:
        self.workload.objective.forward_fused(
            self.workload.model,
            self.model_x,
            self.objective_batch,
            example_count=len(self.batch_x),
        )

    def prepare_fused_backward(self) -> None:
        self.prepare_fused_forward_loss()
        self.fused_forward_loss()

    def fused_backward(self) -> None:
        self.workload.objective.backward_fused(self.workload.model)

    def prepare_optimizer(self) -> None:
        self.prepare_fused_backward()
        self.fused_backward()

    def optimizer(self) -> None:
        self.workload.optimizer.update()

    def operation(self, name: str):
        return getattr(self, name)

    def preparation(self, name: str):
        return getattr(self, f"prepare_{name}", None)
