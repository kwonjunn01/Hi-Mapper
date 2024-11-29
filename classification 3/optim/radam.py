import torch.optim

from optim.mixin import OptimMixin
from geoopt import ManifoldParameter, ManifoldTensor
# from geoopt.utils import copy_or_set_
from torch.nn.utils.clip_grad import clip_grad_norm_


__all__ = ["RiemannianAdam"]


class RiemannianAdam(OptimMixin, torch.optim.Adam):
    r"""
    Riemannian Adam with the same API as :class:`torch.optim.Adam`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float (optional)
        learning rate (default: 1e-3)
    betas : Tuple[float, float] (optional)
        coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps : float (optional)
        term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    amsgrad : bool (optional)
        whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)


    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """

    def __init__(self, *args, stabilize, **kwargs):
        super().__init__(*args, stabilize=stabilize, **kwargs)
        # self.max_grad_norm = max_grad_norm

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                group["step"] += 1
                # group['max_grad_norm'] = self.max_grad_norm
                for point in group["params"]:
                    # if group['max_grad_norm'] > 0:
                    #     clip_grad_norm_(point, group['max_grad_norm'])
                    grad = point.grad
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        manifold = point.manifold
                    else:
                        manifold = self._default_manifold

                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianAdam does not support sparse gradients, use SparseRiemannianAdam instead"
                        )

                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(point)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    # make local variables for easy access
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    # actual step
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        grad.add_(point, alpha=weight_decay)
                    grad = manifold.egrad2rgrad(point, grad)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(
                        manifold.component_inner(point, grad), alpha=1 - betas[1]
                    )
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq
                    else:
                        denom = exp_avg_sq
                    bias_correction1 = 1 - betas[0] ** group["step"]
                    bias_correction2 = 1 - betas[1] ** group["step"]
                    step_size = learning_rate

                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = (exp_avg / bias_correction1) / ((denom / bias_correction2).sqrt() + eps)
                    if not isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        direction = direction + point * weight_decay
                    # transport the exponential averaging to the new point
                    new_point, exp_avg_new = manifold.retr_transp(
                        point, -step_size * direction, exp_avg
                    )
                    # use copy only for user facing point
                    # copy_or_set_(point, new_point)
                    # exp_avg.set_(exp_avg_new)
                    point.copy_(new_point)
                    exp_avg.copy_(exp_avg_new)

                if (
                    group["stabilize"] is not None
                    and group["step"] % group["stabilize"] == 0
                ):
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            state = self.state[p]
            if not state:  # due to None grads
                continue
            manifold = p.manifold
            exp_avg = state["exp_avg"]
            # copy_or_set_(p, manifold.projx(p))
            # exp_avg.set_(manifold.proju(p, exp_avg))
            p.copy_(manifold.projx(p))
            exp_avg.copy_(manifold.proju(p, exp_avg))