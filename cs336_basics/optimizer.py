import math
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 1
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.grad.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.grad.data)

                # weight decay
                p.data -= group["lr"] * group["weight_decay"] * p.data

                #
                adjusted_lr = (
                    group["lr"]
                    * math.sqrt(1 - group["beta2"] ** state["step"])
                    / (1 - group["beta1"] ** state["step"])
                )
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["exp_avg"] = (
                    group["beta1"] * exp_avg + (1 - group["beta1"]) * grad
                )
                state["exp_avg_sq"] = group["beta2"] * exp_avg_sq + (
                    1 - group["beta2"]
                ) * grad.pow(2)

                p.data -= (
                    adjusted_lr
                    * state["exp_avg"]
                    / (state["exp_avg_sq"].sqrt() + group["eps"])
                )
                state["step"] += 1
