import torch
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler


class SimpleDreamerOptimizer:
    """Minimal PyTorch Dreamer optimizer: AdamW + AMP + fixed gradient clipping."""

    def __init__(
        self,
        parameters,
        lr=4e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        grad_clip=100.0,
        use_amp=True,
    ):
        self.parameters = list(parameters)
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.grad_clip = grad_clip
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

    def __call__(self, loss: torch.Tensor):
        """Perform backward and optimization step on a scalar loss."""

        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Backward with AMP
        with autocast(enabled=self.use_amp):
            self.scaler.scale(loss).backward()

        # Unscale gradients for clipping
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        clip_grad_norm_(self.parameters, self.grad_clip)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
