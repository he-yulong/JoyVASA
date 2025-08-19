# joyvasa_pl/lightning_module.py
from typing import Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .denoiser import TransformerDenoiser
from .losses import simple_loss, velocity_loss, smooth_loss, expression_loss


class DiffusionMotionLightning(pl.LightningModule):
    def __init__(self, Dm=256, Da=768, d_model=512, nhead=8, nlayers=6,
                 Wpre=25, Wcur=100, lr=1e-4, betas=(1e-4, 0.02),
                 lambda_vel=5.0, lambda_smooth=0.5, lambda_exp=0.1, max_t=1000):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerDenoiser(Dm, Da, d_model, nhead, nlayers, Wpre, Wcur)

        # diffusion schedule (linear betas)
        self.register_buffer("betas", torch.linspace(betas[0], betas[1], max_t))
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

        # which dims are δ inside motion vector? (Dm layout: [R|t|δ|s]) -> adapt to your code
        self.delta_slice = slice(Dm // 2, Dm // 2 + Dm // 4)  # <- TODO: set correctly

    def q_sample(self, X0_cur, t, noise=None):
        """Forward diffusion: x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*ε, applied per-batch timestep."""
        if noise is None:
            noise = torch.randn_like(X0_cur)
        a_bar = self.alphas_cumprod[t].view(-1, 1, 1)  # (B,1,1)
        return (a_bar.sqrt() * X0_cur) + ((1 - a_bar).sqrt() * noise), noise

    def training_step(self, batch: Dict, batch_idx: int):
        X0 = batch["X0"].to(self.device)  # (B, Wpre+Wcur, Dm)
        A = batch["A"].to(self.device)  # (B, Wpre+Wcur, Da)
        t = batch["t"].to(self.device).view(-1)

        # split past/current
        Wpre, Wcur = self.hparams.Wpre, self.hparams.Wcur
        X_past = X0[:, :Wpre, :]
        X0_cur = X0[:, -Wcur:, :]
        A_full = A

        # forward diffusion on current window
        Xt_cur, noise = self.q_sample(X0_cur, t)

        # predict clean motion for current window
        Xhat0_cur = self.model(Xt_cur, A_full, X_past, t)

        # === losses (Eqs. 4–8) ===
        Ls = simple_loss(X0_cur, Xhat0_cur)
        Lv = velocity_loss(X0_cur, Xhat0_cur)
        Lsm = smooth_loss(Xhat0_cur)
        # expression (δ) slice loss
        d_gt = X0_cur[:, :, self.delta_slice]
        d_pr = Xhat0_cur[:, :, self.delta_slice]
        Lexp = expression_loss(d_gt, d_pr)

        loss = Ls + self.hparams.lambda_vel * Lv + self.hparams.lambda_smooth * Lsm + self.hparams.lambda_exp * Lexp
        self.log_dict({"train/L_simple": Ls, "train/L_vel": Lv, "train/L_smooth": Lsm, "train/L_exp": Lexp,
                       "train/loss": loss}, prog_bar=True, on_step=True, on_epoch=True, batch_size=X0.size(0))
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int):
        X0 = batch["X0"].to(self.device)
        A = batch["A"].to(self.device)
        t = batch["t"].to(self.device).view(-1)

        Wpre, Wcur = self.hparams.Wpre, self.hparams.Wcur
        X_past = X0[:, :Wpre, :]
        X0_cur = X0[:, -Wcur:, :]
        A_full = A
        Xt_cur, _ = self.q_sample(X0_cur, t)
        Xhat0_cur = self.model(Xt_cur, A_full, X_past, t)

        Ls = simple_loss(X0_cur, Xhat0_cur)
        self.log("val/L_simple", Ls, prog_bar=True, batch_size=X0.size(0))
        return Ls

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20000)
        return {"optimizer": opt, "lr_scheduler": sched}
