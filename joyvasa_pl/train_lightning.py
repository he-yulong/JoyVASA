# joyvasa_pl/train_lightning.py
import os, torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from data_module import MotionAudioDataModule
from lightning_module import DiffusionMotionLightning

def main():
    # TODO: load your preprocessed lists -> train_items, val_items  (each item: (X0, A) tensors)
    train_items, val_items = load_your_items()  # <-- implement

    dm = MotionAudioDataModule(train_items, val_items,
                               batch_size=16, num_workers=8, Wpre=25, Wcur=100, t_max=1000)

    model = DiffusionMotionLightning(
        Dm=256, Da=768, d_model=512, nhead=8, nlayers=6,
        Wpre=25, Wcur=100, lr=1e-4, betas=(1e-4, 0.02),
        lambda_vel=5.0, lambda_smooth=0.5, lambda_exp=0.1, max_t=1000
    )

    ckpt = ModelCheckpoint(monitor="val/L_simple", mode="min", save_top_k=3, save_last=True)
    lrmon = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger("lightning_logs", name="joyvasa_stage2")

    trainer = pl.Trainer(
        max_steps=20000,
        precision="bf16-mixed",  # "16-mixed" on older GPUs
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,       # bump if you need to fit memory
        log_every_n_steps=25,
        val_check_interval=1000,
        callbacks=[ckpt, lrmon],
        logger=logger,
        devices=1, accelerator="gpu", strategy="auto"
    )

    torch.set_float32_matmul_precision("high")
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
