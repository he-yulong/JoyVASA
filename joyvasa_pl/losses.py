# joyvasa_pl/losses.py
import torch
import torch.nn.functional as F

def simple_loss(X0, Xhat0):  # L2
    return F.mse_loss(Xhat0, X0)

def velocity_loss(X0, Xhat0):
    # Δ over time
    dX0    = X0[:, 1:]    - X0[:, :-1]
    dXhat0 = Xhat0[:, 1:] - Xhat0[:, :-1]
    return F.mse_loss(dXhat0, dX0)

def smooth_loss(Xhat0):
    # second derivative
    v1 = Xhat0[:, 2:] - Xhat0[:, 1:-1]
    v0 = Xhat0[:, 1:-1] - Xhat0[:, :-2]
    acc = v1 - v0
    return (acc**2).mean()

def expression_loss(delta_gt, delta_pred):
    # L2 on the expression (δ) subvector only
    return F.mse_loss(delta_pred, delta_gt)
