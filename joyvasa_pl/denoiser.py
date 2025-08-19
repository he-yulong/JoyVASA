import torch, math
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model*4), nn.SiLU(), nn.Linear(d_model*4, d_model))

    def forward(self, t: torch.Tensor, d_model:int) -> torch.Tensor:
        # sinusoidal t -> project
        half = d_model // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)

class TransformerDenoiser(nn.Module):
    """
    Predicts clean motion X_hat0 for the current window (Wcur, Dm),
    conditioned on [past motion (Wpre, Dm), audio (Wpre+Wcur, Da), time t].
    Implements the compact conditioning used in the paper (§3.3.1). :contentReference[oaicite:5]{index=5}
    """
    def __init__(self, Dm=256, Da=768, d_model=512, nhead=8, nlayers=6, Wpre=25, Wcur=100):
        super().__init__()
        self.Wpre, self.Wcur = Wpre, Wcur
        self.t_proj = TimeEmbedding(d_model)

        self.in_proj = nn.Linear(Dm + Da, d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4)
        self.decoder = nn.TransformerDecoder(layer, num_layers=nlayers)
        self.out = nn.Linear(d_model, Dm)

        # learnable start tokens for initial window (A_start, X_start)
        self.A_start = nn.Parameter(torch.zeros(1, 1, Da))
        self.X_start = nn.Parameter(torch.zeros(1, 1, Dm))

    def forward(self, Xt_cur, A_full, X_past, t):
        """
        Xt_cur: (B, Wcur, Dm)  noisy current motion
        A_full: (B, Wpre+Wcur, Da)
        X_past:(B, Wpre, Dm)   clean past
        t:      (B,)           timestep
        """
        B = Xt_cur.size(0)
        # replace missing prefix with learnable tokens if sequence shorter
        if X_past.size(1) == 0:
            X_past = self.X_start.expand(B, 1, -1)
        if A_full.size(1) == 0:
            A_full = self.A_start.expand(B, 1, -1)

        # concat cond: [X_past, A_full]
        cond = torch.cat([X_past, A_full], dim=1)                 # (B, Wpre+Wpre+Wcur?, but A_full already includes past+cur)
        # Query is current window tokens built from [Xt_cur || A_cur]
        A_cur = A_full[:, -Xt_cur.size(1):, :]
        q = torch.cat([Xt_cur, A_cur], dim=-1)
        q = self.in_proj(q)                                       # (B, Wcur, d_model)
        k = self.in_proj(torch.cat([cond[:, -A_full.size(1):, :A_full.size(-1)],], dim=-1)) if False else self.in_proj(torch.cat([torch.zeros_like(A_full[..., :0]), A_full], dim=-1))  # simplify: keys from audio only

        # add time embedding
        t_emb = self.t_proj(t, q.size(-1)).unsqueeze(1)           # (B,1,d_model)
        q = q + t_emb

        # cross-decoder: queries attend to conditioning
        # (We keep it simple; you can add causal masks if needed.)
        h = self.decoder(tgt=q, memory=self.in_proj(torch.cat([X_past, A_full], dim=-1)))
        Xhat0_cur = self.out(h)                                   # (B, Wcur, Dm)
        return Xhat0_cur
