import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Blocchi CNN coerenti (GroupNorm + SiLU) ---
def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )

def deconv_block(in_ch, out_ch, k=4, s=2, p=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )

# --- Encoder: 32x32x3 -> 4x4xD ---
class Encoder(nn.Module):
    def __init__(self, z_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3,   64, k=3, s=2, p=1),   # 16x16
            conv_block(64, 128, k=3, s=2, p=1),   # 8x8
            conv_block(128,256, k=3, s=2, p=1),   # 4x4
            conv_block(256, z_ch, k=3, s=1, p=1)  # 4x4 (D)
        )
    def forward(self, x):
        return self.net(x)

# --- Decoder: 4x4xD -> 32x32x3 ---
class Decoder(nn.Module):
    def __init__(self, z_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(z_ch, 256, k=3, s=1, p=1),   # 4x4
            deconv_block(256,128, k=4, s=2, p=1),   # 8x8
            deconv_block(128,64,  k=4, s=2, p=1),   # 16x16
            deconv_block(64, 32,  k=4, s=2, p=1),   # 32x32
            nn.Conv2d(32, 3, kernel_size=1)         # logits/regressione
        )
    def forward(self, zq):
        return self.net(zq)

# --- Vector Quantizer ---
class VectorQuantizer(nn.Module):
    def __init__(self, K=128, D=64, beta=0.25):
        super().__init__()
        self.K = K
        self.D = D
        self.beta = beta
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1.0 / K, 1.0 / K)

    @torch.no_grad()
    def _get_perplexity(self, counts):
        probs = counts.float() / max(1, int(counts.sum().item()))
        mask = probs > 0
        if mask.any():
            H = -(probs[mask] * probs[mask].log()).sum()
            return torch.exp(H).item()
        return 0.0

    def forward(self, z_e):
        B, D, H, W = z_e.shape
        z = z_e.permute(0,2,3,1).contiguous().view(-1, D)   # (BHW, D)
        e = self.embedding.weight                           # (K,D)

        # dist^2
        dist = (z**2).sum(dim=1, keepdim=True) \
             + (e**2).sum(dim=1).unsqueeze(0) \
             - 2 * z @ e.t()

        indices = torch.argmin(dist, dim=1)
        z_q = F.embedding(indices, e).view(B,H,W,D).permute(0,3,1,2).contiguous()

        # Losses
        loss_codebook = F.mse_loss(z_q, z_e.detach()) * D
        loss_commit   = F.mse_loss(z_e, z_q.detach()) * D
        loss_vq = loss_codebook + self.beta * loss_commit

        z_q_st = z_e + (z_q - z_e).detach()

        with torch.no_grad():
            counts = torch.bincount(indices, minlength=self.K)
            stats = {
                "perplexity": self._get_perplexity(counts),
                "codes_used": int((counts > 0).sum().item()),
                "usage_ratio": float((counts > 0).sum().item()) / self.K,
                "avg_dist2": float(dist.min(dim=1).values.mean().item()),
            }

        return z_q_st, loss_vq, stats, indices.view(B,H,W)

# --- VQ-VAE completo ---
class VQVAE(nn.Module):
    def __init__(self, K=128, D=64, beta=0.25):
        super().__init__()
        self.encoder = Encoder(D)
        self.vq      = VectorQuantizer(K=K, D=D, beta=beta)
        self.decoder = Decoder(D)
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, loss_vq, stats, idx = self.vq(z_e)
        x_rec = self.decoder(z_q)
        return x_rec, loss_vq, stats, idx
