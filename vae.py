# vae.py
import torch
import torch.nn as nn

# -------- blocchi di utilità --------
class ConvGNAct(nn.Module):
    """Conv2d -> GroupNorm -> SiLU"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.gn   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class ResBlock(nn.Module):
    """Residual block con ConvGNAct ×2 e skip proiettato se cambia il canale"""
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.c1 = ConvGNAct(in_ch, out_ch, k=3, s=1, p=1, groups=groups)
        self.c2 = ConvGNAct(out_ch, out_ch, k=3, s=1, p=1, groups=groups)
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1))
    def forward(self, x):
        h = self.c1(x)
        h = self.c2(h)
        return h + self.skip(x)

# -------- Encoder / Decoder / VAE --------
class Encoder(nn.Module):
    def __init__(self, latent_dim: int, base_ch: int = 64, groups: int = 8):
        super().__init__()
        self.down1 = nn.Sequential(
            ConvGNAct(3, base_ch, k=4, s=2, p=1, groups=groups),   # 16x16
            ResBlock(base_ch, base_ch, groups=groups),
        )
        self.down2 = nn.Sequential(
            ConvGNAct(base_ch, base_ch*2, k=4, s=2, p=1, groups=groups),  # 8x8
            ResBlock(base_ch*2, base_ch*2, groups=groups),
        )
        self.down3 = nn.Sequential(
            ConvGNAct(base_ch*2, base_ch*4, k=4, s=2, p=1, groups=groups),  # 4x4
            ResBlock(base_ch*4, base_ch*4, groups=groups),
        )
        self.conv_mu     = nn.Conv2d(base_ch*4, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(base_ch*4, latent_dim, kernel_size=1)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        h = self.down3(x)
        mu     = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)  # stabilità numerica
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, base_ch: int = 64, groups: int = 8):
        super().__init__()
        self.in_proj = ConvGNAct(latent_dim, base_ch*4, k=1, s=1, p=0, groups=groups)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(groups, base_ch*2), num_channels=base_ch*2),
            nn.SiLU(inplace=True),
            ResBlock(base_ch*2, base_ch*2, groups=groups),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(groups, base_ch), num_channels=base_ch),
            nn.SiLU(inplace=True),
            ResBlock(base_ch, base_ch, groups=groups),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.in_proj(z)
        h = self.up1(h)
        h = self.up2(h)
        x = self.up3(h)
        return torch.sigmoid(x)

class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
