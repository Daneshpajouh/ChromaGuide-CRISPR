import torch
import torch.nn as nn
import torch.nn.functional as F

class RNAGenesisVAE(nn.Module):
    """
    VAE for compressing sgRNA sequences into a continuous latent space.
    Component 1 of RNAGenesis (Latent Diffusion).
    """
    def __init__(self, seq_len=23, latent_dim=256):
        super(RNAGenesisVAE, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # --- ENCODER ---
        # Input: (B, 4, 23)
        self.enc_conv1 = nn.Conv1d(4, 32, kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1) # Downsample -> 12
        self.enc_conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1) # Downsample -> 6

        self.flatten_dim = 128 * 6
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # --- DECODER ---
        self.dec_fc = nn.Linear(latent_dim, self.flatten_dim)

        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) # Up -> 12
        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0) # Up -> 23?
        # Note: Output padding calculation for 6->12->23 needs care.
        # 6 * 2 - 2*1 + 3 + op = 12 + 1 + op = 13?
        # ConvTranspose output = (L-1)*stride - 2*padding + kernel + output_padding
        # L_in=6. (5)*2 - 2 + 3 + op = 10-2+3+op = 11+op. Need 12. So op=1. Correct.
        # L_in=12. (11)*2 - 2 + 3 + op = 22-2+3+op = 23+op. Need 23. So op=0. Correct.

        self.dec_conv3 = nn.ConvTranspose1d(32, 4, kernel_size=3, stride=1, padding=1) # 23 -> 23

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(h.size(0), 128, 6)

        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        logits = self.dec_conv3(h) # (B, 4, 23)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar

def vae_loss_function(recon_logits, x, mu, logvar, beta_kl=0.4):
    """
    SOTA Loss: CE + beta * KL
    x: One-hot (B, 4, L) or Softmax probs
    recon_logits: (B, 4, L)
    """
    # Cross Entropy
    # x is one-hot. We want class indices for CE loss, OR use float target
    # Since x is float tensor from dataset wrapper, let's use BCEWithLogits or generic CE
    # Standard: argmax x to get indices

    target_indices = torch.argmax(x, dim=1) # (B, L)
    ce_loss = F.cross_entropy(recon_logits, target_indices, reduction='sum')

    # KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return ce_loss + beta_kl * kld_loss
