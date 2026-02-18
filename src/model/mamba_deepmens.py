import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing standard Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    from src.model.mamba_minimal import MambaMinimal as Mamba
    MAMBA_AVAILABLE = False
    print("⚠️ Mamba-SSM not found. Using pure PyTorch fallback (Slower).")

class MambaDeepMEns(nn.Module):
    """
    DeepMEns architecture with BiLSTM replaced by BiMamba (Mamba-2).
    SOTA Innovation: O(L) complexity, better long-range dependency.
    """
    def __init__(self,
                 seq_len=23,
                 num_shape_features=4,
                 dropout=0.3):
        super(MambaDeepMEns, self).__init__()

        # 1. SEQUENCE BRANCH
        # Multi-scale CNNs
        self.conv_seq_3 = nn.Conv1d(4, 64, kernel_size=3, padding=1)
        self.conv_seq_5 = nn.Conv1d(4, 64, kernel_size=5, padding=2)
        self.conv_seq_7 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.bn_seq = nn.BatchNorm1d(192)

        # BiMamba for global context
        # Mamba is causal (unidirectional), so we use two for bidirectional
        # Input dim: 192 (from CNNs)
        # We project to d_model for Mamba
        self.mamba_proj = nn.Linear(192, 128)

        self.mamba_fwd = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)
        self.mamba_bwd = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)

        # 2. SHAPE BRANCH
        self.conv_shape = nn.Conv1d(num_shape_features, 32, kernel_size=3, padding=1)
        self.bn_shape = nn.BatchNorm1d(32)
        self.pool_shape = nn.MaxPool1d(2)

        # 3. POSITION BRANCH
        self.pos_embedding = nn.Embedding(seq_len + 1, 32)
        # Replacing Pos-LSTM with small Mamba or just MLP?
        # Keeping BiLSTM here might be fine, but let's go Full Mamba for consistency
        self.mamba_pos = Mamba(d_model=32, d_state=16, d_conv=2, expand=2) # Small

        # 4. FUSION
        # Seq: 128*2 = 256 (BiMamba)
        # Shape: 32 (Max Pool)
        # Pos: 32 (Global Pool)

        self.fc1 = nn.Linear(256 + 32 + 32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x_seq, x_shape, x_pos):
        # --- Sequence Branch ---
        c3 = F.relu(self.conv_seq_3(x_seq))
        c5 = F.relu(self.conv_seq_5(x_seq))
        c7 = F.relu(self.conv_seq_7(x_seq))

        c_out = torch.cat([c3, c5, c7], dim=1)
        c_out = self.bn_seq(c_out) # (B, 192, L)

        # Permute for Mamba (B, L, C)
        x_mamba = c_out.permute(0, 2, 1)
        x_mamba = self.mamba_proj(x_mamba) # (B, L, 128)

        # BiMamba
        # Fwd
        out_fwd = self.mamba_fwd(x_mamba)
        # Bwd (flip, run, flip)
        out_bwd = self.mamba_bwd(torch.flip(x_mamba, [1]))
        out_bwd = torch.flip(out_bwd, [1])

        mamba_out = torch.cat([out_fwd, out_bwd], dim=2) # (B, L, 256)
        seq_feat, _ = torch.max(mamba_out, dim=1) # Global Max Pool

        # --- Shape Branch ---
        s_out = F.relu(self.conv_shape(x_shape))
        s_out = self.bn_shape(s_out)
        s_out = self.pool_shape(s_out)
        shape_feat, _ = torch.max(s_out, dim=2)

        # --- Position Branch ---
        p_emb = self.pos_embedding(x_pos) # (B, L, 32)
        pos_out = self.mamba_pos(p_emb)
        pos_feat, _ = torch.max(pos_out, dim=1)

        # --- Fusion ---
        fused = torch.cat([seq_feat, shape_feat, pos_feat], dim=1)

        out = self.activation(self.fc1(fused))
        out = self.dropout(out)
        out = self.activation(self.fc2(out))
        out = self.dropout(out)

        score = self.output(out)
        return torch.sigmoid(score)
