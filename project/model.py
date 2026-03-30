import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

class SinusoidalTimeEmbedding(nn.Module):
    """Encodes the scalar time step t into a high-dimensional vector."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t shape: [Batch_size]
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GraphSRFlowModel(pl.LightningModule):
    def __init__(
        self, 
        input_dim=3,        # (E, eta, phi)
        hidden_dim=256, 
        num_heads=8, 
        num_layers=6, 
        lr=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # 1. Input Projections
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        
        # We concatenate x_t (noise/HR mix) and lr_features, so input is 2 * input_dim
        self.node_proj = nn.Linear(input_dim * 2, hidden_dim)

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. MLP Head (Predicts the vector field)
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, x_t, lr_features, t):
        """
        Args:
            x_t: Tensor of shape [B, N, input_dim] (The intermediate noisy state)
            lr_features: Tensor of shape [B, N, input_dim] (The conditioning upsampled LR context)
            t: Tensor of shape [B] (Time steps)
        Returns:
            Predicted vector field of shape [B, N, input_dim]
        """
        B, N, _ = x_t.shape

        # Embed time and expand to match node sequence length
        t_emb = self.time_embed(t)               # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1) # [B, N, hidden_dim]

        # Concatenate noisy state and LR conditioning
        x_concat = torch.cat([x_t, lr_features], dim=-1) # [B, N, input_dim * 2]
        
        # Project to hidden dimensions and add time embedding
        # (Adding time embedding to the node features acts as global temporal conditioning)
        h = self.node_proj(x_concat) + t_emb

        # Pass through Transformer
        # Transformer treats the nodes as a fully-connected graph (sequence without positional encoding)
        h = self.transformer(h)

        # Predict vector field v_theta
        v_pred = self.mlp_head(h)
        return v_pred

    def _shared_step(self, batch, batch_idx, step_type):
        """Shared logic for training and validation steps."""
        # Unpack batch: assuming lr_features is already upsampled to match hr_features shape
        lr_features, x_1 = batch 
        
        B, N, C = x_1.shape
        device = x_1.device

        # 1. Sample time step t ~ Uniform(0, 1)
        t = torch.rand(B, device=device)
        
        # Reshape t for broadcasting over [B, N, C]
        t_expand = t.view(B, 1, 1)

        # 2. Sample random noise x_0 ~ N(0, 1)
        x_0 = torch.randn_like(x_1)

        # 3. Compute interpolant x_t and target vector field u_t
        x_t = t_expand * x_1 + (1.0 - t_expand) * x_0
        u_t = x_1 - x_0

        # 4. Forward pass: predict vector field
        v_pred = self(x_t, lr_features, t)

        # 5. Compute Flow Matching Loss (MSE)
        loss = self.criterion(v_pred, u_t)

        self.log(f"{step_type}_loss", loss, prog_bar=True, batch_size=B)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Optional: Add a learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
