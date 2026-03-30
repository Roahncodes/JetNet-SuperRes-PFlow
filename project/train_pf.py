import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# 1. PyTorch Lightning Model
# -----------------------------------------------------------------------------
class ParticleFlowTransformer(pl.LightningModule):
    def __init__(
        self, 
        input_dim=3,           # e.g., (Energy, eta, phi)
        hidden_dim=256, 
        num_layers=6, 
        num_heads=8, 
        max_particles=50,      # Max expected particles per event (for cardinality classes)
        lr=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # 1. Input embedding for high-resolution cells
        self.cell_embed = nn.Linear(input_dim, hidden_dim)

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

        # 3. Learnable Particle Queries (similar to DETR/MaskFormer)
        # These represent the potential particles we are trying to reconstruct
        self.query_embed = nn.Embedding(max_particles, hidden_dim)

        # 4. Head 1: Cardinality Prediction
        # Pools the encoder output to predict the total number of particles in the event.
        self.cardinality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_particles + 1) # Classes: 0 to max_particles - 1
        )

        # 5. Head 2: Incidence Matrix Projections (Cross-Attention)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cell_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, cells):
        """
        Args:
            cells: Tensor of shape [Batch, Num_Cells, Input_Dim]
        Returns:
            cardinality_logits: [Batch, Max_Particles]
            incidence_log_probs: [Batch, Num_Cells, Max_Particles]
        """
        B, N, _ = cells.shape

        # --- Encode Cells ---
        x = self.cell_embed(cells)
        encoded_cells = self.transformer(x) # [B, N, hidden_dim]

        # --- Head 1: Cardinality ---
        # Global average pooling over the cell dimension
        pooled_cells = encoded_cells.mean(dim=1) # [B, hidden_dim]
        cardinality_logits = self.cardinality_head(pooled_cells) # [B, max_particles]

        # --- Head 2: Incidence Matrix ---
        # Generate queries for this batch
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1) # [B, max_particles, hidden_dim]

        # Project for attention
        q = self.query_proj(queries)       # [B, max_particles, hidden_dim]
        k = self.cell_proj(encoded_cells)  # [B, N, hidden_dim]

        # Compute similarity (scaled dot-product)
        # Resulting shape: [B, N, max_particles]
        attn_scores = torch.bmm(k, q.transpose(1, 2)) / (self.hparams.hidden_dim ** 0.5)

        # Apply log_softmax over the particle dimension.
        # This enforces that the sum of fractional assignments of a single cell 
        # to all possible particles equals 1.
        incidence_log_probs = F.log_softmax(attn_scores, dim=-1)

        return cardinality_logits, incidence_log_probs

    def _shared_step(self, batch, batch_idx, step_type):
        cells, target_cardinality, target_incidence = batch

        # Forward pass
        pred_cardinality_logits, pred_incidence_log_probs = self(cells)

        # --- Loss 1: Cross Entropy for Cardinality ---
        loss_card = F.cross_entropy(pred_cardinality_logits, target_cardinality)

        # --- Loss 2: KL Divergence for Incidence Matrix ---
        # F.kl_div expects log-probabilities as input and standard probabilities as target.
        # target_incidence shape: [B, N, max_particles]
        loss_inc = F.kl_div(
            pred_incidence_log_probs, 
            target_incidence, 
            reduction='batchmean'
        )

        # Total Loss (you can weight these if one dominates)
        loss = loss_card + loss_inc

        # Logging
        self.log(f"{step_type}_loss", loss, prog_bar=True)
        self.log(f"{step_type}_loss_card", loss_card)
        self.log(f"{step_type}_loss_inc", loss_inc)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

# -----------------------------------------------------------------------------
# 2. Data Preparation (Dummy Data for immediate execution)
# -----------------------------------------------------------------------------
from project.dataset import JetNetSRDataset

# 1. Move the collate function OUTSIDE so it is a standalone, top-level function
def pflow_collate(batch):
    # batch is a list of (lr_features, hr_features)
    cells = torch.stack([item[1] for item in batch]) # Use HR as input cells
    
    B, N, C = cells.shape
    
    # Target Cardinality: count how many particles have non-zero pT
    target_cardinality = (cells[:, :, 2] > 0).sum(dim=1).long()
    
    # Target Incidence Matrix: Identity Matrix
    target_incidence = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
    
    return cells, target_cardinality, target_incidence


# 2. Your dataloader function just references it now
def get_jetnet_pflow_dataloaders(batch_size=32):
    """Adapts the JetNet dataset for Particle Flow training."""
    from project.dataset import JetNetSRDataset
    
    train_dataset = JetNetSRDataset(jet_type='t', split='train', max_particles=30)
    val_dataset = JetNetSRDataset(jet_type='t', split='valid', max_particles=30)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pflow_collate, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pflow_collate, num_workers=4, persistent_workers=True)
    
    return train_loader, val_loader

# -----------------------------------------------------------------------------
# 3. Main Training Script
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--ckpt_dir", type=str, default="./pflow_checkpoints")
    args = parser.parse_args()

    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium') # Speeds up training on Ampere+ GPUs

    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader = get_jetnet_pflow_dataloaders(batch_size=args.batch_size)

    # Init Model
    model = ParticleFlowTransformer(
        lr=args.lr, 
        max_particles=30 # <--- FORCE IT TO 30 HERE
    )

    # 3. Callbacks & Logger
    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="pflow-{epoch:02d}-{val_loss:.4f}",
        save_top_k=2,
        monitor="val_loss",
        mode="min"
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    logger = TensorBoardLogger("tb_logs", name="pflow_transformer")

    # 4. Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="16-mixed",
        log_every_n_steps=5
    )

    # 5. Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()