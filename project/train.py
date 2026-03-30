import argparse
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Import your modules (adjust import paths based on your actual structure)

from project.model import GraphSRFlowModel
from project.dataset import JetNetSRDataset # Make sure it imports the new class

# Optimize matrix multiplication for modern Ampere+ GPUs
torch.set_float32_matmul_precision('medium')

def get_args():
    parser = argparse.ArgumentParser(description="Train Super-Resolution Flow Matching Model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the .root files")
    parser.add_argument("--train_file", type=str, default="single_e_train_split0.root", help="Training file name")
    parser.add_argument("--val_file", type=str, default="single_e_val.root", help="Validation file name")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size in model")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Where to save model weights")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # 1. Seed everything for reproducibility
    pl.seed_everything(args.seed, workers=True)
    
    # 2. Setup Datasets
    print("Loading datasets...")
    train_path = os.path.join(args.data_dir, args.train_file)
    val_path = os.path.join(args.data_dir, args.val_file)
    


    # Use Top-quark jets, 30 particles per jet, keeping top 15 for Low-Res input
    train_dataset = JetNetSRDataset(jet_type='t', split='train', max_particles=30, lr_ratio=0.5)
    val_dataset = JetNetSRDataset(jet_type='t', split='valid', max_particles=30, lr_ratio=0.5)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 3. Initialize Model
    print("Initializing model...")
    model = GraphSRFlowModel(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr
    )
    
    # 4. Setup Callbacks and Loggers
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="sr-model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,           # Keep the top 3 best models
        monitor="val_loss",     # Metric to track
        mode="min",             # We want to minimize validation loss
        save_last=True          # Always save the most recent epoch to resume if interrupted
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,            # Stop if val_loss doesn't improve for 10 epochs
        mode="min"
    )
    
    logger = TensorBoardLogger("tb_logs", name="flow_matching_sr")
    
    # 5. Configure Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",     # Automatically finds GPUs, MPS (Apple Silicon), or uses CPU
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="16-mixed",   # Mixed precision training for speed/memory efficiency
        log_every_n_steps=10
    )
    
    # 6. Train!
    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print(f"Training complete. Best model saved to: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()