import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from project.model import GraphSRFlowModel
from project.dataset import JetNetSRDataset

def solve_ode_euler(model, lr_features, num_steps=50, device="cpu"):
    B, N, C = lr_features.shape
    x_t = torch.randn((B, N, C), device=device)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_val = step / num_steps
        t_tensor = torch.full((B,), t_val, device=device)
        with torch.no_grad():
            v_pred = model(x_t, lr_features, t_tensor)
        x_t = x_t + v_pred * dt
    return x_t

def plot_calorimeter_event(lr, hr, pred, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    titles = ["Low Resolution (Input)", "High Resolution (Target)", "Predicted (Ensemble Mean)"]
    data = [lr, hr, pred]
    
    for ax, event, title in zip(axes, data, titles):
        # JetNet features are [eta, phi, pT], so pT (energy) is index 2
        eta = event[:, 0]
        phi = event[:, 1]
        energy = event[:, 2] 
        
        scatter = ax.scatter(eta, phi, c=energy, s=(energy - energy.min() + 0.1)*50, 
                             cmap='viridis', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Pseudorapidity (Eta)")
        ax.set_ylabel("Azimuthal Angle (Phi)")
        plt.colorbar(scatter, ax=ax, label="Normalized pT")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Inference for JetNet Super-Resolution")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained PyTorch Lightning checkpoint")
    parser.add_argument("--num_ensemble", type=int, default=10, help="Number of forward passes to average")
    parser.add_argument("--ode_steps", type=int, default=50, help="Number of integration steps for the ODE solver")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Where to save predictions")
    parser.add_argument("--sample_idx", type=int, default=0, help="Which event index to visualize")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    model = GraphSRFlowModel.load_from_checkpoint(args.ckpt_path).to(device)
    model.eval()

    # 2. Load JetNet Test Sample
    print("Loading JetNet test data...")
    dataset = JetNetSRDataset(jet_type='t', split='test', max_particles=30, lr_ratio=0.5)
    
    lr_tensor, hr_tensor = dataset[args.sample_idx]
    lr_batch = lr_tensor.unsqueeze(0).to(device)
    hr_batch = hr_tensor.unsqueeze(0).to(device)

    # 3. Perform Ensemble Inference
    print(f"Running ODE solver {args.num_ensemble} times to build ensemble...")
    predictions = []
    for i in range(args.num_ensemble):
        pred = solve_ode_euler(model, lr_batch, num_steps=args.ode_steps, device=device)
        predictions.append(pred)
        
    stacked_preds = torch.stack(predictions)
    mean_pred = stacked_preds.mean(dim=0)

    # 4. Save Raw Tensors
    output_tensor_path = os.path.join(args.output_dir, f"prediction_event_{args.sample_idx}.pt")
    torch.save({
        "lr_input": lr_batch.cpu(),
        "hr_target": hr_batch.cpu(),
        "pred_mean": mean_pred.cpu()
    }, output_tensor_path)
    print(f"Tensors saved to {output_tensor_path}")

    # 5. Visualize
    plot_calorimeter_event(
        lr=lr_batch.squeeze(0).cpu().numpy(),
        hr=hr_batch.squeeze(0).cpu().numpy(),
        pred=mean_pred.squeeze(0).cpu().numpy(),
        save_path=os.path.join(args.output_dir, f"event_{args.sample_idx}_viz.png")
    )

if __name__ == "__main__":
    main()