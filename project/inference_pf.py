import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Import your model (adjust path as needed)
from project.train_pf import ParticleFlowTransformer

def reconstruct_particles(cells, incidence_matrix, pred_cardinality):
    """
    Reconstructs particle kinematics from cell features and the incidence matrix.
    
    Args:
        cells: [Num_Cells, 3] (Energy, Eta, Phi)
        incidence_matrix: [Num_Cells, Max_Particles] (Fractions summing to 1 over particles)
        pred_cardinality: Integer (Number of actual particles predicted)
        
    Returns:
        Predicted particles: [pred_cardinality, 4] -> (Energy, Eta, Phi, pT)
    """
    E_cells = cells[:, 0]
    eta_cells = cells[:, 1]
    phi_cells = cells[:, 2]
    
    num_max_particles = incidence_matrix.shape[1]
    reconstructed = []
    
    for k in range(num_max_particles):
        A_k = incidence_matrix[:, k] # Fractions for particle k
        
        # Energy weighted properties
        E_k = torch.sum(A_k * E_cells)
        
        # Avoid division by zero for empty particle slots
        if E_k > 1e-6:
            eta_k = torch.sum(A_k * E_cells * eta_cells) / E_k
            phi_k = torch.sum(A_k * E_cells * phi_cells) / E_k
            pT_k = E_k / torch.cosh(eta_k)
            reconstructed.append([E_k.item(), eta_k.item(), phi_k.item(), pT_k.item()])
        else:
            reconstructed.append([0.0, 0.0, 0.0, 0.0])
            
    reconstructed = np.array(reconstructed)
    
    # Sort by Energy (descending) and take the top `pred_cardinality` particles
    sorted_indices = np.argsort(reconstructed[:, 0])[::-1]
    top_particles = reconstructed[sorted_indices][:pred_cardinality]
    
    return top_particles

def calculate_pt(energy, eta):
    return energy / np.cosh(eta)

def hungarian_matching(pred_particles, true_particles):
    """
    Matches predicted particles to ground truth using the Hungarian algorithm.
    Cost is a combination of Delta R (spatial distance) and Energy difference.
    """
    N_pred = len(pred_particles)
    N_true = len(true_particles)
    
    if N_pred == 0 or N_true == 0:
        return []
    
    cost_matrix = np.zeros((N_pred, N_true))
    
    for i in range(N_pred):
        for j in range(N_true):
            # Delta R = sqrt(dEta^2 + dPhi^2)
            dEta = pred_particles[i, 1] - true_particles[j, 1]
            dPhi = pred_particles[i, 2] - true_particles[j, 2]
            
            # Handle Phi periodicity (-pi to pi)
            dPhi = (dPhi + np.pi) % (2 * np.pi) - np.pi
            dR = np.sqrt(dEta**2 + dPhi**2)
            
            # Energy difference fraction
            dE_frac = np.abs(pred_particles[i, 0] - true_particles[j, 0]) / (true_particles[j, 0] + 1e-6)
            
            # Custom cost function (tune weights as necessary)
            cost_matrix[i, j] = dR + 0.5 * dE_frac

    # Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Return matched pairs: [(pred_idx, true_idx), ...]
    return list(zip(row_ind, col_ind))

def plot_evaluation(pred_particles, true_particles, matches, save_path):
    """Generates evaluation plots for the matched particles."""
    if not matches:
        print("No matches to plot.")
        return
        
    res_E, res_pT, res_Eta, res_Phi = [], [], [], []
    
    for p_idx, t_idx in matches:
        p = pred_particles[p_idx]
        t = true_particles[t_idx]
        
        res_E.append((p[0] - t[0]) / (t[0] + 1e-6)) # (Pred - True) / True
        res_pT.append((p[3] - t[3]) / (t[3] + 1e-6))
        res_Eta.append(p[1] - t[1])
        
        dPhi = p[2] - t[2]
        res_Phi.append((dPhi + np.pi) % (2 * np.pi) - np.pi)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Particle Flow Reconstruction Residuals", fontsize=16)
    
    axs[0, 0].hist(res_E, bins=20, color='blue', alpha=0.7)
    axs[0, 0].set_title(r"Energy Residual Fraction $\Delta E / E$")
    
    axs[0, 1].hist(res_pT, bins=20, color='green', alpha=0.7)
    axs[0, 1].set_title(r"Transverse Momentum Residual Fraction $\Delta p_T / p_T$")
    
    axs[1, 0].hist(res_Eta, bins=20, color='red', alpha=0.7)
    axs[1, 0].set_title(r"Pseudorapidity Residual $\Delta \eta$")
    
    axs[1, 1].hist(res_Phi, bins=20, color='purple', alpha=0.7)
    axs[1, 1].set_title(r"Azimuthal Angle Residual $\Delta \phi$")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Evaluation plots saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to PFlow checkpoint")
    parser.add_argument("--sr_output_file", type=str, required=True, help="Path to the .pt file from SR inference")
    parser.add_argument("--output_dir", type=str, default="./pflow_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load PFlow Model
    model = ParticleFlowTransformer.load_from_checkpoint(args.ckpt_path).to(device)
    model.eval()

    # 2. Load the outputs from your Super-Resolution model (MODIFIED)
    print(f"Loading SR predictions from {args.sr_output_file}...")
    sr_data = torch.load(args.sr_output_file)
    
    pred_hr_cells = sr_data["pred_mean"].to(device) # Shape: [1, 30, 3]
    gt_hr_cells = sr_data["hr_target"].squeeze(0).numpy() # Ground truth for evaluation
    
    # Filter ground truth to only include real particles (where energy/pT > 0)
    gt_particles = gt_hr_cells[gt_hr_cells[:, 2] > 0] 
    
    # Format GT particles as [Energy, Eta, Phi, pT]. 
    # JetNet provides [Eta, Phi, pT]. We will approximate Energy = pT * cosh(Eta)
    E_gt = gt_particles[:, 2] * np.cosh(gt_particles[:, 0])
    gt_formatted = np.stack([E_gt, gt_particles[:, 0], gt_particles[:, 1], gt_particles[:, 2]], axis=-1)

    # 3. PFlow Inference
    with torch.no_grad():
        card_logits, inc_log_probs = model(pred_hr_cells)
        pred_cardinality = torch.argmax(card_logits, dim=-1)[0].item()
        incidence_matrix = torch.exp(inc_log_probs)[0] 
        
    print(f"PFlow Model predicted {pred_cardinality} particles.")

    # 4. Reconstruct Kinematics
    pred_particles = reconstruct_particles(
        cells=pred_hr_cells[0].cpu(),
        incidence_matrix=incidence_matrix.cpu(),
        pred_cardinality=pred_cardinality
    )
    
    # 5. Hungarian Matching against the real JetNet targets
    matches = hungarian_matching(pred_particles, gt_formatted)
    print(f"Found {len(matches)} matches between prediction and ground truth.")
    
    # 6. Visualization
    plot_evaluation(
        pred_particles, 
        gt_formatted, 
        matches, 
        save_path=os.path.join(args.output_dir, "pflow_residuals.png")
    )

if __name__ == "__main__":
    main()