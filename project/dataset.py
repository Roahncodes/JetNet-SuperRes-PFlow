import torch
from torch.utils.data import Dataset
import jetnet

class JetNetSRDataset(Dataset):
    """
    PyTorch Dataset for Super-Resolution using the JetNet dataset.
    Automatically downloads and processes the data.
    """
    
    def __init__(self, jet_type='t', split='train', max_particles=30, lr_ratio=0.5):
        """
        Args:
            jet_type (str): Type of jet ('g', 'q', 'w', 'z', 't'). 't' = Top quark.
            split (str): 'train', 'valid', or 'test'.
            max_particles (int): Max number of particles per jet (usually 30 or 150).
            lr_ratio (float): Fraction of particles to keep for the Low-Resolution input.
        """
        super().__init__()
        
        # 1. Automatically fetch data (downloads to ./datasets by default if not present)
        print(f"Loading JetNet {jet_type}-jets ({split} split)...")
        self.dataset = jetnet.datasets.JetNet(
            particle_features=["etarel", "phirel", "ptrel"], # Features: Eta, Phi, pT
            num_particles=max_particles,
            jet_type=jet_type,
            split=split,
            download=True  # <-- This tells it to fetch the data from Zenodo
        )
        
        # JetNet stores data as a numpy array in .particle_data
        self.particle_data = self.dataset.particle_data
        self.num_events = len(self.particle_data)
        
        self.max_particles = max_particles
        self.lr_size = int(self.max_particles * lr_ratio)

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        # High-Resolution Target: Shape [max_particles, 3]
        hr_features = torch.tensor(self.particle_data[idx], dtype=torch.float32)
        
        # Low-Resolution Input: 
        # In JetNet, particles are sorted by pT (energy) descending.
        # To simulate a low-resolution detector, we keep the top `lr_size` highest-energy 
        # particles and zero out the rest (acting as our "upsampled" low-res input).
        lr_features = hr_features.clone()
        lr_features[self.lr_size:] = 0.0 
        
        return lr_features, hr_features