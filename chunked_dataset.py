import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

class ChunkedNPYDataset(Dataset):
    def __init__(self, chunk_dir, clean_prefix="clean_chunk_", noisy_prefix="noisy_chunk_"):
        self.chunk_dir = chunk_dir

        self.clean_chunks = sorted(glob.glob(os.path.join(chunk_dir, f"{clean_prefix}*.npy")))
        self.noisy_chunks = sorted(glob.glob(os.path.join(chunk_dir, f"{noisy_prefix}*.npy")))
        assert len(self.clean_chunks) == len(self.noisy_chunks), "Chunk count mismatch"

        
        self.chunk_sizes = [np.load(f, mmap_mode='r').shape[0] for f in self.clean_chunks]

        
        self.index_map = []
        for chunk_idx, size in enumerate(self.chunk_sizes):
            for i in range(size):
                self.index_map.append((chunk_idx, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        chunk_idx, local_idx = self.index_map[index]

        
        clean_chunk = np.load(self.clean_chunks[chunk_idx], mmap_mode='r')
        noisy_chunk = np.load(self.noisy_chunks[chunk_idx], mmap_mode='r')

        clean = clean_chunk[local_idx]  # (40, 64, 64)
        noisy = noisy_chunk[local_idx]

        
        return {
            "clean": torch.from_numpy(clean).unsqueeze(0).float(),
            "noisy": torch.from_numpy(noisy).unsqueeze(0).float(),
        }
