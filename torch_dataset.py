from torch.utils.data import Dataset, DataLoader
import torch as th

class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, xs, ys):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        x = th.tensor(self.xs[idx])
        y = th.tensor(self.ys[idx], dtype = th.float32)

        sample = {'x': x, 'y': y}

        return sample