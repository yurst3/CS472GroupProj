from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import random

class PaintingsDataset(Dataset):
    def __init__(self, catalog_file, image_dir):
        self.catalog = pd.read_csv(catalog_file, encoding='latin')

        _, _, self.files = next(os.walk(image_dir))
        self.file_count = len(self.files)

    def __getitem__(self, item):
        return self.files[item]

def main():
    batch_size = 16
    shuffle = True
    epochs = 256
    num_workers = 4

    dataset = PaintingsDataset('catalog.csv', "images")

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)

if __name__ == "__main__":
    main()