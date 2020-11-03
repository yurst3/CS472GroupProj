from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

device = torch.device(
    f'cuda:0' if torch.cuda.is_available() else
    'cpu'
)

class PaintingsDataset(Dataset):
    def __init__(self, catalog_file, image_dir, restrict_form=None):
        # List files
        self.image_dir = image_dir
        _, _, self.files = next(os.walk(image_dir))
        self.file_count = len(self.files)

        # Read in catalog and restrict form if necessary
        self.catalog = pd.read_csv(catalog_file, encoding='latin')
        if restrict_form is not None:
            self.catalog = self.catalog[self.catalog.FORM == restrict_form]
        self.catalog = self.catalog[:self.file_count]

        # List unique authors
        self.unique_authors = self.catalog['AUTHOR'].unique()

    def __getitem__(self, item):
        index = int(self.files[item].strip(".jpg"))
        author = self.catalog['AUTHOR'][index]

        file = self.image_dir + "/" + self.files[item]
        img = Image.open(file)
        trans = transforms.ToTensor()

        # Feature and 1-hot encoded label
        feature = trans(img)
        label = torch.FloatTensor([0 if author != self.unique_authors[i] else 1
                                   for i in range(len(self.unique_authors))])

        return feature, label

    def __len__(self):
        return self.file_count

def main():

    # Training Variables
    batch_size = 1
    shuffle = True
    epochs = 256
    num_workers = 4

    dataset = PaintingsDataset('catalog.csv', "images", restrict_form="painting")

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)

    with tqdm(total=len(dataset)) as pbar:
        for epoch in range(epochs):
            for feature, label in dataloader:
                feature, label = feature.to(device), label.to(device)

                pbar.update(1)


if __name__ == "__main__":
    main()