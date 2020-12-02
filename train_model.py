from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

device = torch.device(
    f'cuda:0' if torch.cuda.is_available() else
    'cpu'
)

class PaintingsDataset(Dataset):
    def __init__(self, catalog_file, image_dir, min_entries_per_artist=2, restrict_form=None):
        # List files
        self.image_dir = image_dir
        _, _, self.files = next(os.walk(image_dir))

        # Read in catalog and restrict form if necessary
        self.catalog = pd.read_csv(catalog_file, encoding='latin')
        if restrict_form is not None:
            self.catalog = self.catalog[self.catalog.FORM == restrict_form]

        # List unique authors
        self.authors = self.catalog['AUTHOR'].unique()

        # Remove authors that have less than the minimum number of entries required for each author
        self.authors = [author for author in self.authors
                        if len(self.catalog[self.catalog.AUTHOR == author]) >= min_entries_per_artist]

        # Remove files with authors that aren't in the list of approved authors
        self.files = [file for file in self.files if self.catalog['AUTHOR'][int(file.strip(".jpg"))] in self.authors]

    def __getitem__(self, item):
        index = int(self.files[item].strip(".jpg"))
        author = self.catalog['AUTHOR'][index]

        file = self.image_dir + "/" + self.files[item]
        img = Image.open(file).convert('RGB')
        trans = transforms.ToTensor()

        # Feature and 1-hot encoded label
        feature = trans(img)

        '''
        # One-hot encoding (MSE Loss)
        label = torch.Tensor([0 if author != self.unique_authors[i] else 1
                              for i in range(len(self.unique_authors))])
        '''
        # Integer label (CrossEntropy Loss)
        label = [list(self.authors).index(author)]
        label = torch.Tensor(label).long()

        return feature, label

    def __len__(self):
        return len(self.files)


class Model(nn.Module):
    def __init__(self, in_channels, out_features, dimensions):
        super(Model, self).__init__()

        self.dimensions = dimensions

        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * int((((dimensions[0] - 4)/2)-4)/2) * int((((dimensions[1] - 4)/2)-4)/2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_features)
        self.soft = nn.Softmax(0)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = self.soft(x)
        return x

def main():
    print(device)

    # Path to save model to
    model_path = "model.pth"

    # Minimum number of artworks an artist needs to be included in the data set
    # This is to reduce the number of outlier artists with a small number of artworks
    min_entries_per_artist = 5

    # Dimensions of all the images_32 to be used
    image_dimensions = (128,128)

    # Training variables
    measure_loss = True
    measure_val_acc = True

    # PyTorch DataLoader Variables
    batch_size = 32
    shuffle = True
    epochs = 64
    num_workers = 1

    # TODO: Implement function that divides train/test images into separate directories
    dataset = PaintingsDataset('catalog.csv', "images_128",
                               min_entries_per_artist=min_entries_per_artist,
                               restrict_form="painting")
    print(f"Min Number of Works per Author: {min_entries_per_artist}")
    print(f"Unique Works of Art: {len(dataset)}")
    print(f"Number of Authors: {len(dataset.authors)}")

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)

    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = Model(3, len(dataset.authors), image_dimensions).to(device)

    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epoch_train_losses = []
    epoch_val_losses = []

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            losses = []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze(1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if measure_loss:
                    losses.append(loss.item())

            # Append average loss to epoch losses
            if measure_loss:
                epoch_train_losses.append(sum(losses)/len(losses))
                pbar.set_description(f"Avg Loss: {epoch_train_losses[-1]:.3f}")

            pbar.update(1)

    torch.save(model, model_path)

    if measure_loss:
        plt.plot(epoch_train_losses)
        plt.title('Average Cross-Entropy Loss per Epoch')
        plt.ylabel('Average Loss')
        plt.xlabel('Epoch')
        plt.savefig('Training_Loss.png')

if __name__ == "__main__":
    main()