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
        label = torch.Tensor([0 if author != self.unique_authors[i] else 1
                              for i in range(len(self.unique_authors))])

        return feature, label

    def __len__(self):
        return self.file_count


class Model(nn.Module):
    def __init__(self, in_channels, out_features):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_features)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():

    # Path to save model to
    model_path = "model.pth"

    # Training Variables
    batch_size = 16
    shuffle = True
    epochs = 8
    num_workers = 1

    dataset = PaintingsDataset('catalog.csv', "images", restrict_form="painting")

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)

    model = Model(3, len(dataset.unique_authors)).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epoch_losses = []

    with tqdm(total=len(dataloader) * epochs) as pbar:
        for epoch in range(epochs):
            losses = []

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                pbar.set_description(f"Loss: {loss.item():.3f}")

                pbar.update(1)

            # Append average loss to epoch losses
            epoch_losses.append(sum(losses)/len(losses))

    torch.save(model, model_path)

    plt.plot(epoch_losses)
    plt.title('Average Loss per Epoch')
    plt.ylabel('Average Loss')
    plt.xlabel('Epoch')
    plt.savefig('Training_Loss.png')

if __name__ == "__main__":
    main()