from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random

device = torch.device(
    f'cuda:0' if torch.cuda.is_available() else
    'cpu'
)


def split_data(catalog, data_dir, split_ratio=0.9, min_entries_per_artist=1, restrict_form='painting'):
    _, _, data_files = next(os.walk(data_dir))

    # Read in catalog and restrict form if necessary
    df = pd.read_csv(catalog, encoding='latin')
    if restrict_form is not None:
        df = df[df.FORM == restrict_form]

    # Remove authors that have less than the minimum number of entries required for each author
    authors = df['AUTHOR'].unique()
    authors = [author for author in authors if len(df[df.AUTHOR == author]) >= min_entries_per_artist]

    # Remove files with authors that aren't in the list of approved authors
    data_files = [file for file in data_files if df['AUTHOR'][int(file.strip(".jpg"))] in authors]

    print('Dividing into train/test data...')

    split_num = int(len(data_files) * split_ratio)

    # Repeat until the number test authors are a subset of the train authors
    train_authors = {}
    test_authors = {0}
    while not test_authors.issubset(train_authors):
        shuffle = data_files.copy()
        random.shuffle(shuffle)
        train_files = shuffle[:split_num]
        test_files = shuffle[split_num:]

        train_authors = set([df['AUTHOR'][int(file.strip(".jpg"))] for file in train_files])
        test_authors = set([df['AUTHOR'][int(file.strip(".jpg"))] for file in test_files])

    train_dataset = PaintingsDataset(df, authors, data_dir, train_files)
    test_dataset = PaintingsDataset(df, authors, data_dir, test_files)

    return train_dataset, test_dataset


class PaintingsDataset(Dataset):
    def __init__(self, catalog, authors, image_dir, files):
        self.files = files
        self.image_dir = image_dir
        self.catalog = catalog
        self.authors = authors

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

    ### Training variables ###
    catalog_file = 'catalog.csv'
    data_directory = 'images_128'
    image_dimensions = (128, 128)
    # Will load model from this path if it exists, otherwise it will make a new model and save to this path
    model_path = None
    restrict_form = 'painting'
    min_entries_per_artist = 5
    # Ratio between train/test data, recommend keeping this one high
    split_ratio = 0.9
    epochs = 100
    converge = 0.005
    measure_loss = True
    measure_val_acc = True

    # PyTorch DataLoader Variables
    batch_size = 32
    shuffle = True
    num_workers = 1

    train_dataset, test_dataset = split_data(catalog_file,
                                             data_directory,
                                             split_ratio,
                                             min_entries_per_artist,
                                             restrict_form)

    print(f"Min Number of Works per Author: {min_entries_per_artist}")
    print(f"Number of Authors: {len(train_dataset.authors)}")
    print(f"Unique Works of Art: {len(train_dataset) + len(test_dataset)}")
    print(f"Size of Train Data Set: {len(train_dataset)}")
    print(f"Size of Test Data Set: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)

    val_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False)

    if model_path is not None and os.path.exists(model_path):
        model = torch.load(model_path).to(device)
    else:
        model = Model(3, len(train_dataset.authors), image_dimensions).to(device)

    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epoch_train_losses = []

    with tqdm(total=epochs) as pbar:
        while len(epoch_train_losses) == 0 or (epoch_train_losses[-1] > converge and len(epoch_train_losses) < epochs):
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

    if measure_val_acc:
        val_losses = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_losses.append(loss.item())

        print(f'Average loss over test set: {sum(val_losses)/len(val_losses)}')

    if model_path is not None:
        torch.save(model, model_path)

    if measure_loss:
        plt.plot(epoch_train_losses)
        plt.title('Average Cross-Entropy Loss per Epoch')
        plt.ylabel('Average Loss')
        plt.xlabel('Epoch')
        plt.savefig('Training_Loss.png')

if __name__ == "__main__":
    main()