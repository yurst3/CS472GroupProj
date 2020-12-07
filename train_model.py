from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np

from itertools import combinations
from math import comb

device = torch.device(
    f'cuda:0' if torch.cuda.is_available() else
    'cpu'
)

print(device)

### Training variables ###
catalog_file = 'catalog.csv'
data_directories = ['images_128x128']
image_dimensions = [(128, 128)]
restrict_form = 'painting'
constant_val = True
min_entries_per_artist = np.arange(300, 305, 5)
# Ratio between train/test data, recommend keeping this one high
train_val_ratio = 0.9
train_fraction = 0.9
epochs = 1000
converge = 0.01
# Number of models to train
model_num = 11

# PyTorch DataLoader Variables
batch_size = 64
shuffle = True
num_workers = 1


class DataSplitter:
    def __init__(self, catalog, data_dir, restrict_form='painting', const_val=False):
        self.catalog = catalog
        self.data_dir = data_dir
        self.restrict_form = restrict_form
        self.const_val = const_val

        # Make a list of all files in the data directory
        _, _, self.data_files = next(os.walk(self.data_dir))

        # Read in catalog and restrict form if necessary
        self.df = pd.read_csv(self.catalog, encoding='latin')
        if self.restrict_form is not None:
            self.df = self.df[self.df.FORM == self.restrict_form]

        # Get all unique authors from the dataframe
        self.unique_authors = self.df['AUTHOR'].unique()

    def reduce_data(self, min_entries_per_artist):
        # Remove authors that have less than the minimum number of entries required for each author
        self.unique_authors = [author for author in self.unique_authors if
                               len(self.df[self.df.AUTHOR == author]) >= min_entries_per_artist]

        # Remove files with authors that aren't in the list of approved authors
        self.data_files = [file for file in self.data_files if
                           self.df['AUTHOR'][int(file.strip(".jpg"))] in self.unique_authors]

    # MUST BE CALLED BEFORE get_train()
    def get_val(self, split_ratio):
        split_num = int(len(self.data_files) * split_ratio)

        # Repeat until the number test authors are a subset of the train authors
        self.train_authors = {}
        self.val_authors = {0}
        while not self.val_authors.issubset(self.train_authors):
            shuffle = self.data_files.copy()
            random.shuffle(shuffle)
            self.train_files = shuffle[:split_num]
            self.val_files = shuffle[split_num:]

            self.train_authors = set([self.df['AUTHOR'][int(file.strip(".jpg"))] for file in self.train_files])
            self.val_authors = set([self.df['AUTHOR'][int(file.strip(".jpg"))] for file in self.val_files])

        return PaintingsDataset(self.df, self.unique_authors, self.data_dir, self.val_files)

    # MUST BE CALLED AFTER get_val()
    def get_train(self, train_fraction):
        split_num = int(len(self.train_files) * train_fraction)

        frac_authors = {}

        # Repeat until the fraction authors are a subset of the train authors
        while not self.val_authors.issubset(frac_authors):
            shuffle = self.train_files.copy()
            random.shuffle(shuffle)
            fraction = self.train_files[:split_num]

            frac_authors = set([self.df['AUTHOR'][int(file.strip(".jpg"))] for file in fraction])

        train_dataset = PaintingsDataset(self.df, self.unique_authors, self.data_dir, fraction)

        return train_dataset


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

    comb_sum = sum([comb(model_num, x) for x in range(1, model_num+2, 2)])
    with tqdm(total=len(image_dimensions) * len(min_entries_per_artist)*comb_sum) as pbar:
        for dimensions, data_directory in zip(image_dimensions, data_directories):

            splitter = DataSplitter(catalog=catalog_file,
                                    data_dir=data_directory,
                                    restrict_form=restrict_form,
                                    const_val=constant_val)
            accuracies = []
            for min_entries in min_entries_per_artist:

                pbar.set_description(
                    f"D {dimensions[0]}x{dimensions[1]}, ME {min_entries}, reducing")
                # Reduce the total number of authors and files to match the new minimum entries
                # Return a validation dataset that is 1/10 of the total data
                splitter.reduce_data(min_entries)
                pbar.set_description(
                    f"D {dimensions[0]}x{dimensions[1]}, ME {min_entries}, splitting val")
                val_dataset = splitter.get_val(train_val_ratio)
                val_loader = DataLoader(val_dataset,
                                        batch_size=len(val_dataset),
                                        shuffle=False)

                # Create all models
                models = [Model(3, len(splitter.unique_authors), dimensions).to(device) for i in range(model_num)]

                # Train each model
                for model_index, model in enumerate(models):
                    pbar.set_description(
                        f"D {dimensions[0]}x{dimensions[1]}, ME {min_entries}, M {model_index + 1}, splitting train")
                    train_dataset = splitter.get_train(train_fraction)

                    train_loader = DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

                    epoch_train_losses = []

                    while len(epoch_train_losses) == 0 or (epoch_train_losses[-1] > converge and len(epoch_train_losses) < epochs):
                        losses = []

                        for inputs, target in train_loader:
                            inputs, target = inputs.to(device), target.to(device)
                            target = target.squeeze(1)

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward + backward + optimize
                            outputs = model(inputs)
                            loss = criterion(outputs, target)
                            loss.backward()
                            optimizer.step()

                            losses.append(loss.item())

                        epoch_train_losses.append(sum(losses)/len(losses))
                        pbar.set_description(f"D {dimensions[0]}x{dimensions[1]}, ME {min_entries}, M {model_index+1}, AL {epoch_train_losses[-1]:.3f}, E {len(epoch_train_losses)}")

                # Validate accuracies for all combinations of models from 1 to bag_num
                with torch.no_grad():
                    model_num_acc = []
                    for i in range(1, model_num+2, 2):
                        combo_acc = []

                        pbar.set_description(
                            f"D {dimensions[0]}x{dimensions[1]}, ME {min_entries}, Validating {i} models")

                        # Iterate over all combinations of model nums and calculate the accuracy for each
                        for combo in combinations(models, i):

                            # Predict for each model in the combination
                            probs = []
                            for model in combo:
                                for inputs, target in val_loader:
                                    inputs, target = inputs.to(device), target.to(device)
                                    target = target.squeeze(1)

                                    out = model(inputs)
                                    probs.append(torch.softmax(out, dim=1))

                            # Sum all of the probabilities together
                            prob_sum = torch.sum(torch.stack(probs, dim=0), dim=0)

                            # Calculate the accuracy for the probability sum
                            winners = prob_sum.argmax(dim=1)
                            corrects = (winners == target)
                            accuracy = corrects.sum().float() / float(target.size(0))

                            # Append the accuracy of this combination to the combination accuracies
                            combo_acc.append(accuracy.item())
                            pbar.update(1)
                        # Append average of all combination accuracies to model number accuracies
                        model_num_acc.append(sum(combo_acc)/len(combo_acc))
                    # Append all model number accuracies to accuracies arrays
                    accuracies.append(model_num_acc)

            accuracies = np.array(accuracies)
            np.save(f'300_bag_accuracies.npy', accuracies)

if __name__ == "__main__":
    main()