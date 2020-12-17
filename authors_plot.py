from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

_, _, data_files = next(os.walk('images_128x128'))
df = pd.read_csv('catalog.csv', encoding='latin')
df = df[df.FORM == 'painting']
min_entries_per_artist = np.arange(5, 305, 5)

ratio = []
with tqdm(total=len(min_entries_per_artist)) as pbar:
    for min_entries in min_entries_per_artist:

        # Remove authors that have less than the minimum number of entries required for each author
        authors = df['AUTHOR'].unique()
        authors = [author for author in authors if len(df[df.AUTHOR == author]) >= min_entries]

        # Remove files with authors that aren't in the list of approved authors
        cur_files = [file for file in data_files if df['AUTHOR'][int(file.strip(".jpg"))] in authors]

        ratio.append(len(cur_files)/len(authors))

        pbar.update(1)

plt.style.use('ggplot')
plt.plot(min_entries_per_artist, ratio)
plt.title('Average Artworks per Artists vs Min Entries Per Artists')
plt.ylabel('Average Artworks per Artists')
plt.xlabel('Min Entries Per Artist')
plt.savefig('Artworks-Artists.png')