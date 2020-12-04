from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import numpy as np

min_entries_per_artist = np.arange(5, 305, 5)
accuracies = np.load('accuracies.npy', allow_pickle=True)

plt.style.use('ggplot')
plt.plot(min_entries_per_artist, accuracies[:,0], label='Stopping Loss: 1.0')
plt.plot(min_entries_per_artist, accuracies[:,1], label='Stopping Loss: 0.1')
plt.plot(min_entries_per_artist, accuracies[:,2], label='Stopping Loss: 0.01')
plt.plot(min_entries_per_artist, accuracies[:,3], label='Stopping Loss: 0.005')
plt.legend()
plt.title('Prediction Accuracy vs Min Entries Per Artist')
plt.xlabel('Min Entries Per Artist')
plt.ylabel('Prediction Accuracy %')
plt.savefig('Prediction_Accuracy.png')