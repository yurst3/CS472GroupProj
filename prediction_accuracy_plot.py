from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import numpy as np


bag_acc = np.load('type_restrict_accuracies.npy')*100
domain = np.arange(5,5*(bag_acc.shape[0]+1),5)

plt.style.use('ggplot')
plt.plot(domain, bag_acc[:,0], label='1 Model')
plt.plot(domain, bag_acc[:,1], label='3 Models')
plt.plot(domain, bag_acc[:,2], label='5 Models')
plt.plot(domain, bag_acc[:,3], label='7 Models')
plt.legend()
plt.title('Combining Multiple Models')
plt.xlabel('Min Entries Per Artist')
plt.ylabel('Prediction Accuracy %')
plt.savefig('combination_accuracy.png')